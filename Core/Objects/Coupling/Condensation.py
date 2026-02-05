import numpy as np

from .BaseCoupling import BaseCoupling


class Condensation(BaseCoupling):
    """
    Condensation-based coupling via DOF elimination.

    Eliminates coupled FEM DOFs by expressing them in terms of block DOFs
    using rigid body kinematics. Produces a reduced system: K_red = T'*K*T.

    Attributes
    ----------
    transformation_matrix : ndarray
        T matrix (n_full x n_reduced) such that u_full = T @ u_reduced
    nb_dofs_full : int
        Original DOF count before reduction
    nb_dofs_reduced : int
        Reduced DOF count after elimination
    """

    def __init__(self):
        super().__init__(coupling_type='constraint')
        self.transformation_matrix = None
        self.nb_dofs_full = None
        self.nb_dofs_reduced = None

    def get_system_modification_type(self) -> str:
        return 'matrix_reduction'

    def get_dof_count_change(self, n_base_dofs: int) -> int:
        return -2 * len(self.coupled_nodes)

    def requires_special_solver(self) -> bool:
        return False

    def build_transformation(self, structure):
        """
        Build transformation matrix T for DOF elimination.

        Enforces: u_full = T @ u_reduced

        For coupled FEM nodes: u_fem = C @ u_block (rigid body)
        For uncoupled nodes: identity mapping
        """
        if not self.coupled_nodes:
            return

        n_dofs_full = structure.nb_dofs
        n_blocks = len(structure.list_blocks)

        # Identify uncoupled FEM nodes
        all_fem_node_ids = set(range(n_blocks, len(structure.list_nodes)))
        coupled_fem_node_ids = set(self.coupled_nodes.keys())
        free_fem_node_ids = sorted(all_fem_node_ids - coupled_fem_node_ids)

        # Count reduced DOFs
        n_dofs_reduced = sum(structure.node_dof_counts[structure.list_blocks[i].connect]
                            for i in range(n_blocks))
        n_dofs_reduced += sum(structure.node_dof_counts[node_id]
                             for node_id in free_fem_node_ids)

        self.nb_dofs_reduced = n_dofs_reduced
        self.nb_dofs_full = n_dofs_full

        # Build transformation matrix T (n_dofs_full × n_dofs_reduced)
        T = np.zeros((n_dofs_full, n_dofs_reduced))

        # Map block DOFs (identity - blocks are always independent)
        reduced_dof_counter = 0
        for block_idx in range(n_blocks):
            block_node_id = structure.list_blocks[block_idx].connect
            node_dof_count = structure.node_dof_counts[block_node_id]
            base_global_dof = structure.node_dof_offsets[block_node_id]

            for dof_idx in range(node_dof_count):
                global_dof = base_global_dof + dof_idx
                reduced_dof = reduced_dof_counter + dof_idx
                T[global_dof, reduced_dof] = 1.0

            reduced_dof_counter += node_dof_count

        # Map coupled FEM nodes (constraint matrix)
        for fem_node_id, block_idx in self.coupled_nodes.items():
            block = structure.list_blocks[block_idx]
            block_node_id = block.connect
            fem_node_pos = structure.list_nodes[fem_node_id]

            # Get constraint matrix from stored values
            C = self.constraint_matrices[(fem_node_id, block_idx)]

            # Get DOF counts
            fem_node_dof_count = structure.node_dof_counts[fem_node_id]
            block_dof_count = structure.node_dof_counts[block_node_id]
            base_fem_global_dof = structure.node_dof_offsets[fem_node_id]

            # Compute reduced DOF offset for this block
            block_reduced_dof_offset = sum(
                structure.node_dof_counts[structure.list_blocks[i].connect]
                for i in range(block_idx))

            # Map FEM DOFs through constraint
            for local_dof in range(min(fem_node_dof_count, 2)):  # FEM: u, v
                global_dof = base_fem_global_dof + local_dof

                # Express in terms of block's reduced DOFs
                for block_dof_idx in range(block_dof_count):  # Block: u, v, θ
                    reduced_dof = block_reduced_dof_offset + block_dof_idx
                    T[global_dof, reduced_dof] = C[local_dof, block_dof_idx]

        # Map uncoupled FEM nodes (identity)
        for fem_node_id in free_fem_node_ids:
            fem_node_dof_count = structure.node_dof_counts[fem_node_id]
            base_fem_global_dof = structure.node_dof_offsets[fem_node_id]

            for dof_idx in range(fem_node_dof_count):
                global_dof = base_fem_global_dof + dof_idx
                reduced_dof = reduced_dof_counter + dof_idx
                T[global_dof, reduced_dof] = 1.0

            reduced_dof_counter += fem_node_dof_count

        self.transformation_matrix = T

    def reduce_system(self, K_full: np.ndarray, P_full: np.ndarray):
        """
        Transform to reduced system: K_red = T' @ K @ T, P_red = T' @ P.
        """
        if self.transformation_matrix is None:
            raise RuntimeError("Transformation matrix not built. Call build_transformation() first.")

        T = self.transformation_matrix
        K_reduced = T.T @ K_full @ T
        P_reduced = T.T @ P_full

        return K_reduced, P_reduced

    def expand_solution(self, u_reduced: np.ndarray) -> np.ndarray:
        """Expand solution to full DOF space: u_full = T @ u_reduced."""
        if self.transformation_matrix is None:
            raise RuntimeError("Transformation matrix not built.")
        return self.transformation_matrix @ u_reduced

    def verify_constraints(self, structure, u_full: np.ndarray) -> float:
        """
        Verify constraint satisfaction after solution.

        Parameters
        ----------
        structure : Hybrid
            The structure object.
        u_full : ndarray (n,)
            Full displacement vector.

        Returns
        -------
        max_error : float
            Maximum constraint violation (L2 norm per node).
        """
        max_error = 0.0

        for fem_node_id, block_idx in self.coupled_nodes.items():
            # Get DOF indices
            fem_dofs = structure.node_dof_offsets[fem_node_id]
            u_fem = u_full[fem_dofs: fem_dofs + 2]  # Assuming 2D

            block = structure.list_blocks[block_idx]
            block_node_id = block.connect
            block_dofs = structure.node_dof_offsets[block_node_id]
            q_block = u_full[block_dofs: block_dofs + 3]

            # Get constraint matrix C stored in self.constraint_matrices
            C = self.constraint_matrices[(fem_node_id, block_idx)]

            # Constraint residual: should be zero
            # u_fem = C @ q_block => residual = u_fem - C @ q_block
            residual = u_fem - C @ q_block
            max_error = max(max_error, np.linalg.norm(residual))

        return max_error
