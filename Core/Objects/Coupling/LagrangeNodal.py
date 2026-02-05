from typing import Dict, Tuple

import numpy as np

from .BaseCoupling import BaseCoupling


class LagrangeCoupling(BaseCoupling):
    """
    Lagrange multiplier coupling via augmented saddle point system.

    Enforces coupling constraints exactly using Lagrange multipliers:

        [K   C'] [u]   [P]
        [C   0 ] [l] = [0]

    The multipliers l represent physical constraint forces at the interface.

    Attributes
    ----------
    constraint_matrix_C : ndarray
        Constraint matrix C (n_constraints x n_dofs)
    n_constraints : int
        Total constraints (2 per coupled node)
    multipliers : ndarray
        Lagrange multipliers (interface forces) from solution
    """

    def __init__(self):
        """
        Initialize LagrangeCoupling.
        """
        super().__init__(coupling_type='lagrange')

        # Constraint matrix and system dimensions
        self.constraint_matrix_C = None
        self.n_constraints = 0
        self.n_dofs_base = None
        self.n_dofs_augmented = None

        # Solution storage
        self.multipliers = None  # λ values (constraint forces)

        # DOF mapping for constraint construction
        self.dof_map = {}  # Maps (fem_node_id, block_idx) -> (fem_dofs, block_dofs)

    def get_system_modification_type(self) -> str:
        return 'matrix_augmentation'

    def get_dof_count_change(self, n_base_dofs: int) -> int:
        return 2 * len(self.coupled_nodes)

    def requires_special_solver(self) -> bool:
        return True

    def build_constraint_matrix(self, structure) -> np.ndarray:
        """
        Build constraint matrix C for linear constraints.

        Supports:
        1. FEM-to-Block: Rigid body kinematics (offset + rotation).
        2. FEM-to-FEM:   Direct node tying (u1=u2).
        """
        if not self.coupled_nodes:
            # It's possible no block-fem nodes were found, but maybe fem-fem nodes were added?
            # If we want to support pure FEM-FEM, we need a way to register them.
            # For now, assume coupled_nodes handles both or we add a new list.
            pass

        # Get system dimensions
        self.n_dofs_base = structure.nb_dofs

        # Count constraints
        # Block couplings: 2 per node (from self.coupled_nodes)
        # Node couplings: 2 per pair (from self.coupled_node_pairs)
        n_block_couplings = len(self.coupled_nodes)
        n_node_couplings = len(getattr(self, 'coupled_node_pairs', []))

        self.n_constraints = 2 * (n_block_couplings + n_node_couplings)
        self.n_dofs_augmented = self.n_dofs_base + self.n_constraints

        if self.n_constraints == 0:
            raise ValueError("No coupled nodes registered.")

        # Initialize constraint matrix
        C = np.zeros((self.n_constraints, self.n_dofs_base))

        constraint_row = 0

        # --- 1. FEM-to-Block Couplings ---
        for fem_node_id, block_idx in self.coupled_nodes.items():
            # Get block and positions
            block = structure.list_blocks[block_idx]
            block_node_id = block.connect
            fem_node_pos = structure.list_nodes[fem_node_id]
            block_ref_pos = structure.list_nodes[block_node_id]

            # Get DOF indices
            fem_dof_offset = structure.node_dof_offsets[fem_node_id]
            fem_dofs = np.arange(fem_dof_offset, fem_dof_offset + 2)

            block_dof_offset = structure.node_dof_offsets[block_node_id]
            block_dofs = np.arange(block_dof_offset, block_dof_offset + 3)

            # Store DOF mapping
            self.dof_map[(fem_node_id, block_idx)] = (fem_dofs, block_dofs)

            # Compute relative position
            dx = fem_node_pos[0] - block_ref_pos[0]
            dy = fem_node_pos[1] - block_ref_pos[1]

            # Constraint 1: u_fem - u_block + dy*θ = 0
            C[constraint_row, fem_dofs[0]] = 1.0
            C[constraint_row, block_dofs[0]] = -1.0
            C[constraint_row, block_dofs[2]] = dy

            # Constraint 2: v_fem - v_block - dx*θ = 0
            C[constraint_row + 1, fem_dofs[1]] = 1.0
            C[constraint_row + 1, block_dofs[1]] = -1.0
            C[constraint_row + 1, block_dofs[2]] = -dx

            constraint_row += 2

        # --- 2. FEM-to-FEM Couplings ---
        # self.coupled_node_pairs should be a list of tuples [(node1, node2), ...]
        if hasattr(self, 'coupled_node_pairs'):
            for node1, node2 in self.coupled_node_pairs:
                # Get DOFs
                dof1 = structure.node_dof_offsets[node1]
                dof2 = structure.node_dof_offsets[node2]

                # Assume 2D (u, v) for now
                # Constraint 1: u1 - u2 = 0
                C[constraint_row, dof1] = 1.0
                C[constraint_row, dof2] = -1.0

                # Constraint 2: v1 - v2 = 0
                C[constraint_row + 1, dof1 + 1] = 1.0
                C[constraint_row + 1, dof2 + 1] = -1.0

                constraint_row += 2

        self.constraint_matrix_C = C
        return C

    def add_node_pair(self, node1: int, node2: int):
        """Register a pair of FEM nodes to be tied together."""
        if not hasattr(self, 'coupled_node_pairs'):
            self.coupled_node_pairs = []
        self.coupled_node_pairs.append((node1, node2))

    def build_augmented_system(self, K_base: np.ndarray, P_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build augmented system [K C'; C 0]. Returns (K_aug, P_aug)."""
        if self.constraint_matrix_C is None:
            raise RuntimeError("Constraint matrix C not built.")

        n_dofs = K_base.shape[0]
        n_aug = n_dofs + self.n_constraints

        # Build augmented stiffness matrix
        K_aug = np.zeros((n_aug, n_aug))
        K_aug[:n_dofs, :n_dofs] = K_base
        K_aug[:n_dofs, n_dofs:] = self.constraint_matrix_C.T
        K_aug[n_dofs:, :n_dofs] = self.constraint_matrix_C

        # Build augmented load vector
        P_aug = np.zeros(n_aug)
        P_aug[:n_dofs] = P_base

        return K_aug, P_aug

    def compute_interface_forces(self) -> Dict[Tuple[int, int], np.ndarray]:
        """Return interface forces: {(fem_node_id, block_idx): [Fx, Fy]}."""
        if self.multipliers is None:
            raise RuntimeError("No solution available.")

        forces = {}
        multiplier_idx = 0

        # The loop order MUST match build_constraint_matrix order
        for fem_node_id, block_idx in self.coupled_nodes.items():
            lambda_x = self.multipliers[multiplier_idx]  # Force from u-constraint
            lambda_y = self.multipliers[multiplier_idx + 1]  # Force from v-constraint

            # Forces acting ON the FEM node from the constraint
            forces[(fem_node_id, block_idx)] = np.array([lambda_x, lambda_y])
            multiplier_idx += 2

        return forces

    def get_constraint_residuals(self, u: np.ndarray) -> np.ndarray:
        """Compute linearized constraint residuals: r = C @ u."""
        if self.constraint_matrix_C is None:
            raise RuntimeError("Constraint matrix not built.")
        return self.constraint_matrix_C @ u

    def compute_exact_constraints(self, structure) -> np.ndarray:
        """
        Compute exact geometric constraint violation for large displacements.

        g(u) = x_fem_current - x_anchor_current

        where x_anchor_current includes exact rigid body rotation.
        """
        if not self.coupled_nodes:
            return np.zeros(0)

        residuals = np.zeros(self.n_constraints)
        row = 0

        # Iterate in the same order as build_constraint_matrix
        for fem_node_id, block_idx in self.coupled_nodes.items():
            # --- 1. Get Current FEM Position ---
            # Original position
            X_fem = structure.list_nodes[fem_node_id]

            # Displacement
            fem_dofs = structure.node_dof_offsets[fem_node_id]
            # Safety check for 2D
            u_fem = structure.U[fem_dofs]  # ux
            v_fem = structure.U[fem_dofs + 1]  # uy

            x_fem_curr = X_fem[0] + u_fem
            y_fem_curr = X_fem[1] + v_fem

            # --- 2. Get Current Block Anchor Position ---
            block = structure.list_blocks[block_idx]
            block_node_id = block.connect

            # Block DOFs
            block_dofs = structure.node_dof_offsets[block_node_id]
            u_b = structure.U[block_dofs]  # ux block
            v_b = structure.U[block_dofs + 1]  # uy block
            theta = structure.U[block_dofs + 2]  # rotation

            # Reference (Center) Initial Position
            X_block_ref = structure.list_nodes[block_node_id]

            # Anchor Initial Position (matches FEM initial pos)
            # Relative vector from Block Center to Anchor
            rel_x = X_fem[0] - X_block_ref[0]
            rel_y = X_fem[1] - X_block_ref[1]

            # Exact Rigid Body Rotation
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Rotated relative vector
            rel_x_rot = rel_x * cos_t - rel_y * sin_t
            rel_y_rot = rel_x * sin_t + rel_y * cos_t

            # Current Anchor Position
            x_anchor_curr = (X_block_ref[0] + u_b) + rel_x_rot
            y_anchor_curr = (X_block_ref[1] + v_b) + rel_y_rot

            # --- 3. Compute Residual (Gap) ---
            # g = x_fem - x_anchor
            residuals[row] = x_fem_curr - x_anchor_curr
            residuals[row + 1] = y_fem_curr - y_anchor_curr

            row += 2

        return residuals

    def update_constraint_matrix(self, structure) -> np.ndarray:
        """
        Update the Tangent constraint matrix C based on current displacements.

        This provides the Jacobian of the constraints: J = dg/du.
        Used for the Left-Hand Side (Stiffness Matrix) of the Newton solver.

        Parameters
        ----------
        structure : Hybrid
            Structure with current displacements.

        Returns
        -------
        C : np.ndarray
            Updated Tangent Constraint Matrix.
        """
        if not self.coupled_nodes:
            return np.zeros((0, 0))

        # Reset C
        C = np.zeros((self.n_constraints, self.n_dofs_base))
        row = 0

        for fem_node_id, block_idx in self.coupled_nodes.items():
            # Get Block info
            block = structure.list_blocks[block_idx]
            block_node_id = block.connect

            # Get Current Block Rotation
            block_dofs_idx = structure.node_dof_offsets[block_node_id]
            theta = structure.U[block_dofs_idx + 2]

            # Get Initial Relative Position (Radius Vector)
            X_fem = structure.list_nodes[fem_node_id]
            X_block = structure.list_nodes[block_node_id]
            dx_init = X_fem[0] - X_block[0]
            dy_init = X_fem[1] - X_block[1]

            # Compute Current Lever Arm (Rotated Radius)
            # This represents the current dx, dy in the linearized formula
            # but derived strictly from rotation matrix derivative.
            # d(rel_rot)/dtheta = [-sin -cos; cos -sin] * [dx; dy]

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Derivative of position w.r.t theta gives the lever arm terms
            # d(x_anchor)/dtheta = -rel_x*sin - rel_y*cos = - (rel_y_rotated)
            # d(y_anchor)/dtheta =  rel_x*cos - rel_y*sin = + (rel_x_rotated)

            curr_dx = dx_init * cos_t - dy_init * sin_t  # Current x-offset
            curr_dy = dx_init * sin_t + dy_init * cos_t  # Current y-offset

            # DOF indices
            fem_dofs = structure.node_dof_offsets[fem_node_id]
            u_fem_idx, v_fem_idx = fem_dofs, fem_dofs + 1

            block_dofs = structure.node_dof_offsets[block_node_id]
            u_b_idx, v_b_idx, th_b_idx = block_dofs, block_dofs + 1, block_dofs + 2

            # --- Fill Matrix Rows ---
            # Constraint 1: x_fem - x_anchor = 0
            # Derivatives:
            # d/du_fem = 1
            # d/du_block = -1
            # d/dtheta_block = - d(x_anchor)/dtheta = -(-curr_dy) = +curr_dy

            C[row, u_fem_idx] = 1.0
            C[row, u_b_idx] = -1.0
            C[row, th_b_idx] = curr_dy  # Tangent term

            # Constraint 2: y_fem - y_anchor = 0
            # Derivatives:
            # d/dv_fem = 1
            # d/dv_block = -1
            # d/dtheta_block = - d(y_anchor)/dtheta = -(curr_dx) = -curr_dx

            C[row + 1, v_fem_idx] = 1.0
            C[row + 1, v_b_idx] = -1.0
            C[row + 1, th_b_idx] = -curr_dx  # Tangent term

            row += 2

        self.constraint_matrix_C = C
        return C

    def get_geometric_stiffness(self, structure, multipliers=None) -> np.ndarray:
        """
        Compute geometric stiffness from constraints: K_geo = ∂(C^T*λ)/∂u.

        For large displacement, the constraint Jacobian C depends on u (via theta).
        This leads to a second-order term (curvature of the constraint manifold).

        K_geo ≈ - λ * Hessian(Constraint)
        """
        if multipliers is None:
            if self.multipliers is None:
                raise RuntimeError("No multipliers available.")
            multipliers = self.multipliers

        K_geo = np.zeros((self.n_dofs_base, self.n_dofs_base))
        row = 0

        for fem_node_id, block_idx in self.coupled_nodes.items():
            # Get Multipliers for this node
            lam_x = multipliers[row]
            lam_y = multipliers[row + 1]

            # Get Block Info
            block = structure.list_blocks[block_idx]
            block_node_id = block.connect
            block_dofs = structure.node_dof_offsets[block_node_id]
            th_idx = block_dofs + 2

            # Get Rotation and Geometry
            theta = structure.U[th_idx]
            X_fem = structure.list_nodes[fem_node_id]
            X_block = structure.list_nodes[block_node_id]
            dx_init = X_fem[0] - X_block[0]
            dy_init = X_fem[1] - X_block[1]

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Current rotated relative vector (Lever Arm)
            curr_dx = dx_init * cos_t - dy_init * sin_t
            curr_dy = dx_init * sin_t + dy_init * cos_t

            # The geometric stiffness affects the ROTATIONAL degree of freedom of the block.
            # It comes from the derivative of the tangent terms in C w.r.t theta.
            #
            # Force Moment M_constraint = G_theta^T * lambda
            # M = lam_x * (+curr_dy) + lam_y * (-curr_dx)
            #
            # dM/dtheta = lam_x * d(curr_dy)/dtheta + lam_y * d(-curr_dx)/dtheta
            # d(curr_dy)/dtheta = curr_dx
            # d(-curr_dx)/dtheta = -(-curr_dy) = curr_dy
            #
            # K_geo_theta = lam_x * curr_dx + lam_y * curr_dy
            # This term goes into K[theta, theta] (assuming external force convention K*u = P)
            # But usually Hessian is on the LHS, so it adds to Stiffness.

            k_geo_val = lam_x * curr_dx + lam_y * curr_dy

            # Add to Global Matrix (Diagonal term for block rotation)
            K_geo[th_idx, th_idx] += k_geo_val

            row += 2

        return K_geo

    def verify_block_equilibrium(self, structure):
        """
        Verify that interface forces satisfy equilibrium on each block.

        Returns
        -------
        residuals : dict
            {block_idx: residual_norm} for each block
        """
        interface_forces = self.compute_interface_forces()
        residuals = {}

        for block_idx, block in enumerate(structure.list_blocks):
            # Sum interface forces on this block
            force_sum = np.zeros(3)  # [Fx, Fy, M]

            # Iterate over coupled nodes for this block
            for (fem_node_id, blk_idx), f_vec in interface_forces.items():
                if blk_idx == block_idx:
                    # f_vec corresponds to lambda
                    # Force on Block from constraint = N^T * lambda
                    f = f_vec

                    fem_pos = structure.list_nodes[fem_node_id]
                    block_ref = block.ref_point

                    dx = fem_pos[0] - block_ref[0]
                    dy = fem_pos[1] - block_ref[1]

                    # Force contribution to block (N^T * lambda)
                    # N = [[1, 0, -dy], [0, 1, dx]]
                    # N^T @ lambda = [lambda_x, lambda_y, -dy*lambda_x + dx*lambda_y]
                    force_sum[0] += f[0]
                    force_sum[1] += f[1]

                    # Moment contribution: -dy * lx + dx * ly = cross(r, f)
                    force_sum[2] += -dy * f[0] + dx * f[1]

            # Add external loads on block
            # structure.P contains external loads.
            if hasattr(structure, 'P') and structure.P is not None:
                block_dofs = structure.node_dof_offsets[block.connect]
                ext_loads = structure.P[block_dofs: block_dofs + 3]
                # Equilibrium: F_int - F_ext = 0
                # F_int = K*u + C^T*lambda
                # For rigid block K*u is contact forces + ...
                # If we neglect other stiffnesses, P_ext + force_sum (from constraints) = 0
                total_force = force_sum + ext_loads
                residuals[block_idx] = np.linalg.norm(total_force)
            else:
                residuals[block_idx] = np.linalg.norm(force_sum)

        return residuals

    # ============================================================
    # Utility Methods
    # ============================================================

    def _is_positive_definite(self, K: np.ndarray) -> bool:
        try:
            np.linalg.cholesky(K)
            return True
        except np.linalg.LinAlgError:
            return False

    def get_info(self) -> Dict:
        base_info = super().get_info()
        lagrange_info = {
            'n_constraints': self.n_constraints,
            'n_dofs_base': self.n_dofs_base,
            'solution_available': self.multipliers is not None,
        }
        return {**base_info, **lagrange_info}

    def validate(self) -> bool:
        """Validate coupling configuration (Override to support FEM-FEM pairs)."""
        self.diagnostics['activation_errors'].clear()

        has_block_nodes = bool(self.coupled_nodes)
        has_node_pairs = hasattr(self, 'coupled_node_pairs') and bool(self.coupled_node_pairs)

        if not has_block_nodes and not has_node_pairs:
            self.diagnostics['activation_errors'].append(
                "No coupled nodes (Block-FEM) or node pairs (FEM-FEM) registered")
            return False

        return True

    def __repr__(self) -> str:
        status = "active" if self.active else "inactive"
        return (f"LagrangeCoupling(status={status}, "
                f"constraints={self.n_constraints})")
