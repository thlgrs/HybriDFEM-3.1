import copy
from typing import Dict, List, Tuple

import numpy as np

from .BaseCoupling import BaseCoupling
from .IntegrationRule import generate_interface_integration_points, gauss_points_1d
from .InterfaceDetection import detect_mortar_interfaces, detect_fem_master_interfaces
from .ShapeFunction import evaluate_shape_functions_at_point


class MortarCoupling(BaseCoupling):
    """
    True Mortar method coupling via nodal Lagrange multipliers.

    This method enforces coupling constraints weakly through numerical
    integration along the slave (FEM) side, forming an augmented saddle
    point system:

        [K   B^T] [u]   [P]
        [B    0 ] [λ] = [0]

    Where:
        - K: Base stiffness matrix (n_dofs × n_dofs)
        - B: Mortar constraint matrix (n_constraints × n_dofs)
        - u: Displacements (n_dofs)
        - λ: Nodal Lagrange multipliers (n_constraints)
        - P: Applied loads (n_dofs)

    Key Features (True Mortar Method)
    ---------------------------------
    - **Non-matching mesh capability**: Works with offset, misaligned meshes
    - **Weak constraint enforcement**: Integral constraints via Gauss quadrature
    - **Optimal convergence**: Maintains FEM convergence rates (LBB stable)
    - **Nodal multipliers**: λ defined at slave (FEM) nodes, not integration points
    - **Variational consistency**: Mathematically rigorous weak formulation
    - **Saddle point system**: Symmetric but indefinite

    Mathematical Formulation (Popp's Method)
    ----------------------------------------
    The Lagrange multiplier field λ is discretized using shape functions:

        λ_h = Σⱼ Φⱼ(x) λⱼ

    For standard mortar, Φⱼ = Nⱼ (same shape functions as displacement).

    The weak constraint is:

        ∫_Γ Φⱼ · (u_slave - u_master) dΓ = 0  for all j

    This leads to two coupling matrices:
        - D: Slave-Lagrange coupling: D_jk = ∫_Γ Φⱼ · Nₖ^slave dΓ
        - M: Master-Lagrange coupling: M_jl = ∫_Γ Φⱼ · [rigid body terms] dΓ

    The constraint equation becomes: D·u_slave - M·q_block = 0

    Constraint Matrix Structure
    ---------------------------
    The B matrix combines D and M:
        B = [D columns for slave DOFs | -M columns for block DOFs | zeros]

    This is assembled row-by-row for each slave node's two DOFs (u, v).

    Integration Domain
    ------------------
    Integration is performed over the SLAVE (FEM) side:
        - Loop over slave element edges on interface
        - Place Gauss points on each slave edge
        - Project to master (block) for rigid body contribution

    Attributes
    ----------
    interfaces : List[MortarInterface]
        Detected interfaces between blocks and FEM
    slave_nodes_global : List[int]
        Global node IDs of all slave nodes across all interfaces
    matrix_D : np.ndarray or None
        Slave mass matrix (n_slave_nodes*2 × n_slave_dofs)
    matrix_M : np.ndarray or None
        Master coupling matrix (n_slave_nodes*2 × n_master_dofs)
    constraint_matrix_C : np.ndarray or None
        Global constraint matrix B (n_constraints × n_dofs)
    integration_order : int
        Gauss quadrature order (1, 2, or 3)
    interface_tolerance : float
        Distance tolerance for interface detection
    n_constraints : int
        Total number of constraints (2 per slave node)
    n_dofs_base : int
        Number of DOFs in base system (before augmentation)
    n_dofs_augmented : int
        Number of DOFs in augmented system (base + constraints)
    multipliers : np.ndarray or None
        Lagrange multipliers (interface tractions) from solution
    """

    def __init__(self, integration_order: int = 2, interface_tolerance: float = 1e-4,
                 interface_orientation: str = 'horizontal'):
        """
        Initialize mortar method coupling.

        Parameters
        ----------
        integration_order : int, optional
            Gauss quadrature order (1, 2, or 3), default=2
            - 1: 1-point rule (linear exact)
            - 2: 2-point rule (cubic exact) - recommended
            - 3: 3-point rule (quintic exact)
        interface_tolerance : float, optional
            Distance tolerance for interface detection, default=1e-4
            Should be larger than nodal coupling tolerance (1e-9)
        interface_orientation : str, optional
            Filter interfaces by orientation, default='horizontal'
            - 'horizontal': Keep only horizontal interfaces (beam-on-column)
            - 'vertical': Keep only vertical interfaces
            - None: Keep all detected interfaces
            For typical beam-on-column coupling, use 'horizontal' to avoid
            spurious vertical interface detection and rank deficiency.
        """
        super().__init__(coupling_type='mortar')

        # Mortar-specific settings
        self.integration_order = integration_order
        self.interface_tolerance = interface_tolerance
        self.interface_orientation = interface_orientation
        self.verbose = False

        # Interface and integration data
        self.interfaces = []  # List[MortarInterface]
        self.integration_points = []  # List[IntegrationPoint] (kept for backward compat)

        # TRUE MORTAR: Slave node tracking
        self.slave_nodes_global = []  # All slave node IDs across all interfaces
        self.slave_node_to_row = {}   # Maps slave_node_id -> constraint row index

        # TRUE MORTAR: D and M matrices
        self.matrix_D = None  # Slave mass matrix (n_constraints × n_slave_dofs)
        self.matrix_M = None  # Master coupling matrix (n_constraints × n_master_dofs)

        # Constraint matrix and system dimensions
        self.constraint_matrix_C = None  # Combined constraint matrix B
        self.n_constraints = 0
        self.n_dofs_base = None
        self.n_dofs_augmented = None

        # Solution storage
        self.multipliers = None  # λ values (interface tractions)

        # Mapping for constraint construction (kept for backward compat)
        self.integration_point_map = {}  # Maps integration point → constraint rows

    # ============================================================
    # CouplingBase Abstract Method Implementations
    # ============================================================

    def get_system_modification_type(self) -> str:
        """
        Return system modification type for mortar coupling.

        Returns
        -------
        str
            'matrix_augmentation' - adds multipliers as new unknowns
        """
        return 'matrix_augmentation'

    def get_dof_count_change(self, n_base_dofs: int) -> int:
        """
        Return change in DOF count due to mortar coupling.

        Mortar coupling adds 2 constraints (and thus 2 multipliers)
        per integration point, increasing the system size.

        Parameters
        ----------
        n_base_dofs : int
            Number of DOFs in base system (before coupling)

        Returns
        -------
        int
            Number of additional DOFs (positive for augmentation)
            = 2 * n_integration_points
        """
        return self.n_constraints

    def requires_special_solver(self) -> bool:
        """
        Check if mortar coupling requires special solver.

        Returns
        -------
        bool
            True - Mortar requires saddle point solver for indefinite system
        """
        return True

    def validate(self) -> bool:
        """
        Validate mortar coupling configuration.

        Override CouplingBase.validate() since mortar uses interfaces,
        not coupled_nodes.

        Returns
        -------
        valid : bool
            True if configuration is valid
        """
        self.diagnostics['activation_errors'].clear()

        # Check for interfaces (mortar doesn't use coupled_nodes)
        if not self.interfaces:
            self.diagnostics['activation_errors'].append(
                "No interfaces detected")
            return False

        # TRUE MORTAR: Check for slave nodes (primary) or integration points (backward compat)
        # The true mortar method uses slave nodes; the old collocation used integration points
        has_slave_nodes = len(self.slave_nodes_global) > 0
        has_integration_points = len(self.integration_points) > 0

        if not has_slave_nodes and not has_integration_points:
            self.diagnostics['activation_errors'].append(
                "No slave nodes or integration points generated")
            return False

        # Check constraint matrix was built
        if self.constraint_matrix_C is None:
            self.diagnostics['activation_errors'].append(
                "Constraint matrix C not built")
            return False

        return True

    # ============================================================
    # Mortar-Specific Methods: Interface Detection
    # ============================================================

    def detect_interfaces(self, structure) -> List:
        """
        Detect mortar interfaces in the hybrid structure.

        Automatically identifies coupling interfaces and determines the
        appropriate master/slave assignment based on mesh density.

        For standard cases (coarse blocks, fine FEM): Block = Master, FEM = Slave
        For inverted cases (fine blocks, coarse FEM): FEM = Master, Block = Slave

        Parameters
        ----------
        structure : Hybrid
            Hybrid structure containing blocks and FEM elements

        Returns
        -------
        interfaces : List[MortarInterface]
            Detected interfaces

        Notes
        -----
        The method first tries block-master interfaces. If no slave edges
        are found (indicating blocks are finer than FEM), it switches to
        FEM-master interfaces.
        """
        # First try standard approach: Block as master
        self.interfaces = detect_mortar_interfaces(
            structure,
            tolerance=self.interface_tolerance,
            interface_orientation=self.interface_orientation
        )

        # Check if we can find slave edges with this approach
        can_find_slave_edges = False
        if self.interfaces:
            # Test first interface
            test_interface = self.interfaces[0]
            test_interface.detect_slave_edges(structure, tolerance=self.interface_tolerance)
            can_find_slave_edges = len(test_interface.slave_edges) > 0

        # If no slave edges found, try FEM-master approach
        if self.interfaces and not can_find_slave_edges:
            self.interfaces = detect_fem_master_interfaces(
                structure,
                tolerance=self.interface_tolerance,
                interface_orientation=self.interface_orientation
            )

        return self.interfaces

    def generate_integration_points(self) -> List:
        """
        Generate integration points along all detected interfaces.

        Uses Gauss quadrature rules to place integration points
        along each interface segment.

        Returns
        -------
        integration_points : List[IntegrationPoint]
            All integration points across all interfaces

        Notes
        -----
        Uses Week 1 infrastructure: IntegrationRule module
        """
        if not self.interfaces:
            raise RuntimeError("No interfaces detected. Call detect_interfaces() first.")

        self.integration_points = []

        for interface_id, interface in enumerate(self.interfaces):
            int_points = generate_interface_integration_points(
                interface,
                integration_order=self.integration_order,
                interface_id=interface_id
            )
            self.integration_points.extend(int_points)

        return self.integration_points

    # ============================================================
    # Mortar-Specific Methods: Constraint Matrix Assembly
    # ============================================================

    def build_constraint_matrix(self, structure) -> np.ndarray:
        """
        Build global mortar constraint matrix B via true mortar integration.

        Implements the standard mortar method (Popp's reference):
        - Lagrange multipliers are defined at slave (FEM) nodes
        - Integration is performed over slave element edges
        - D matrix: slave-slave mass matrix on interface
        - M matrix: slave-master coupling matrix

        The constraint equation is: D·u_slave - M·q_block = 0

        Parameters
        ----------
        structure : Hybrid
            Hybrid structure containing blocks, FEM elements, and nodes

        Returns
        -------
        B : np.ndarray
            Constraint matrix (n_constraints × n_dofs)
            n_constraints = 2 * n_slave_nodes

        Algorithm (True Mortar)
        -----------------------
        1. Detect interfaces between blocks and FEM
        2. For each interface:
           a. Detect slave (FEM) element edges on interface
           b. Build list of slave nodes
        3. Build D and M matrices by integrating over slave edges:
           - D_jk = ∫_Γ Φⱼ · Nₖ dΓ  (j, k are slave nodes)
           - M_jl = ∫_Γ Φⱼ · [rigid body] dΓ  (j is slave, l is block DOF)
        4. Assemble global constraint matrix B from D and M
        """
        # Step 1: Detect interfaces
        if not self.interfaces:
            self.detect_interfaces(structure)

        if not self.interfaces:
            raise ValueError("No interfaces found. Check structure geometry and tolerance.")

        # Step 2: Detect slave edges and build slave node list
        self._detect_all_slave_edges(structure)

        if not self.slave_nodes_global:
            raise ValueError("No slave nodes found on interfaces. Check mesh alignment.")

        # Get system dimensions
        self.n_dofs_base = structure.nb_dofs
        n_slave_nodes = len(self.slave_nodes_global)
        self.n_constraints = 2 * n_slave_nodes  # 2 DOFs (u, v) per slave node
        self.n_dofs_augmented = self.n_dofs_base + self.n_constraints

        # Step 3: Build D and M matrices via slave edge integration
        self._build_D_and_M_matrices(structure)

        # Step 4: Assemble global constraint matrix B from D and M
        B = self._assemble_constraint_matrix_B(structure)

        # Store constraint matrix (using C for API compatibility)
        self.constraint_matrix_C = B

        return B

    def _detect_all_slave_edges(self, structure):
        """
        Detect slave edges on all interfaces and build global slave node list.

        This populates:
        - interface.slave_edges for each interface
        - self.slave_nodes_global (unique, sorted)
        - self.slave_node_to_row mapping

        For block-master interfaces: detects FEM edges as slave
        For FEM-master interfaces: detects block faces as slave
        """
        self.slave_nodes_global = []
        seen_nodes = set()

        for interface in self.interfaces:
            # Detect slave edges based on master type
            if interface.master_type == 'block':
                # Standard case: FEM edges are slave
                interface.detect_slave_edges(structure, tolerance=self.interface_tolerance)
            else:
                # FEM master case: check if we have FEM slave elements or block faces
                if interface.fem_element_ids:
                    # FEM-FEM coupling: FEM edges are slave
                    interface.detect_slave_edges(structure, tolerance=self.interface_tolerance)
                elif interface.block_faces:
                    # Inverted case: Block faces are slave
                    interface.detect_block_slave_edges(structure, tolerance=self.interface_tolerance)
                else:
                    # Try to detect FEM slave edges
                    interface.detect_slave_edges(structure, tolerance=self.interface_tolerance)

            # Add unique slave nodes to global list
            for node_id in interface.slave_nodes_sorted:
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    self.slave_nodes_global.append(node_id)

        # Build slave node to constraint row mapping
        # Each slave node contributes 2 constraint rows (u and v)
        self.slave_node_to_row = {}
        for i, node_id in enumerate(self.slave_nodes_global):
            self.slave_node_to_row[node_id] = 2 * i

    def _build_D_and_M_matrices(self, structure):
        """
        Build D and M matrices by integrating over slave element edges.

        D_jk = ∫_Γ Φⱼ(x) · Nₖ(x) dΓ
        M_jl = ∫_Γ Φⱼ(x) · [kinematic term for master DOF l] dΓ

        Optimized implementation using vectorized operations and pre-computed kernels.
        """
        n_slave_nodes = len(self.slave_nodes_global)

        # Initialize matrices
        # D couples to slave nodes (subset of DOFs) -> Keep local dense mapping
        self.matrix_D = np.zeros((self.n_constraints, 2 * n_slave_nodes))

        # M couples to Master DOFs using global DOF indexing
        # Since Master can be Blocks (3 DOFs) or FEM (2 DOFs), we use full global DOF size
        self.matrix_M = np.zeros((self.n_constraints, structure.nb_dofs))

        # Optimization 1: Fast Lookup for Slave Nodes
        slave_node_idx_map = {node_id: i for i, node_id in enumerate(self.slave_nodes_global)}

        # Optimization 2: Pre-compute Reference Kernels for linear (2 nodes) and quadratic (3 nodes)
        kernels = {}
        for n_nodes in [2, 3]:
            xi_points, weights = gauss_points_1d(self.integration_order)
            K_ref = np.zeros((n_nodes, n_nodes))
            vec_ref = np.zeros(n_nodes)

            for xi, w in zip(xi_points, weights):
                N = self._evaluate_edge_shape_functions(xi, n_nodes)
                K_ref += np.outer(N, N) * w
                vec_ref += N * w
            kernels[n_nodes] = (K_ref, vec_ref)

        # Loop over each interface and dispatch by master type
        for interface in self.interfaces:
            if interface.master_type == 'block':
                self._build_block_master_contributions(
                    interface, structure, slave_node_idx_map, kernels
                )
            elif interface.master_type == 'fem':
                # Check if this is FEM-FEM coupling (slave edges are FEM, not blocks)
                has_fem_slave = (interface.slave_edges and
                                 not hasattr(interface.slave_edges[0], 'is_block_edge'))
                if has_fem_slave:
                    self._build_fem_fem_contributions(
                        interface, structure, slave_node_idx_map, kernels
                    )
                else:
                    self._build_fem_master_contributions(
                        interface, structure, slave_node_idx_map
                    )
            else:
                raise ValueError(f"Unknown master type: {interface.master_type}")

    def _build_block_master_contributions(self, interface, structure, slave_node_idx_map, kernels):
        """Helper to build contributions for a Block master (Vectorized)."""
        block_idx = interface.master_id
        block = structure.list_blocks[block_idx]
        block_node_id = block.connect
        block_ref_pos = np.array(structure.list_nodes[block_node_id])
        xref, yref = block_ref_pos[0], block_ref_pos[1]

        # Global Master DOFs
        block_dof_offset = structure.node_dof_offsets[block_node_id]
        col_block_u = block_dof_offset
        col_block_v = block_dof_offset + 1
        col_block_theta = block_dof_offset + 2

        # Loop over each slave edge on this interface
        for slave_edge in interface.slave_edges:
            edge_length = slave_edge.edge_length
            edge_global_nodes = slave_edge.global_node_ids
            n_edge_nodes = len(edge_global_nodes)

            # Get coordinates of edge nodes
            edge_coords = np.array([structure.list_nodes[nid] for nid in edge_global_nodes])
            x_nodes = edge_coords[:, 0]
            y_nodes = edge_coords[:, 1]

            if n_edge_nodes in kernels:
                K_ref, vec_ref = kernels[n_edge_nodes]
            else:
                continue

            # --- Vectorized Integration on Physical Edge ---
            D_local = K_ref * edge_length
            int_N_local = vec_ref * edge_length
            M_theta_u_local = -(D_local @ y_nodes - yref * int_N_local)
            M_theta_v_local = (D_local @ x_nodes - xref * int_N_local)

            # --- Assembly ---
            valid_indices = []
            global_dof_indices = []

            for i, node_id in enumerate(edge_global_nodes):
                if node_id in slave_node_idx_map:
                    valid_indices.append(i)
                    global_dof_indices.append(slave_node_idx_map[node_id])

            for i_local, idx_global in zip(valid_indices, global_dof_indices):
                row_u = 2 * idx_global
                row_v = 2 * idx_global + 1

                # D Matrix
                for k_local, k_global_idx in zip(valid_indices, global_dof_indices):
                    val = D_local[i_local, k_local]
                    col_u = 2 * k_global_idx
                    col_v = 2 * k_global_idx + 1
                    self.matrix_D[row_u, col_u] += val
                    self.matrix_D[row_v, col_v] += val

                # M Matrix (Global indexing)
                self.matrix_M[row_u, col_block_u] += int_N_local[i_local]
                self.matrix_M[row_u, col_block_theta] += M_theta_u_local[i_local]
                self.matrix_M[row_v, col_block_v] += int_N_local[i_local]
                self.matrix_M[row_v, col_block_theta] += M_theta_v_local[i_local]

    def _build_fem_fem_contributions(self, interface, structure, slave_node_idx_map, kernels):
        """
        Build contributions for FEM-FEM mortar coupling.

        Both master and slave are FEM element edges. Integration is performed
        over slave edges, and master displacement is interpolated to integration points.

        Constraint: ∫_Γ Φⱼ · (u_slave - u_master) dΓ = 0

        Parameters
        ----------
        interface : MortarInterface
            Interface with master_type='fem' and FEM slave edges
        structure : Hybrid or Structure_FEM
            Structure containing FEM elements
        slave_node_idx_map : dict
            Mapping from slave node ID to local index
        kernels : dict
            Pre-computed integration kernels
        """
        # Get master edge info
        master_elem_id = interface.master_id
        master_edge_id = interface.master_face_id
        master_elem = structure.list_fes[master_elem_id]

        # Master edge vertices and nodes
        master_v0 = np.array(interface.master_vertices[0])
        master_v1 = np.array(interface.master_vertices[1])
        master_length = interface.interface_length

        master_node_global = interface.master_global_nodes
        master_coords = [np.array(structure.list_nodes[nid]) for nid in master_node_global]

        # Get Gauss points
        xi_points, weights = gauss_points_1d(self.integration_order)

        # Loop over each slave edge
        for slave_edge in interface.slave_edges:
            edge_length = slave_edge.edge_length
            edge_global_nodes = slave_edge.global_node_ids
            n_edge_nodes = len(edge_global_nodes)

            if n_edge_nodes not in kernels:
                continue

            K_ref, vec_ref = kernels[n_edge_nodes]

            # Get slave edge vertices
            slave_v0 = slave_edge.vertices[0]
            slave_v1 = slave_edge.vertices[-1]

            # D matrix: slave-slave coupling (using pre-computed kernel)
            D_local = K_ref * edge_length

            # Assembly for D matrix
            valid_indices = []
            global_dof_indices = []

            for i, node_id in enumerate(edge_global_nodes):
                if node_id in slave_node_idx_map:
                    valid_indices.append(i)
                    global_dof_indices.append(slave_node_idx_map[node_id])

            for i_local, idx_global in zip(valid_indices, global_dof_indices):
                row_u = 2 * idx_global
                row_v = 2 * idx_global + 1

                # D Matrix
                for k_local, k_global_idx in zip(valid_indices, global_dof_indices):
                    val = D_local[i_local, k_local]
                    col_u = 2 * k_global_idx
                    col_v = 2 * k_global_idx + 1
                    self.matrix_D[row_u, col_u] += val
                    self.matrix_D[row_v, col_v] += val

            # M matrix: slave-master coupling (requires integration)
            # Integrate along slave edge, interpolate master at each point
            for xi, w in zip(xi_points, weights):
                # Physical point on slave edge
                x_q = (1.0 - xi) * slave_v0 + xi * slave_v1
                w_q = w * edge_length

                # Slave shape functions
                N_slave = self._evaluate_edge_shape_functions(xi, n_edge_nodes)

                # Project x_q onto master edge to get parametric coordinate
                master_vec = master_v1 - master_v0
                if master_length > 1e-12:
                    t_master = np.dot(x_q - master_v0, master_vec) / (master_length ** 2)
                    t_master = np.clip(t_master, 0.0, 1.0)
                else:
                    t_master = 0.5

                # Master shape functions (linear interpolation)
                N_master = np.array([1.0 - t_master, t_master])

                # Assemble M matrix contribution
                for i_local, idx_global in zip(valid_indices, global_dof_indices):
                    row_u = 2 * idx_global
                    row_v = 2 * idx_global + 1

                    # M couples slave test functions to master DOFs
                    for m_idx, m_node_id in enumerate(master_node_global[:2]):
                        m_dof_offset = structure.node_dof_offsets[m_node_id]
                        val = N_slave[i_local] * N_master[m_idx] * w_q

                        self.matrix_M[row_u, m_dof_offset] += val
                        self.matrix_M[row_v, m_dof_offset + 1] += val

    def _build_fem_master_contributions(self, interface, structure, slave_node_idx_map):
        """
        Helper to build contributions for FEM master with block slave.

        When blocks are finer than FEM (FEM-master interfaces), we integrate over
        block edges and the constraint is that block rigid body motion equals
        FEM interpolated displacement.

        For each block on the interface:
        - Block motion: u = u_block - (y-yref)*θ, v = v_block + (x-xref)*θ
        - FEM motion: u = Σ Nᵢ·uᵢ (interpolated from master FEM edge)
        - Constraint: block_motion = fem_motion (in weak form)
        """
        # Get master FEM element info
        master_elem_id = interface.master_id
        master_edge_id = interface.master_face_id
        master_elem = structure.list_fes[master_elem_id]

        # Get master edge node DOFs
        master_node_local = interface.master_local_nodes
        master_node_global = [master_elem.connect[n] for n in master_node_local]
        master_dofs = []
        for nid in master_node_global:
            dof_offset = structure.node_dof_offsets[nid]
            n_dofs = structure.node_dof_counts[nid]
            master_dofs.extend(range(dof_offset, dof_offset + min(n_dofs, 2)))  # Only u, v

        # Master edge vertices for interpolation
        master_v0 = np.array(master_elem.nodes[master_node_local[0]])
        master_v1 = np.array(master_elem.nodes[master_node_local[-1]])
        master_length = np.linalg.norm(master_v1 - master_v0)

        # Gauss points for integration
        xi_points, weights = gauss_points_1d(self.integration_order)

        # Process each block edge (slave)
        for slave_edge in interface.slave_edges:
            if not hasattr(slave_edge, 'block_id'):
                continue  # Skip non-block edges

            block_id = slave_edge.block_id
            block = structure.list_blocks[block_id]
            block_node_id = block.connect

            # Get block reference position and DOFs
            block_ref_pos = np.array(structure.list_nodes[block_node_id])
            xref, yref = block_ref_pos[0], block_ref_pos[1]
            block_dof_offset = structure.node_dof_offsets[block_node_id]

            # Get slave node index for constraint row
            if block_node_id not in slave_node_idx_map:
                continue
            slave_idx = slave_node_idx_map[block_node_id]
            row_u = 2 * slave_idx
            row_v = 2 * slave_idx + 1

            # Block edge geometry
            edge_v0 = slave_edge.vertices[0]
            edge_v1 = slave_edge.vertices[-1]
            edge_length = slave_edge.edge_length

            # Integrate over block edge
            for xi, w in zip(xi_points, weights):
                # Physical point on block edge
                x_q = (1 - xi) * edge_v0 + xi * edge_v1
                w_q = w * edge_length

                # Block rigid body contributions (slave - goes into D conceptually)
                # u_block = u + (y_q - yref) * (-θ)
                # v_block = v + (x_q - xref) * θ
                dy = x_q[1] - yref
                dx = x_q[0] - xref

                # D matrix: contribution from block DOFs
                # For u equation: coefficient of u_block is +1
                self.matrix_D[row_u, 2 * slave_idx] += w_q
                # For u equation: coefficient of θ is -dy
                # (handled in M matrix for block DOFs)
                # For v equation: coefficient of v_block is +1
                self.matrix_D[row_v, 2 * slave_idx + 1] += w_q

                # M matrix: block rotation contribution (since M uses global DOF indexing)
                # For u equation: -dy * θ term
                self.matrix_M[row_u, block_dof_offset + 2] -= -dy * w_q  # Actually +dy*w_q
                # For v equation: +dx * θ term
                self.matrix_M[row_v, block_dof_offset + 2] -= dx * w_q  # Actually -dx*w_q

                # FEM master contributions (goes into M matrix)
                # Project x_q onto master edge to get parametric coordinate
                edge_vec = master_v1 - master_v0
                proj_t = np.dot(x_q - master_v0, edge_vec) / (master_length ** 2)
                proj_t = np.clip(proj_t, 0.0, 1.0)

                # Evaluate FEM shape functions (linear edge)
                N_fem = np.array([1.0 - proj_t, proj_t])

                # Add FEM contributions to M matrix
                for i_node, master_nid in enumerate(master_node_global):
                    master_dof_offset = structure.node_dof_offsets[master_nid]
                    # u equation: coefficient of u_fem is -Nᵢ (because constraint is u_block - u_fem = 0)
                    self.matrix_M[row_u, master_dof_offset] += N_fem[i_node] * w_q
                    # v equation: coefficient of v_fem is -Nᵢ
                    self.matrix_M[row_v, master_dof_offset + 1] += N_fem[i_node] * w_q

    def _evaluate_edge_shape_functions(self, xi: float, n_nodes: int) -> np.ndarray:
        """
        Evaluate 1D shape functions along an element edge.

        Parameters
        ----------
        xi : float
            Parametric coordinate along edge [0, 1]
        n_nodes : int
            Number of nodes on edge (2 for linear, 3 for quadratic)

        Returns
        -------
        np.ndarray
            Shape function values at xi
        """
        if n_nodes == 2:
            # Linear shape functions
            return np.array([1.0 - xi, xi])
        elif n_nodes == 3:
            # Quadratic shape functions (standard)
            # Node 0 at xi=0, Node 2 at xi=1, Node 1 at xi=0.5
            N0 = (1.0 - xi) * (1.0 - 2.0 * xi)
            N1 = 4.0 * xi * (1.0 - xi)
            N2 = xi * (2.0 * xi - 1.0)
            return np.array([N0, N1, N2])
        else:
            # Default to linear
            return np.array([1.0 - xi, xi])

    def _assemble_constraint_matrix_B(self, structure) -> np.ndarray:
        """
        Assemble global constraint matrix B from D and M matrices.

        The constraint is: D·u_slave - M·u_master = 0
        Rearranged for B·u = 0: B = [+D columns for slave DOFs] + [-M for master DOFs]

        Since the new architecture uses M with global DOF indexing directly,
        we just need to map D's local indices to global DOFs.

        Returns
        -------
        B : np.ndarray
            Global constraint matrix (n_constraints × n_dofs)
        """
        n_dofs = self.n_dofs_base
        B = np.zeros((self.n_constraints, n_dofs))

        # 1. M Matrix is already in global DOF coordinates
        # B -= M (because constraint is D*u_slave - M*u_master = 0)
        B -= self.matrix_M

        # 2. D Matrix needs local-to-global mapping
        # D columns correspond to (u, v) pairs for each slave node in order
        for i, slave_node in enumerate(self.slave_nodes_global):
            dof_offset = structure.node_dof_offsets[slave_node]
            D_col_u = 2 * i
            D_col_v = 2 * i + 1

            # Add D columns to global DOF positions
            # Note: += to handle potential overlap (shouldn't happen for D)
            B[:, dof_offset] += self.matrix_D[:, D_col_u]
            B[:, dof_offset + 1] += self.matrix_D[:, D_col_v]

        return B

    # ============================================================
    # System Assembly (same as Lagrange method)
    # ============================================================

    def build_augmented_system(self, K_base: np.ndarray, P_base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build augmented saddle point system.

        Constructs the augmented system:

            [K   C^T] [u]   [P]
            [C    0 ] [λ] = [0]

        Parameters
        ----------
        K_base : np.ndarray
            Base stiffness matrix (n_dofs × n_dofs)
        P_base : np.ndarray
            Base load vector (n_dofs)

        Returns
        -------
        K_aug : np.ndarray
            Augmented stiffness matrix (n_aug × n_aug)
        P_aug : np.ndarray
            Augmented load vector (n_aug)

        Notes
        -----
        Identical structure to LagrangeCoupling, but C matrix is different.
        """
        if self.constraint_matrix_C is None:
            raise RuntimeError("Constraint matrix C not built. Call build_constraint_matrix() first.")

        n_dofs = K_base.shape[0]
        n_constraints = self.constraint_matrix_C.shape[0]
        n_aug = n_dofs + n_constraints

        # Validate dimensions
        if K_base.shape != (n_dofs, n_dofs):
            raise ValueError(f"K_base must be square: got {K_base.shape}")
        if P_base.shape[0] != n_dofs:
            raise ValueError(f"P_base size mismatch: got {P_base.shape[0]}, expected {n_dofs}")
        if self.constraint_matrix_C.shape[1] != n_dofs:
            raise ValueError(
                f"C has wrong number of columns: got {self.constraint_matrix_C.shape[1]}, expected {n_dofs}")

        # Build augmented stiffness matrix
        K_aug = np.zeros((n_aug, n_aug))

        # Top-left: Base stiffness K
        K_aug[:n_dofs, :n_dofs] = K_base

        # Top-right: C^T (constraint coupling)
        K_aug[:n_dofs, n_dofs:] = self.constraint_matrix_C.T

        # Bottom-left: C (transpose of top-right for symmetry)
        K_aug[n_dofs:, :n_dofs] = self.constraint_matrix_C

        # Bottom-right: 0 (no interaction between multipliers)
        # Already zero from initialization

        # Build augmented load vector
        P_aug = np.zeros(n_aug)
        P_aug[:n_dofs] = P_base
        # P_aug[n_dofs:] = 0  (constraint equations = 0)

        return K_aug, P_aug

    def extract_solution(self, u_aug: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract displacements and multipliers from augmented solution.

        The augmented solution vector contains:
            u_aug = [u₁, u₂, ..., uₙ, λ₁, λ₂, ..., λₘ]^T

        Parameters
        ----------
        u_aug : np.ndarray
            Augmented solution vector (n_dofs + n_constraints)

        Returns
        -------
        u : np.ndarray
            Displacements (n_dofs)
        multipliers : np.ndarray
            Lagrange multipliers (n_constraints)
        """
        if self.constraint_matrix_C is None:
            raise RuntimeError("Constraint matrix not built. Call build_constraint_matrix() first.")

        n_dofs = self.n_dofs_base
        n_constraints = self.n_constraints

        if u_aug.shape[0] != n_dofs + n_constraints:
            raise ValueError(f"Solution size mismatch: got {u_aug.shape[0]}, expected {n_dofs + n_constraints}")

        # Split solution vector
        u = u_aug[:n_dofs]
        multipliers = u_aug[n_dofs:]

        # Store multipliers
        self.multipliers = multipliers

        return u, multipliers

    # ============================================================
    # Post-Processing Methods
    # ============================================================

    def compute_interface_tractions(self) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Compute interface traction distribution from Lagrange multipliers.

        In the true mortar method, Lagrange multipliers are defined at slave
        (FEM) nodes. The multipliers λ represent nodal interface tractions.

        Returns
        -------
        tractions : Dict[int, Dict[int, np.ndarray]]
            Dictionary mapping interface_id to dict of {node_id: traction_vector}
            Each traction vector is [tx, ty] at a slave node

        Notes
        -----
        Units: Force per unit area [N/m²] or nodal forces depending on
        interpretation of the weak form.
        Tractions act on both block and FEM (Newton's 3rd law)
        """
        if self.multipliers is None:
            raise RuntimeError("No solution available. Solve system first.")

        tractions = {}

        # Build mapping of slave nodes to interfaces
        for interface in self.interfaces:
            interface_id = interface.block_id
            if interface_id not in tractions:
                tractions[interface_id] = {}

            for node_id in interface.slave_nodes_sorted:
                if node_id in self.slave_node_to_row:
                    row_u = self.slave_node_to_row[node_id]
                    row_v = row_u + 1

                    if row_u < len(self.multipliers) and row_v < len(self.multipliers):
                        lambda_x = self.multipliers[row_u]
                        lambda_y = self.multipliers[row_v]
                        tractions[interface_id][node_id] = np.array([lambda_x, lambda_y])
                    else:
                        tractions[interface_id][node_id] = np.array([0.0, 0.0])

        return tractions

    def compute_nodal_interface_forces(self, structure) -> Dict[int, np.ndarray]:
        """
        Compute nodal interface forces from Lagrange multipliers.

        The constraint force on each slave node can be computed as:
            F_slave = D^T @ λ  (distributed to slave nodes)
            F_block = -M^T @ λ (distributed to block DOFs)

        Returns
        -------
        forces : Dict[int, np.ndarray]
            Dictionary mapping node_id to force vector [Fx, Fy]
        """
        if self.multipliers is None:
            raise RuntimeError("No solution available. Solve system first.")

        forces = {}

        # Compute slave node forces: F = D^T @ λ
        for i, node_id in enumerate(self.slave_nodes_global):
            col_u = 2 * i
            col_v = 2 * i + 1

            # F_u = D[:, col_u]^T @ λ (but D is transposed in structure)
            Fx = self.matrix_D[:, col_u].T @ self.multipliers
            Fy = self.matrix_D[:, col_v].T @ self.multipliers

            forces[node_id] = np.array([Fx, Fy])

        return forces

    def get_constraint_residuals(self, u: np.ndarray) -> np.ndarray:
        """
        Compute constraint residuals: r = C*u.

        For exact constraint enforcement, residuals should be near machine
        precision (< 1e-12).

        Parameters
        ----------
        u : np.ndarray
            Displacement vector (n_dofs)

        Returns
        -------
        residuals : np.ndarray
            Constraint residuals (n_constraints)
            Should be ~0 for satisfied constraints
        """
        if self.constraint_matrix_C is None:
            raise RuntimeError("Constraint matrix not built.")

        residuals = self.constraint_matrix_C @ u
        return residuals

    # ============================================================
    # Large Displacement Methods (Phase 2)
    # ============================================================

    def update_constraint_matrix(self, structure) -> np.ndarray:
        """
        Update constraint matrix C for large displacement analysis (Mortar method).

        Implements 'Option B': Temporarily deforms the FEM mesh to search for
        contact points in the current configuration.
        """
        if not self.integration_points:
            raise RuntimeError("Integration points not initialized. Call build_constraint_matrix() first.")

        # Re-initialize constraint matrix
        C = np.zeros((self.n_constraints, self.n_dofs_base))

        # ============================================================
        # [NEW] STEP 0: Temporarily update FEM nodes to deformed config
        # ============================================================
        # 1. Backup original coordinates (deep copy to be safe)
        original_nodes = copy.deepcopy(structure.list_nodes)

        # 2. Deform the mesh
        # We loop over nodes and add current displacements (u, v)
        for node_id, node_coords in enumerate(structure.list_nodes):
            # Get DOF indices
            dof_offset = structure.node_dof_offsets[node_id]
            n_dofs = structure.node_dof_counts[node_id]

            # Only update translational DOFs (u, v)
            # Assuming 2D structure where first 2 DOFs are u, v
            if n_dofs >= 2:
                u_disp = structure.U[dof_offset]
                v_disp = structure.U[dof_offset + 1]

                # Create new coordinate array
                new_coords = np.array(node_coords, copy=True)
                new_coords[0] += u_disp
                new_coords[1] += v_disp

                structure.list_nodes[node_id] = new_coords

        # Use try...finally to ensure we ALWAYS restore the nodes
        try:
            # Rebuild constraint rows for each integration point
            constraint_row = 0
            points_not_found = 0

            for ip_idx, int_point in enumerate(self.integration_points):
                # Get interface and block
                interface = self.interfaces[int_point.interface_id]
                block_idx = interface.block_id
                block = structure.list_blocks[block_idx]
                block_node_id = block.connect

                # Get INITIAL integration point position and weight
                xq_init = int_point.position.copy()  # Initial position
                wq = int_point.weight  # Weight doesn't change

                # ================================================================
                # STEP 1: Update integration point position (rigid body motion)
                # ================================================================
                # Get current block displacements
                block_dof_offset = structure.node_dof_offsets[block_node_id]
                block_dofs = np.arange(block_dof_offset, block_dof_offset + 3)

                u_block = structure.U[block_dofs[0]]  # Block x-displacement
                v_block = structure.U[block_dofs[1]]  # Block y-displacement
                theta_block = structure.U[block_dofs[2]]  # Block rotation

                # Get INITIAL block reference position from our BACKUP
                block_ref_pos_init = original_nodes[block_node_id]

                # Compute relative position (from block ref to integration point)
                dx_init = xq_init[0] - block_ref_pos_init[0]
                dy_init = xq_init[1] - block_ref_pos_init[1]

                # Apply rigid body transformation to integration point
                u_xq = u_block - dy_init * theta_block
                v_xq = v_block + dx_init * theta_block

                # Current integration point position
                xq_current = xq_init + np.array([u_xq, v_xq])

                # ================================================================
                # STEP 2: Re-evaluate FEM shape functions at current position
                # ================================================================
                # NOW this works because structure.list_nodes is deformed!
                result = evaluate_shape_functions_at_point(structure, xq_current, tolerance=self.interface_tolerance)

                if result is None:
                    # Point not found in deformed FEM elements
                    points_not_found += 1
                    continue

                elem_id, N, xi, eta = result
                element = structure.list_fes[elem_id]

                # ================================================================
                # STEP 3: Assemble FEM node contributions
                # ================================================================
                for local_node_id in range(element.nd):
                    global_node_id = element.connect[local_node_id]
                    node_dof_offset = structure.node_dof_offsets[global_node_id]
                    node_dof_count = structure.node_dof_counts[global_node_id]

                    # FEM nodes have 2 DOFs
                    if node_dof_count != 2:
                        continue

                    u_dof = node_dof_offset
                    v_dof = node_dof_offset + 1

                    # Weighted shape function
                    N_weighted = N[local_node_id] * wq

                    # Add to constraint rows
                    C[constraint_row, u_dof] += N_weighted
                    C[constraint_row + 1, v_dof] += N_weighted

                # ================================================================
                # STEP 4: Add block contribution with CURRENT geometry
                # ================================================================
                # Get CURRENT block reference position
                block_ref_pos_current = block_ref_pos_init + np.array([u_block, v_block])

                # Compute CURRENT relative position
                dx_current = xq_current[0] - block_ref_pos_current[0]
                dy_current = xq_current[1] - block_ref_pos_current[1]

                # Block rigid body displacement constraints
                C[constraint_row, block_dofs[0]] -= wq  # -u_block
                C[constraint_row, block_dofs[2]] += dy_current * wq  # +dy*θ

                C[constraint_row + 1, block_dofs[1]] -= wq  # -v_block
                C[constraint_row + 1, block_dofs[2]] -= dx_current * wq  # -dx*θ

                constraint_row += 2

        finally:
            # ============================================================
            # [NEW] Restore original nodes NO MATTER WHAT
            # ============================================================
            structure.list_nodes = original_nodes

        # Trim if needed
        if constraint_row < self.n_constraints:
            C = C[:constraint_row, :]
            self.n_constraints = constraint_row
            self.n_dofs_augmented = self.n_dofs_base + self.n_constraints

        # Store updated constraint matrix
        self.constraint_matrix_C = C

        return C

    def get_geometric_stiffness(self, structure, multipliers=None) -> np.ndarray:
        """
        Compute geometric stiffness K_geo = ∂(C^T*λ)/∂u for Mortar coupling.

        For Mortar method, geometric stiffness arises from:
        1. Integration point positions changing with deformation
        2. Rigid body coupling terms (dx, dy) depending on current configuration

        This creates coupling between block rotation and translations.

        Parameters
        ----------
        structure : Hybrid
            Hybrid structure with current state
        multipliers : np.ndarray, optional
            Lagrange multipliers (default: self.multipliers)

        Returns
        -------
        K_geo : np.ndarray
            Geometric stiffness matrix (n_dofs × n_dofs)

        Notes
        -----
        **Physical Meaning**:
        When a block rotates, integration points move with it, changing:
        - Their positions relative to FEM elements
        - The rigid body coupling terms dx, dy
        - The constraint forces at those points

        **Mathematical Derivation** (for one integration point):
        Constraint force from integration point q:
            F_q = G_q^T * λ_q

        For rotation DOF:
            F_θ = -dy*λ_x + dx*λ_y  (weighted by wq)

        Geometric stiffness (derivatives):
            ∂F_θ/∂u_block = -λ_y * ∂dx/∂u_block = -λ_y
            ∂F_θ/∂v_block = λ_x * ∂dy/∂v_block = λ_x
            ∂F_θ/∂u_xq = λ_y * ∂dx/∂u_xq = λ_y
            ∂F_θ/∂v_xq = -λ_x * ∂dy/∂v_xq = -λ_x

        But u_xq depends on block DOFs through rigid body kinematics,
        so the full derivative includes these couplings.

        **Simplification**:
        For small-angle approximation (θ << 1), the geometric stiffness
        has the same structure as Lagrange method, but summed over
        integration points instead of nodes.

        **Computational Cost**:
        - O(n_integration_points) operations
        - Similar to Lagrange for typical discretizations
        """
        # Use provided multipliers or stored ones
        if multipliers is None:
            if self.multipliers is None:
                raise RuntimeError("No multipliers available. Solve system first.")
            multipliers = self.multipliers

        # Initialize geometric stiffness
        K_geo = np.zeros((self.n_dofs_base, self.n_dofs_base))

        # Build contributions from each integration point
        constraint_row = 0
        for ip_idx, int_point in enumerate(self.integration_points):
            if constraint_row >= len(multipliers):
                break  # Safety check

            # Get multipliers for this integration point
            lambda_x = multipliers[constraint_row]      # From g_u constraint
            lambda_y = multipliers[constraint_row + 1]  # From g_v constraint

            # Get interface and block
            interface = self.interfaces[int_point.interface_id]
            block_idx = interface.block_id
            block = structure.list_blocks[block_idx]
            block_node_id = block.connect

            # Get block DOFs
            block_dof_offset = structure.node_dof_offsets[block_node_id]
            block_dofs = np.arange(block_dof_offset, block_dof_offset + 3)

            u_block_dof = block_dofs[0]
            v_block_dof = block_dofs[1]
            theta_dof = block_dofs[2]

            # Get integration point weight
            wq = int_point.weight

            # Geometric stiffness from rotation-displacement coupling
            # Similar structure to Lagrange, but include weight wq
            #
            # For constraint g₁: Σ Nᵢ·uᵢ - u_block - dy*θ = 0
            #   Force on θ: -dy*λ_x * wq
            #   ∂F_θ/∂v_block = λ_x * wq  (dy changes with v_block)
            #
            # For constraint g₂: Σ Nᵢ·vᵢ - v_block - dx*θ = 0
            #   Force on θ: dx*λ_y * wq
            #   ∂F_θ/∂u_block = -λ_y * wq  (dx changes with u_block)

            # Weighted contributions
            lambda_x_weighted = lambda_x * wq
            lambda_y_weighted = lambda_y * wq

            # Rotation-translation coupling (block DOFs only for Mortar)
            K_geo[theta_dof, u_block_dof] += -lambda_y_weighted
            K_geo[theta_dof, v_block_dof] += lambda_x_weighted

            # Symmetrize
            K_geo[u_block_dof, theta_dof] += -lambda_y_weighted
            K_geo[v_block_dof, theta_dof] += lambda_x_weighted

            constraint_row += 2

        return K_geo

    def verify_constraints(self, structure):
        """
        Verify mortar constraint satisfaction.

        The weak constraint is satisfied when C @ u ≈ 0.
        """
        if self.constraint_matrix_C is None:
            return 0.0

        C = self.constraint_matrix_C
        u = structure.U

        residual = C @ u
        residual_norm = np.linalg.norm(residual)

        return residual_norm

    # ============================================================
    # Information and Diagnostics
    # ============================================================

    def get_info(self) -> Dict:
        """
        Get detailed information about the mortar coupling configuration.

        Returns
        -------
        info : Dict
            Dictionary with coupling details
        """
        base_info = super().get_info()

        # Count slave edges across all interfaces
        n_slave_edges = sum(len(iface.slave_edges) for iface in self.interfaces
                          if hasattr(iface, 'slave_edges'))

        mortar_info = {
            'method': 'true_mortar',  # Indicate this is the true mortar implementation
            'integration_order': self.integration_order,
            'interface_tolerance': self.interface_tolerance,
            'n_interfaces': len(self.interfaces),
            'n_slave_nodes': len(self.slave_nodes_global),
            'n_slave_edges': n_slave_edges,
            'n_constraints': self.n_constraints,
            'n_dofs_base': self.n_dofs_base,
            'n_dofs_augmented': self.n_dofs_augmented,
            'constraint_matrix_built': self.constraint_matrix_C is not None,
            'D_matrix_built': self.matrix_D is not None,
            'M_matrix_built': self.matrix_M is not None,
            'solution_available': self.multipliers is not None,
        }

        if self.matrix_D is not None:
            mortar_info['D_matrix_rank'] = int(np.linalg.matrix_rank(self.matrix_D))
        if self.matrix_M is not None:
            mortar_info['M_matrix_rank'] = int(np.linalg.matrix_rank(self.matrix_M))

        if self.multipliers is not None:
            mortar_info['max_multiplier'] = float(np.max(np.abs(self.multipliers)))
            mortar_info['multiplier_norm'] = float(np.linalg.norm(self.multipliers))

        return {**base_info, **mortar_info}

    def __repr__(self) -> str:
        """String representation."""
        status = "active" if self.active else "inactive"
        n_ifaces = len(self.interfaces)
        n_slave_nodes = len(self.slave_nodes_global)
        n_const = self.n_constraints

        return (f"MortarCoupling(status={status}, "
                f"method=true_mortar, "
                f"interfaces={n_ifaces}, "
                f"slave_nodes={n_slave_nodes}, "
                f"constraints={n_const}, "
                f"augmented_dofs={self.n_dofs_augmented})")
