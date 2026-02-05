"""
Structure_Hybrid - Hybrid Block-FEM Structures with Coupling
=============================================================

This module implements hybrid structures that combine discrete rigid blocks
(DFEM) with continuous finite elements (FEM). It provides four coupling methods
to enforce compatibility between the two domains.

Key Concepts for Students:
--------------------------
1. **Multiple Inheritance**: Hybrid inherits from both Structure_Block and
   Structure_FEM, combining their capabilities.

2. **Variable DOF per Node**: Block nodes have 3 DOFs [ux, uy, rz], while
   FEM nodes have 2 DOFs [ux, uy]. The node_dof_offsets tracks this.

3. **Coupling Methods** (see Documentation/03_core_concepts.md for details):
   - 'constraint': DOF elimination via transformation matrix T (fastest)
   - 'penalty': Virtual springs at coupled nodes (simple, approximate)
   - 'lagrange': Lagrange multipliers (exact, saddle-point system)
   - 'mortar': Distributed multipliers for non-matching meshes (most general)

4. **Coupling Workflow**:
   a. Build structure: add blocks and elements
   b. Call make_nodes() to create DOF system
   c. Call enable_block_fem_coupling(method='...') to activate coupling
   d. Solve with appropriate solver (linear or saddle-point)

5. **Constraint Equation**: u_fem = C * q_block
   where C is the rigid body constraint matrix relating FEM node motion
   to block reference point motion (translation + rotation effect).

   For a FEM node at (x, y) coupled to block with ref point at (x0, y0):
   [ u_fem ]   [ 1  0  -(y-y0) ] [ u_block ]
   [ v_fem ] = [ 0  1   (x-x0) ] [ v_block ]
                                 [ θ_block ]

Typical Usage:
    >>> St = Hybrid()
    >>> St.add_block(block)                          # Add blocks
    >>> St.list_fes.append(element)                  # Add FEM elements
    >>> St.make_nodes()                              # Build DOF system
    >>> St.detect_interfaces()                       # Find block contacts
    >>> St.make_cfs(contact=contact_law)             # Create contact faces
    >>> St.enable_block_fem_coupling(method='constraint')  # Enable coupling
    >>> St = Static.solve_linear(St)                 # Solve
"""

# Standard imports
import warnings
from typing import Dict

import numpy as np

from Core.Structures.Structure_2D import Structure_2D
from Core.Structures.Structure_Block import Structure_Block
from Core.Structures.Structure_FEM import Structure_FEM


class Hybrid(Structure_Block, Structure_FEM):
    """
    Hybrid structure combining discrete blocks and continuous FEM elements.

    This class enables analysis of structures where part of the domain is
    modeled with rigid blocks (e.g., masonry units) and part with continuous
    FEM elements (e.g., mortar, infill, or surrounding structure).

    Supports four coupling methods: constraint, penalty, Lagrange, and mortar.

    Attributes
    ----------
    list_blocks : List[Block_2D]
        Rigid blocks in the hybrid structure.
    list_fes : List[BaseFE]
        Finite elements in the hybrid structure.
    coupling_enabled : bool
        Whether block-FEM coupling is active.
    """

    def __init__(self, fixed_dofs_per_node: bool = False, merge_coincident_nodes: bool = True):
        """
        Initialize hybrid structure.

        Parameters
        ----------
        fixed_dofs_per_node : bool
            If True, all nodes have 3 DOFs.
        merge_coincident_nodes : bool
            If True, nodes at same position are merged.
        """
        # Bypass parent __init__ to avoid MRO complications
        Structure_2D.__init__(self, structure_type="HYBRID")

        # Initialize attributes from both parent classes
        self.list_blocks = []
        self.list_cfs = []
        self.list_fes = []
        self.fixed_dofs_per_node = fixed_dofs_per_node
        self.merge_coincident_nodes = merge_coincident_nodes

        # Hybrid-specific attributes
        self.list_hybrid_cfs = []
        self.hybrid_tolerance = 1e-6
        self.hybrid_n_integration_points = 2

        # Coupling infrastructure
        self.coupling_enabled = False
        self.coupled_fem_nodes = {}  # {fem_node_id: block_id}
        self.coupling_T = None  # Transformation matrix: u_full = T * u_reduced
        self.coupling_dof_map = None
        self.nb_dofs_reduced = None
        self.nb_dofs_full = None

        # Coupling method instances (only one active at a time)
        self.constraint_coupling = None
        self.penalty_coupling = None
        self.lagrange_coupling = None
        self.mortar_coupling = None

    def make_nodes(self):
        """Build node list from both blocks and FEM elements."""
        self.list_nodes = []

        self._make_nodes_block()
        self._make_nodes_fem()

        self.nb_dofs = self.compute_nb_dofs()
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = len(self.dof_free)

    def detect_coupled_fem_nodes(self, tolerance: float = 1e-9) -> Dict[int, int]:
        """
        Detect which FEM nodes are coupled to rigid blocks.

        Parameters
        ----------
        tolerance : float
            Distance tolerance for node matching.

        Returns
        -------
        dict
            {fem_node_id: block_id} mapping.
        """
        coupled_nodes = {}

        # Block nodes are the first len(self.list_blocks) nodes
        n_blocks = len(self.list_blocks)

        for block_idx in range(n_blocks):
            block = self.list_blocks[block_idx]
            block_node_id = block.connect  # Global node ID for this block

            # Check all FEM nodes for proximity to this block
            for fem_node_id in range(n_blocks, len(self.list_nodes)):
                fem_node_pos = self.list_nodes[fem_node_id]
                block_node_pos = self.list_nodes[block_node_id]

                dist = np.linalg.norm(fem_node_pos - block_node_pos)

                if dist <= tolerance:
                    coupled_nodes[fem_node_id] = block_idx
                    break  # A FEM node can only couple to one block

        return coupled_nodes

    def build_coupling_transformation(self):
        """Build transformation matrix T for block-FEM coupling."""

        # Import ConstraintCoupling
        from Core.Objects.Coupling import Condensation

        # Create ConstraintCoupling instance if not exists
        if self.constraint_coupling is None:
            self.constraint_coupling = Condensation()

        # Add all coupled nodes with their constraint matrices
        for fem_node_id, block_idx in self.coupled_fem_nodes.items():
            block = self.list_blocks[block_idx]
            fem_node_pos = self.list_nodes[fem_node_id]

            # Get constraint matrix: u_fem = C * q_block
            C = block.constraint_matrix_for_node(fem_node_pos)

            # Register with coupling object
            self.constraint_coupling.add_coupled_node(fem_node_id, block_idx, C)

        # Build transformation matrix using ConstraintCoupling
        self.constraint_coupling.build_transformation(self)

        if not hasattr(self, '_original_global_dof'):
            self._original_global_dof = self._global_dof

        def mapped_global_dof(node_id, dof_idx):
            """Map (node, dof) to reduced matrix index."""
            full_idx = self._original_global_dof(node_id, dof_idx)

            if full_idx < 0 or full_idx >= self.coupling_T.shape[0]:
                return -1

            row = self.coupling_T[full_idx, :]
            reduced_indices = np.where(np.isclose(row, 1.0))[0]

            if len(reduced_indices) == 1:
                return reduced_indices[0]

            return -1

        self._global_dof = mapped_global_dof

        # Extract results from ConstraintCoupling
        self.coupling_T = self.constraint_coupling.transformation_matrix
        self.nb_dofs_full = self.constraint_coupling.nb_dofs_full
        self.nb_dofs_reduced = self.constraint_coupling.nb_dofs_reduced

        # Update DOF count for solver
        self.nb_dofs = self.nb_dofs_reduced
        self.coupling_enabled = True

        # Transform load vectors from full space to reduced space
        # P_reduced = T^T * P_full (same transformation as K)
        P_full = self.P.copy() if hasattr(self, 'P') and self.P is not None else np.zeros(self.nb_dofs_full)
        self.P = self.coupling_T.T @ P_full

        if hasattr(self, 'P_fixed') and self.P_fixed is not None:
            P_fixed_full = self.P_fixed.copy()
            self.P_fixed = self.coupling_T.T @ P_fixed_full
        else:
            self.P_fixed = np.zeros(self.nb_dofs, dtype=float)

        # Reset displacement and residual (no solution yet)
        self.U = np.zeros(self.nb_dofs, dtype=float)
        if hasattr(self, 'P_r'):
            self.P_r = np.zeros(self.nb_dofs, dtype=float)

        # Map fixed DOFs from full space to reduced space
        dof_fix_full = self.dof_fix.copy() if hasattr(self, 'dof_fix') else np.array([], dtype=int)

        # Find which reduced DOFs correspond to fixed full DOFs
        dof_fix_reduced = []
        T = self.coupling_T
        for full_dof in dof_fix_full:
            # Check which reduced DOFs this full DOF maps to (non-zero entries in T)
            reduced_dofs_for_this_full_dof = np.where(np.abs(T[full_dof, :]) > 1e-10)[0]
            # If it maps 1-to-1 to a single reduced DOF, fix that reduced DOF
            if len(reduced_dofs_for_this_full_dof) == 1:
                red_dof = reduced_dofs_for_this_full_dof[0]
                if red_dof not in dof_fix_reduced:
                    dof_fix_reduced.append(red_dof)

        self.dof_fix = np.array(dof_fix_reduced, dtype=int)
        self.dof_free = np.setdiff1d(np.arange(self.nb_dofs_reduced, dtype=int), self.dof_fix)
        self.nb_dof_fix = len(self.dof_fix)
        self.nb_dof_free = len(self.dof_free)

    def enable_block_fem_coupling(self, tolerance: float = 1e-9, method: str = 'constraint',
                                 penalty=None, integration_order: int = 2,
                                  interface_tolerance: float = 1e-4,
                                  interface_orientation: str = 'horizontal'):
        """
        Enable coupling between blocks and FEM elements.

        Parameters
        ----------
        tolerance : float
            Distance tolerance for detecting coupled nodes.
        method : str
            'constraint', 'penalty', 'lagrange', or 'mortar'.
        penalty : None, 'auto', or float
            Penalty stiffness (for method='penalty').
        integration_order : int
            Gauss quadrature order for mortar method.
        interface_tolerance : float
            Distance tolerance for interface detection (mortar).
        interface_orientation : str
            Filter interfaces: 'horizontal', 'vertical', or None.
        """

        # Validate method parameter
        valid_methods = ['constraint', 'penalty', 'lagrange', 'mortar']
        if method not in valid_methods:
            raise ValueError(f"Unknown coupling method: '{method}'. "
                           f"Valid options: {valid_methods}")

        # Step 1: Detect coupled nodes (for nodal methods: constraint, penalty, lagrange)
        # Mortar method uses interface detection instead, so skip this check for mortar
        if method != 'mortar':
            self.coupled_fem_nodes = self.detect_coupled_fem_nodes(tolerance=tolerance)

            if not self.coupled_fem_nodes:
                self.coupling_enabled = False
                return
        else:
            # Mortar doesn't use coupled nodes - it detects interfaces
            self.coupled_fem_nodes = {}

        # Step 2: Configure method-specific coupling
        if method == 'constraint':
            # DOF elimination via transformation matrix T
            self.build_coupling_transformation()
            coupling_name = "Constraint (DOF elimination)"

        elif method == 'penalty':
            # Penalty stiffness springs
            self.coupling_enabled = True
            # Handle penalty parameter backward compatibility
            if penalty is None:
                penalty = 'auto'  # Default to auto-scaling
            self._configure_penalty_coupling(penalty=penalty)
            coupling_name = "Penalty (stiffness springs)"

        elif method == 'lagrange':
            # Lagrange multipliers
            self.coupling_enabled = True
            self._configure_lagrange_coupling()
            coupling_name = "Lagrange (multipliers)"

        elif method == 'mortar':
            # Mortar method (distributed Lagrange multipliers)
            self.coupling_enabled = True
            self._configure_mortar_coupling(
                integration_order=integration_order,
                interface_tolerance=interface_tolerance,
                interface_orientation=interface_orientation
            )
            coupling_name = "Mortar (distributed multipliers)"

    def _configure_penalty_coupling(self, penalty):
        """Configure penalty stiffness coupling."""
        from Core.Objects.Coupling import PenaltyCoupling, compute_rigid_body_constraint


        # Determine penalty factor
        if penalty == 'auto':
            penalty_factor = 1000.0  # Default auto value
            auto_scale = True
        else:
            penalty_factor = float(penalty)
            auto_scale = False

        # Create PenaltyCoupling instance
        self.penalty_coupling = PenaltyCoupling(
            penalty_factor=penalty_factor,
            auto_scale=auto_scale
        )

        # Estimate base stiffness for auto-scaling
        if auto_scale:
            # Get typical material properties from FEM elements
            E_typical = 200e9  # Default: steel
            t_typical = 0.01   # Default: 10mm
            L_typical = 1.0    # Default: 1m

            # Try to extract from actual FEM elements
            if len(self.list_fes) > 0:
                first_elem = self.list_fes[0]
                if hasattr(first_elem, 'geom') and hasattr(first_elem.geom, 't'):
                    t_typical = first_elem.geom.t

            self.penalty_coupling.set_base_stiffness(E=E_typical, t=t_typical, L=L_typical)
        else:
            # Explicit stiffness provided: k_p = factor * base -> k_p = penalty * 1.0
            self.penalty_coupling.base_stiffness = 1.0

        # Add constraint matrices for all coupled nodes
        n_blocks = len(self.list_blocks)

        for fem_node_id, block_idx in self.coupled_fem_nodes.items():
            block = self.list_blocks[block_idx]

            # Get positions
            fem_node_pos = self.list_nodes[fem_node_id]
            block_node_id = block.connect
            block_ref_pos = self.list_nodes[block_node_id]

            # Compute constraint matrix
            C = compute_rigid_body_constraint(
                node_position=fem_node_pos,
                block_ref_point=block_ref_pos,
                small_angle=True
            )

            # Register with penalty coupling
            self.penalty_coupling.add_coupled_node(
                fem_node_id=fem_node_id,
                block_idx=block_idx,
                constraint_matrix=C
            )

        # Activate penalty coupling
        self.penalty_coupling.activate()

    def _configure_lagrange_coupling(self):
        """Configure Lagrange multiplier coupling."""
        from Core.Objects.Coupling import LagrangeCoupling



        # Create LagrangeCoupling instance
        self.lagrange_coupling = LagrangeCoupling()

        # Add all coupled nodes with their constraint matrices
        for fem_node_id, block_idx in self.coupled_fem_nodes.items():
            block = self.list_blocks[block_idx]

            # Get positions
            fem_node_pos = self.list_nodes[fem_node_id]
            block_node_id = block.connect
            block_ref_pos = self.list_nodes[block_node_id]

            # Compute constraint matrix C (2×3): u_fem = C * [u_block, v_block, θ]
            # This is used to construct the global G matrix
            C = block.constraint_matrix_for_node(fem_node_pos)

            # Register with Lagrange coupling
            self.lagrange_coupling.add_coupled_node(
                fem_node_id=fem_node_id,
                block_idx=block_idx,
                constraint_matrix=C
            )

        # Build global constraint matrix G
        self.lagrange_coupling.build_constraint_matrix(self)

        # Activate Lagrange coupling
        self.lagrange_coupling.activate()

    def _configure_mortar_coupling(self, integration_order: int = 2,
                                   interface_tolerance: float = 1e-4,
                                   interface_orientation: str = 'horizontal'):
        """Configure mortar method coupling."""
        from Core.Objects.Coupling import MortarCoupling

        # Create MortarCoupling instance
        self.mortar_coupling = MortarCoupling(
            integration_order=integration_order,
            interface_tolerance=interface_tolerance,
            interface_orientation=interface_orientation
        )

        # Build global constraint matrix G
        # This automatically:
        # 1. Detects interfaces between blocks and FEM
        # 2. Generates integration points along interfaces
        # 3. Evaluates FEM shape functions at integration points
        # 4. Assembles G matrix via numerical integration
        self.mortar_coupling.build_constraint_matrix(self)

        # Activate mortar coupling
        self.mortar_coupling.activate()

    def _assemble_penalty_coupling(self, K: np.ndarray):
        """Assemble penalty coupling contributions into stiffness matrix."""
        if self.penalty_coupling is None or not self.penalty_coupling.active:
            return

        n_blocks = len(self.list_blocks)

        # Use full DOF count if coupling is enabled (before transformation)
        n_dofs = K.shape[0]

        for fem_node_id, block_idx in self.coupled_fem_nodes.items():
            block = self.list_blocks[block_idx]

            # Get DOF indices for FEM node (2 DOFs: ux, uy)
            dof_offset = self.node_dof_offsets[fem_node_id]
            dof_count = self.node_dof_counts[fem_node_id]
            fem_dofs = np.arange(dof_offset, dof_offset + dof_count, dtype=int)

            # Get DOF indices for block (3 DOFs: ux, uy, rz)
            block_dofs = block.dofs

            # Get constraint matrix
            C = self.penalty_coupling.constraint_matrices[(fem_node_id, block_idx)]

            # Compute penalty coupling stiffness
            K_coupling, _ = self.penalty_coupling.compute_coupling_stiffness(
                fem_node_dofs=fem_dofs,
                block_dofs=block_dofs,
                constraint_matrix=C,
                n_dofs=n_dofs
            )

            # Add to global stiffness matrix
            K += K_coupling

    def expand_displacement(self, u_reduced: np.ndarray) -> np.ndarray:
        """
        Expand displacement from reduced DOFs to full DOFs.

        Parameters
        ----------
        u_reduced : np.ndarray
            Displacement vector in reduced DOF space.

        Returns
        -------
        np.ndarray
            Displacement vector in full DOF space.
        """
        if not self.coupling_enabled:
            # No coupling - just return as-is
            return u_reduced

        if not hasattr(self, 'coupling_T'):
            raise RuntimeError("Coupling transformation matrix not built")

        # Apply transformation: u_full = T * u_reduced
        u_full = self.coupling_T @ u_reduced

        return u_full

    def _get_P_r_hybrid(self):
        """Helper to add hybrid coupling forces to internal force vector."""
        if not hasattr(self, 'list_hybrid_cfs'):
            return

        if not hasattr(self, 'P_r'):
            raise RuntimeError("Residual P_r not initialized")

        if not hasattr(self, 'U'):
            warnings.warn("Displacement U not found - cannot compute coupling forces")
            return

        for cf in self.list_hybrid_cfs:
            f_cf, dof_indices = cf.get_pf_glob(self.U)

            if f_cf is not None:
                dof_array = np.array(dof_indices)
                self.P_r[dof_array] += f_cf

    def get_P_r(self):
        """Compute internal force vector for hybrid structure."""
        self.dofs_defined()

        # For constraint coupling, need to work in full space
        if self.coupling_enabled and self.constraint_coupling is not None:
            # Save reduced U and expand to full space
            U_reduced = self.U.copy()
            self.U = self.expand_displacement(U_reduced)

            # Assemble residual in full space
            nb_dofs_full = self.nb_dofs_full if hasattr(self,
                                                        'nb_dofs_full') and self.nb_dofs_full is not None else self.nb_dofs
            self.P_r = np.zeros(nb_dofs_full, dtype=float)
            self._get_P_r_block()
            self._get_P_r_fem()
            self._get_P_r_hybrid()

            # Transform residual to reduced space
            P_full = self.P_r
            self.P_r = self.coupling_T.T @ P_full

            # Restore reduced U
            self.U = U_reduced
        else:
            # No coupling, penalty coupling, or Lagrange: work in current space
            self.P_r = np.zeros(self.nb_dofs, dtype=float)
            self._get_P_r_block()
            self._get_P_r_fem()
            self._get_P_r_hybrid()

        return self.P_r

    def get_M_str(self, no_inertia: bool = False):
        """Assemble mass matrix for hybrid structure."""
        self.dofs_defined()

        # Assemble full mass matrix
        if self.coupling_enabled and self.constraint_coupling is not None:
            # Constraint-based coupling: use full DOF space for transformation
            nb_dofs_full = self.nb_dofs_full if hasattr(self,
                                                        'nb_dofs_full') and self.nb_dofs_full is not None else self.nb_dofs
            self.M = np.zeros((nb_dofs_full, nb_dofs_full), dtype=float)
        else:
            # No coupling or penalty coupling: use current DOF count
            self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)

        # Compose contributions
        self._mass_block(no_inertia=no_inertia)
        self._mass_fem(no_inertia=no_inertia)

        # Apply coupling transformation only for constraint-based coupling
        if self.coupling_enabled and self.constraint_coupling is not None:
            M_full = self.M
            self.M = self.coupling_T.T @ M_full @ self.coupling_T

        return self.M

    def _stiffness_hybrid(self):
        """Helper to add hybrid coupling stiffness to global matrix."""
        if not hasattr(self, 'list_hybrid_cfs'):
            return

        if not hasattr(self, 'K'):
            raise RuntimeError("Global stiffness K not initialized")

        for cf in self.list_hybrid_cfs:
            K_cf, dof_indices = cf.get_kf_glob(getattr(self, 'U', None))

            if K_cf is not None:
                dof_array = np.array(dof_indices)
                self.K[np.ix_(dof_array, dof_array)] += K_cf

    def get_K_str(self):
        """Assemble tangent stiffness matrix for hybrid structure."""
        self.dofs_defined()

        # Assemble full stiffness matrix
        if self.coupling_enabled and self.constraint_coupling is not None:
            # Constraint-based coupling: use full DOF space for transformation
            nb_dofs_full = self.nb_dofs_full if hasattr(self,
                                                        'nb_dofs_full') and self.nb_dofs_full is not None else self.nb_dofs
            self.K = np.zeros((nb_dofs_full, nb_dofs_full), dtype=float)
        else:
            # No coupling, penalty coupling, or Lagrange: use current DOF count
            self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)

        self._stiffness_block()
        self._stiffness_fem()
        self._stiffness_hybrid()

        # Add penalty coupling contributions (works in full DOF space)
        if self.coupling_enabled and self.penalty_coupling is not None:
            self._assemble_penalty_coupling(self.K)

        # Apply coupling transformation only for constraint-based coupling
        if self.coupling_enabled and self.constraint_coupling is not None:
            K_full = self.K
            self.K = self.coupling_T.T @ K_full @ self.coupling_T

        # Note: Lagrange coupling augmentation happens in solver, not here

        return self.K

    def get_K_str0(self):
        """Assemble initial stiffness matrix for hybrid structure."""
        self.dofs_defined()

        # Assemble full initial stiffness matrix
        if self.coupling_enabled and self.constraint_coupling is not None:
            # Constraint-based coupling: use full DOF space for transformation
            nb_dofs_full = self.nb_dofs_full if hasattr(self,
                                                        'nb_dofs_full') and self.nb_dofs_full is not None else self.nb_dofs
            self.K0 = np.zeros((nb_dofs_full, nb_dofs_full), dtype=float)
        else:
            # No coupling, penalty coupling, or Lagrange: use current DOF count
            self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)

        self._stiffness0_block()
        self._stiffness0_fem()

        # Add penalty coupling contributions (works in full DOF space)
        if self.coupling_enabled and self.penalty_coupling is not None:
            self._assemble_penalty_coupling(self.K0)

        # Apply coupling transformation only for constraint-based coupling
        if self.coupling_enabled and self.constraint_coupling is not None:
            K0_full = self.K0
            self.K0 = self.coupling_T.T @ K0_full @ self.coupling_T

        # Note: Lagrange coupling augmentation happens in solver, not here

        return self.K0

    def get_K_str_LG(self):
        """Assemble geometric stiffness matrix for hybrid structure."""
        self.dofs_defined()

        # Assemble full large geometry stiffness matrix
        if self.coupling_enabled and self.constraint_coupling is not None:
            # Constraint-based coupling: use full DOF space for transformation
            nb_dofs_full = self.nb_dofs_full if hasattr(self,
                                                        'nb_dofs_full') and self.nb_dofs_full is not None else self.nb_dofs
            self.K_LG = np.zeros((nb_dofs_full, nb_dofs_full), dtype=float)
        else:
            # No coupling or penalty coupling: use current DOF count
            self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)

        self._stiffness_LG_block()
        self._stiffness_LG_fem()

        # Add penalty coupling contributions (works in full DOF space)
        if self.coupling_enabled and self.penalty_coupling is not None:
            self._assemble_penalty_coupling(self.K_LG)

        # Apply coupling transformation only for constraint-based coupling
        if self.coupling_enabled and self.constraint_coupling is not None:
            K_LG_full = self.K_LG
            self.K_LG = self.coupling_T.T @ K_LG_full @ self.coupling_T

        return self.K_LG

    def set_lin_geom(self, lin_geom=True):
        """Enable or disable linear geometry assumption."""
        for cf in self.list_cfs:
            cf.set_lin_geom(lin_geom)

        for fe in self.list_fes:
            fe.lin_geom = lin_geom
