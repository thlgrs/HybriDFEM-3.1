import warnings
from typing import Dict, Tuple, Optional

import numpy as np

from .BaseCoupling import BaseCoupling
from .ConstraintMatrix import compute_constraint_violation


class PenaltyCoupling(BaseCoupling):
    """
    Penalty stiffness coupling method.

    Adds penalty springs at the interface to enforce kinematic compatibility.
    The penalty stiffness is k_p = penalty_factor * base_stiffness.

    Attributes
    ----------
    penalty_factor : float
        Dimensionless multiplier (typical: 1e2 to 1e4)
    base_stiffness : float
        Reference stiffness (E*t/L)
    penalty_stiffness : float
        Actual spring stiffness k_p
    """

    def __init__(
        self,
            penalty_factor: float = 1e4,
        base_stiffness: Optional[float] = None,
            auto_scale: bool = False
    ):
        """
        Parameters
        ----------
        penalty_factor : float
            Dimensionless multiplier (default: 1000)
        base_stiffness : float, optional
            Reference stiffness. If None, computed from material properties
        auto_scale : bool
            Automatically compute base_stiffness
        """
        super().__init__(coupling_type="penalty")

        self.penalty_factor = penalty_factor
        self.base_stiffness = base_stiffness
        self.auto_scale = auto_scale

        # Computed penalty stiffness (set during activate())
        self.penalty_stiffness = None

        self.condition_number_warning_threshold = 1e12
        self.last_coupling_energy = 0.0

    def get_system_modification_type(self) -> str:
        return 'matrix_addition'

    def get_dof_count_change(self, n_base_dofs: int) -> int:
        return 0

    def requires_special_solver(self) -> bool:
        return False

    def set_base_stiffness(self, E: float, t: float, L: float):
        """Set base stiffness: k_base = E * t * L."""
        self.base_stiffness = E * t * L

    def compute_penalty_stiffness(self) -> float:
        """Compute k_p = penalty_factor * base_stiffness."""
        if self.base_stiffness is None:
            if self.auto_scale:
                warnings.warn(
                    "Auto-scaling requested but no material properties available. "
                    "Using default base_stiffness = 1e6 N/m. "
                    "Call set_base_stiffness() for better results."
                )
                self.base_stiffness = 1e6  # Default fallback
            else:
                raise ValueError(
                    "base_stiffness not set. Either provide base_stiffness "
                    "or enable auto_scale and call set_base_stiffness()."
                )

        k_p = self.penalty_factor * self.base_stiffness

        return k_p

    def activate(self):
        """Activate coupling and compute penalty stiffness."""
        self.penalty_stiffness = self.compute_penalty_stiffness()
        super().activate()

    def compute_coupling_stiffness(
        self,
        fem_node_dofs: np.ndarray,
        block_dofs: np.ndarray,
        constraint_matrix: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute penalty coupling stiffness contribution.

        Adds to global stiffness:
            K[fem,fem] += k_p*I, K[fem,block] += -k_p*C,
            K[block,fem] += -k_p*C', K[block,block] += k_p*C'*C

        Returns (K_coupling, coupling_info).
        """
        if not self.active or self.penalty_stiffness is None:
            raise RuntimeError("PenaltyCoupling must be activated before use. Call activate().")

        n_dofs = kwargs.get('n_dofs')
        if n_dofs is None:
            raise ValueError("Must provide 'n_dofs' in kwargs")

        # Initialize sparse coupling matrix
        K_coupling = np.zeros((n_dofs, n_dofs), dtype=float)

        # Extract penalty stiffness
        k_p = self.penalty_stiffness
        C = constraint_matrix  # 2×3 matrix

        # Check dimensions
        assert fem_node_dofs.shape == (2,), f"FEM node must have 2 DOFs, got {fem_node_dofs.shape}"
        assert block_dofs.shape == (3,), f"Block must have 3 DOFs, got {block_dofs.shape}"
        assert C.shape == (2, 3), f"Constraint matrix must be 2×3, got {C.shape}"

        # Penalty stiffness blocks
        I2 = np.eye(2)
        K_ff = k_p * I2                  # 2×2: FEM-FEM coupling
        K_fb = -k_p * C                  # 2×3: FEM-Block coupling
        K_bf = -k_p * C.T                # 3×2: Block-FEM coupling
        K_bb = k_p * (C.T @ C)           # 3×3: Block-Block coupling

        # Assemble into global matrix
        # FEM-FEM block
        for i in range(2):
            for j in range(2):
                K_coupling[fem_node_dofs[i], fem_node_dofs[j]] += K_ff[i, j]

        # FEM-Block block
        for i in range(2):
            for j in range(3):
                K_coupling[fem_node_dofs[i], block_dofs[j]] += K_fb[i, j]

        # Block-FEM block
        for i in range(3):
            for j in range(2):
                K_coupling[block_dofs[i], fem_node_dofs[j]] += K_bf[i, j]

        # Block-Block block
        for i in range(3):
            for j in range(3):
                K_coupling[block_dofs[i], block_dofs[j]] += K_bb[i, j]

        # Diagnostics
        coupling_info = {
            'penalty_stiffness': k_p,
            'K_ff': K_ff.copy(),
            'K_fb': K_fb.copy(),
            'K_bf': K_bf.copy(),
            'K_bb': K_bb.copy(),
            'fem_dofs': fem_node_dofs.copy(),
            'block_dofs': block_dofs.copy(),
            'C_matrix': C.copy()
        }

        return K_coupling, coupling_info

    def compute_coupling_forces(
        self,
        u_global: np.ndarray,
        fem_node_dofs: np.ndarray,
        block_dofs: np.ndarray,
        constraint_matrix: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute penalty forces from displacement state.

        F_penalty = k_p * (u_fem - C @ u_block)

        Returns (F_coupling, force_info).
        """
        if not self.active or self.penalty_stiffness is None:
            raise RuntimeError("PenaltyCoupling must be activated before use.")

        n_dofs = len(u_global)
        F_coupling = np.zeros(n_dofs, dtype=float)

        k_p = self.penalty_stiffness
        C = constraint_matrix

        # Extract displacements
        u_fem = u_global[fem_node_dofs]      # [u, v]
        u_block = u_global[block_dofs]       # [u, v, θ]

        # Compute constraint violation
        delta, violation_magnitude = compute_constraint_violation(u_fem, u_block, C)

        # Penalty forces (proportional to violation)
        # F_fem = k_p * delta
        # F_block = -k_p * C^T @ delta
        F_fem = k_p * delta
        F_block = -k_p * (C.T @ delta)

        # Assemble into global force vector
        F_coupling[fem_node_dofs] += F_fem
        F_coupling[block_dofs] += F_block

        # Compute penalty energy
        penalty_energy = 0.5 * k_p * violation_magnitude**2
        self.last_coupling_energy = penalty_energy

        # Diagnostics
        force_info = {
            'violation_magnitude': violation_magnitude,
            'violation_vector': delta,
            'penalty_energy': penalty_energy,
            'fem_force': F_fem,
            'block_force': F_block
        }

        return F_coupling, force_info

    def compute_constraint_errors(self, structure, u):
        """
        Compute constraint violation for each coupling pair.

        Parameters
        ----------
        structure : Hybrid
            The hybrid structure instance.
        u : ndarray (n,)
            Global displacement vector.

        Returns
        -------
        errors : dict
            {fem_node_id: error_norm} for each coupled node.
        max_error : float
            Maximum constraint violation (L2 norm).
        """
        errors = {}
        max_error = 0.0

        for (fem_node_id, block_idx), C in self.constraint_matrices.items():
            # Get DOF indices
            fem_dofs_idx = structure.node_dof_offsets[fem_node_id]
            u_fem = u[fem_dofs_idx: fem_dofs_idx + 2]

            block = structure.list_blocks[block_idx]
            block_dofs_idx = structure.node_dof_offsets[block.connect]
            q_block = u[block_dofs_idx: block_dofs_idx + 3]

            # Constraint residual: g = u_fem - C @ q_block
            g_ij = u_fem - C @ q_block
            error = np.linalg.norm(g_ij)

            errors[fem_node_id] = error
            max_error = max(max_error, error)

        return errors, max_error

    def estimate_interface_forces(self, structure, u):
        """
        Estimate interface forces from constraint violations.

        Parameters
        ----------
        structure : Hybrid
            The hybrid structure instance.
        u : ndarray (n,)
            Global displacement vector.

        Returns
        -------
        forces : dict
            {fem_node_id: lambda_approx} estimated interface forces.
        """
        if self.penalty_stiffness is None:
            self.penalty_stiffness = self.compute_penalty_stiffness()

        forces = {}

        for (fem_node_id, block_idx), C in self.constraint_matrices.items():
            # Get DOF indices
            fem_dofs_idx = structure.node_dof_offsets[fem_node_id]
            u_fem = u[fem_dofs_idx: fem_dofs_idx + 2]

            block = structure.list_blocks[block_idx]
            block_dofs_idx = structure.node_dof_offsets[block.connect]
            q_block = u[block_dofs_idx: block_dofs_idx + 3]

            # Interface force estimate: lambda approx alpha * g
            g_ij = u_fem - C @ q_block
            forces[fem_node_id] = self.penalty_stiffness * g_ij

        return forces

    def __repr__(self):
        status = "active" if self.active else "inactive"
        k_p_str = f"{self.penalty_stiffness:.2e}" if self.penalty_stiffness else "not computed"
        return (f"PenaltyCoupling({status}, α={self.penalty_factor}, "
                f"k_p={k_p_str} N/m, nodes={len(self.coupled_nodes)})")
