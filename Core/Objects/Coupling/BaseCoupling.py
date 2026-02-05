from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


class BaseCoupling(ABC):
    """
    Abstract base class for block-FEM coupling methods.

    Provides shared infrastructure: coupled node registry, constraint matrices,
    validation, and activation state.

    Attributes
    ----------
    coupled_nodes : dict
        Maps {fem_node_id: block_idx}
    constraint_matrices : dict
        Maps {(fem_node_id, block_idx): C_matrix (2x3)}
    active : bool
        Whether coupling is activated
    """

    def __init__(self, coupling_type: str):
        """
        Parameters
        ----------
        coupling_type : str
            Identifier: 'penalty', 'constraint', 'lagrange', or 'mortar'
        """
        self.coupling_type = coupling_type
        self.active = False

        # Common storage: {fem_node_id: block_idx}
        self.coupled_nodes: Dict[int, int] = {}

        # Common storage: {(fem_node_id, block_idx): C_matrix (2×3)}
        self.constraint_matrices: Dict[Tuple[int, int], np.ndarray] = {}

        # Diagnostics
        self.diagnostics = {
            'num_coupled_nodes': 0,
            'activation_errors': []
        }

    def add_coupled_node(self, fem_node_id: int, block_idx: int,
                         constraint_matrix: np.ndarray):
        """
        Register a coupled FEM node.

        The constraint matrix C relates FEM displacement to block motion:
            u_fem = C @ [u_block, v_block, theta_block]

        From rigid body kinematics:
            C = [[1, 0, -(y-yc)],
                 [0, 1,  (x-xc)]]
        """
        if constraint_matrix.shape != (2, 3):
            raise ValueError(
                f"Constraint matrix must be (2,3), got {constraint_matrix.shape}")

        self.coupled_nodes[fem_node_id] = block_idx
        self.constraint_matrices[(fem_node_id, block_idx)] = constraint_matrix.copy()
        self.diagnostics['num_coupled_nodes'] = len(self.coupled_nodes)

    def get_coupled_pairs(self) -> List[Tuple[int, int, np.ndarray]]:
        """Return list of (fem_node_id, block_idx, C_matrix) tuples."""
        return [(fn, bi, self.constraint_matrices[(fn, bi)])
                for fn, bi in self.coupled_nodes.items()]

    def validate(self) -> bool:
        """Validate coupling configuration. Returns True if valid."""
        self.diagnostics['activation_errors'].clear()

        if not self.coupled_nodes:
            self.diagnostics['activation_errors'].append(
                "No coupled nodes registered")
            return False

        return True

    def activate(self):
        """Activate coupling after validation. Raises ValueError if invalid."""
        if not self.validate():
            errors = ", ".join(self.diagnostics['activation_errors'])
            raise ValueError(f"Cannot activate coupling: {errors}")
        self.active = True

    def deactivate(self):
        """Deactivate coupling."""
        self.active = False

    def get_info(self) -> Dict:
        """Return dictionary with coupling metadata."""
        return {
            'coupling_type': self.coupling_type,
            'active': self.active,
            'num_coupled_nodes': len(self.coupled_nodes),
            'system_modification': self.get_system_modification_type(),
            'requires_special_solver': self.requires_special_solver()
        }

    # Abstract methods - subclasses must implement

    @abstractmethod
    def get_system_modification_type(self) -> str:
        """
        Return system modification type:
        - 'matrix_addition': Adds terms to K (Penalty)
        - 'matrix_reduction': Reduces K via transformation (Constraint)
        - 'matrix_augmentation': Creates saddle point system (Lagrange, Mortar)
        """
        pass

    @abstractmethod
    def get_dof_count_change(self, n_base_dofs: int) -> int:
        """
        Return change in DOF count:
        - 0: Same size (Penalty)
        - <0: Reduction (Constraint)
        - >0: Augmentation (Lagrange, Mortar)
        """
        pass

    @abstractmethod
    def requires_special_solver(self) -> bool:
        """
        Return True if saddle point solver required (Lagrange, Mortar).
        Return False for standard solver (Penalty, Constraint).
        """
        pass


class NoCoupling(BaseCoupling):
    """Null coupling for baseline comparison. No system modification."""

    def __init__(self):
        super().__init__(coupling_type='none')

    def get_system_modification_type(self) -> str:
        return 'none'

    def get_dof_count_change(self, n_base_dofs: int) -> int:
        return 0

    def requires_special_solver(self) -> bool:
        return False
