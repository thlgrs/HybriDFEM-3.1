import warnings
from typing import Tuple, Optional

import numpy as np


class ConstraintMatrix:
    """
    Utility class for constraint matrix computations.

    This class encapsulates the rigid body kinematics relating a point's
    displacement to a rigid body's motion.
    """

    def __init__(self, small_angle: bool = True):
        """
        Initialize constraint matrix computer.

        Parameters:
            small_angle: Use small angle approximation (linear kinematics)
                        If False, uses exact rotation (nonlinear)
        """
        self.small_angle = small_angle

    def compute(
        self,
        node_position: np.ndarray,
        block_ref_point: np.ndarray,
        theta: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute constraint matrix for a node relative to a block.

        Parameters:
            node_position: [x, y] position of the FEM node
            block_ref_point: [x_c, y_c] reference point of the rigid block
            theta: Current rotation angle (rad). Only used if small_angle=False

        Returns:
            C: 2×3 constraint matrix

        Examples:
            cm = ConstraintMatrix()
            C = cm.compute([1.0, 2.0], [0.5, 1.0])
            # C = [[1, 0, -1], [0, 1, 0.5]]
        """
        x, y = node_position
        x_c, y_c = block_ref_point

        if self.small_angle or theta is None:
            # Linear kinematics (small angle approximation)
            # u = u_b - (y - y_c) * θ
            # v = v_b + (x - x_c) * θ
            C = np.array([
                [1.0, 0.0, -(y - y_c)],
                [0.0, 1.0, (x - x_c)]
            ], dtype=float)
        else:
            # Exact rotation (for large angles)
            # u = u_b + (x-x_c)*cos(θ) - (y-y_c)*sin(θ) - (x-x_c)
            # v = v_b + (x-x_c)*sin(θ) + (y-y_c)*cos(θ) - (y-y_c)
            #
            # Linearizing around current θ:
            # ∂u/∂θ = -(x-x_c)*sin(θ) - (y-y_c)*cos(θ)
            # ∂v/∂θ =  (x-x_c)*cos(θ) - (y-y_c)*sin(θ)

            dx = x - x_c
            dy = y - y_c
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            C = np.array([
                [1.0, 0.0, -dx * sin_theta - dy * cos_theta],
                [0.0, 1.0, dx * cos_theta - dy * sin_theta]
            ], dtype=float)

        return C

    def compute_derivative(
        self,
        node_position: np.ndarray,
        block_ref_point: np.ndarray,
        theta: float
    ) -> np.ndarray:
        """
        Compute derivative of constraint matrix with respect to θ.

        Used for geometric stiffness in large rotation analysis.

        Parameters:
            node_position: [x, y] position of node
            block_ref_point: [x_c, y_c] reference point
            theta: Current rotation angle (rad)

        Returns:
            dC_dtheta: 2×3 matrix ∂C/∂θ
        """
        if self.small_angle:
            # For small angles, C is constant (doesn't depend on θ)
            return np.zeros((2, 3), dtype=float)

        x, y = node_position
        x_c, y_c = block_ref_point
        dx = x - x_c
        dy = y - y_c
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # ∂C/∂θ for exact rotation
        dC_dtheta = np.array([
            [0.0, 0.0, -dx * cos_theta + dy * sin_theta],
            [0.0, 0.0, -dx * sin_theta - dy * cos_theta]
        ], dtype=float)

        return dC_dtheta

    def verify_constraint(
        self,
        u_fem: np.ndarray,
        u_block: np.ndarray,
        C: np.ndarray,
        tolerance: float = 1e-10
    ) -> bool:
        """
        Verify that constraint is satisfied: u_fem = C @ u_block

        Parameters:
            u_fem: FEM node displacement [u, v]
            u_block: Block displacement [u, v, θ]
            C: Constraint matrix 2×3
            tolerance: Tolerance for verification

        Returns:
            satisfied: True if constraint is satisfied within tolerance
        """
        u_expected = C @ u_block
        error = np.linalg.norm(u_fem - u_expected)

        if error > tolerance:
            warnings.warn(
                f"Constraint violation: ||u_fem - C*u_block|| = {error:.2e} > {tolerance:.2e}"
            )
            return False

        return True


def compute_rigid_body_constraint(
    node_position: np.ndarray,
    block_ref_point: np.ndarray,
    small_angle: bool = True
) -> np.ndarray:
    """
    Convenience function to compute constraint matrix.

    This is the recommended interface for most use cases.

    Parameters:
        node_position: [x, y] coordinates of FEM node
        block_ref_point: [x_c, y_c] reference point of rigid block
        small_angle: Use linear kinematics (True) or exact rotation (False)

    Returns:
        C: 2×3 constraint matrix

    Examples:
        C = compute_rigid_body_constraint([1.0, 2.0], [0.5, 1.0])
        print(C)
        [[ 1.   0.  -1. ]
         [ 0.   1.   0.5]]

    Notes:
        This function reuses the same formulation as Block_2D.constraint_matrix_for_node()
        to ensure consistency with the existing HybriDFEM implementation.
    """
    cm = ConstraintMatrix(small_angle=small_angle)
    return cm.compute(node_position, block_ref_point)


def compute_lever_arm(node_position: np.ndarray, block_ref_point: np.ndarray) -> float:
    """
    Compute lever arm distance from block reference point to node.

    This is used in penalty stiffness calculations.

    Parameters:
        node_position: [x, y] node coordinates
        block_ref_point: [x_c, y_c] block reference point

    Returns:
        r: Lever arm distance (meters)
    """
    return np.linalg.norm(np.array(node_position) - np.array(block_ref_point))


def compute_constraint_violation(
    u_fem: np.ndarray,
    u_block: np.ndarray,
    C: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Compute constraint violation vector and its magnitude.

    Violation vector: δ = u_fem - C @ u_block

    Parameters:
        u_fem: FEM node displacement [u, v]
        u_block: Block displacement [u, v, θ]
        C: Constraint matrix 2×3

    Returns:
        delta: Violation vector [δu, δv]
        magnitude: ||δ|| (violation magnitude)

    Notes:
        In penalty method, the violation is penalized: E_penalty = (k/2) * ||δ||²
    """
    delta = u_fem - C @ u_block
    magnitude = np.linalg.norm(delta)
    return delta, magnitude
