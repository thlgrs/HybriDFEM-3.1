"""
Triangular Finite Elements for 2D Plane Stress/Strain Analysis
===============================================================

This module implements triangular finite elements:

1. **Triangle3** (CST): 3-node linear, constant strain
2. **Triangle6** (LST): 6-node quadratic, linear strain

Key Concepts for Students:
--------------------------

**Shape Functions (N)**:
    Shape functions interpolate nodal values within the element.
    At node i: N_i = 1, N_j = 0 for j != i
    Property: sum(N_i) = 1 (partition of unity)

**Natural (Area) Coordinates for Triangles**:
    Uses coordinates (xi, eta) where:
    - xi = 0 to 1 (from node 1 toward node 2)
    - eta = 0 to 1 (from node 1 toward node 3)
    - zeta = 1 - xi - eta (third coordinate)

**B Matrix (Strain-Displacement)**:
    Relates strain to nodal displacements: epsilon = B * u
    For 2D: B relates [eps_xx, eps_yy, gamma_xy] to [u1, v1, u2, v2, ...]

**Stiffness Matrix**:
    K = integral(B^T * D * B * t * dA)
    Computed using Gauss quadrature: K = sum(w_i * B^T * D * B * t * det(J))

**Jacobian**:
    Maps natural to physical coordinates: J = dx/d(xi)
    det(J) is the area ratio between physical and reference elements

Node Numbering (counter-clockwise):
    Triangle3:           Triangle6:
        3                    3
       /\\                   /\\
      /  \\                 6  5
     /    \\               /    \\
    1------2             1---4---2

References:
    - Zienkiewicz & Taylor, "The Finite Element Method", Chapter 6
    - Cook et al., "Concepts and Applications of FEA", Chapter 6
"""

from typing import Tuple, List, Union

import numpy as np

from Core.Objects.ConstitutiveLaw.Material import PlaneStress, PlaneStrain
from Core.Objects.FEM.Element2D import Element2D, Geometry2D


class Triangle3(Element2D):
    """
    3-node linear triangular element (CST - Constant Strain Triangle) for 2D elasticity.

    Properly inherits from Element2D and implements required abstract methods.
    Uses natural coordinates: (ξ, η) ∈ [0,1] with ζ = 1-ξ-η for the third coordinate.

    Node numbering (counter-clockwise):
        3
        |\\
        | \\
        |  \\
        1---2
    """

    def __init__(self, nodes: List[Tuple[float, float]], mat: Union[PlaneStress, PlaneStrain], geom: Geometry2D):
        """
        Initialize triangle element using parent Element2D constructor.

        Args:
            nodes: List of 3 node coordinates [(x1,y1), (x2,y2), (x3,y3)]
            mat: ConstitutiveLaw object (PlaneStress or PlaneStrain)
            geom: Geometry2D object containing thickness
        """
        # Call parent constructor - this sets up nodes, mat, t, nd, dpn, edof, dofs, connect, rotation_dofs
        super().__init__(nodes, mat, geom)

        # Verify we have exactly 3 nodes for a triangle
        if self.nd != 3:
            raise ValueError(f"Triangle element requires exactly 3 nodes, got {self.nd}")

        # Additional triangle-specific attributes (optional, for convenience)
        self.area = self._compute_area()

    def _compute_area(self) -> float:
        """
        Compute triangle area using cross product formula.
        Uses nodes from parent class.
        """
        x, y = self._xy_arrays()
        return 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

    def N_dN(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Shape functions and derivatives for linear triangle in natural coordinates.

        Natural coordinates: (ξ, η) ∈ [0,1] with ζ = 1-ξ-η
        N1 = ζ = 1-ξ-η  (at node 1)
        N2 = ξ          (at node 2)
        N3 = η          (at node 3)

        Args:
            xi: First natural coordinate (0 to 1)
            eta: Second natural coordinate (0 to 1)

        Returns:
            N: Shape functions at (ξ,η) - array of shape (3,)
            dN_dxi: ∂N/∂ξ - array of shape (3,)
            dN_deta: ∂N/∂η - array of shape (3,)
        """
        zeta = 1.0 - xi - eta

        # Shape functions
        N = np.array([zeta, xi, eta])

        # Derivatives with respect to natural coordinates
        # ∂N/∂ξ = [∂N1/∂ξ, ∂N2/∂ξ, ∂N3/∂ξ] = [-1, 1, 0]
        dN_dxi = np.array([-1.0, 1.0, 0.0])

        # ∂N/∂η = [∂N1/∂η, ∂N2/∂η, ∂N3/∂η] = [-1, 0, 1]
        dN_deta = np.array([-1.0, 0.0, 1.0])

        return N, dN_dxi, dN_deta

    def quad_rule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quadrature rule for triangle in natural coordinates.

        Uses 1-point Gauss rule (centroid) for linear triangle.
        For linear elements with constant strain, 1-point rule is exact.

        Returns:
            XI: Array of ξ coordinates for integration points
            ETA: Array of η coordinates for integration points
            W: Array of weights (note: area of reference triangle is 0.5)
        """
        # 1-point rule at centroid (exact for linear triangle)
        XI = np.array([1.0 / 3.0])
        ETA = np.array([1.0 / 3.0])
        W = np.array([0.5])  # Weight for reference triangle with area 0.5

        return XI, ETA, W

    def compute_stress(self, u: np.ndarray, xi: float = 1.0 / 3.0, eta: float = 1.0 / 3.0) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Compute stress and strain at a point in natural coordinates.

        For Triangle3 (CST), stress is constant throughout the element,
        so the xi/eta parameters don't affect the result.

        Args:
            u: Displacement vector [u1, v1, u2, v2, u3, v3]
            xi: First natural coordinate (default: 1/3 = centroid)
            eta: Second natural coordinate (default: 1/3 = centroid)

        Returns:
            stress: Stress vector [sigma_xx, sigma_yy, tau_xy]
            strain: Strain vector [eps_xx, eps_yy, gamma_xy]
        """
        # Use parent's B_matrix method (or legacy version - they're identical for triangles)
        N, dN_dxi, dN_deta = self.N_dN(xi, eta)
        J, detJ, Jinv = self.jacobian(dN_dxi, dN_deta)
        grads_nat = np.vstack((dN_dxi, dN_deta))
        grads_xy = Jinv @ grads_nat
        dN_dx, dN_dy = grads_xy[0], grads_xy[1]
        B = self.B_matrix(dN_dx, dN_dy)

        # Compute strain and stress
        strain = B @ u
        stress = self.mat.D @ strain

        return stress, strain

    def compute_nodal_forces(self, u: np.ndarray) -> np.ndarray:
        """
        Compute internal nodal forces from displacement vector.

        This is the constitutive chain:
        u → ε = B*u → σ = D*ε → F_int = K*u

        Args:
            u: Displacement vector [u1, v1, u2, v2, u3, v3]

        Returns:
            F_internal: Internal force vector [F1x, F1y, F2x, F2y, F3x, F3y]
        """
        K = self.get_k_glob()
        F_internal = K @ u
        return F_internal


# Backward compatibility alias
Triangle = Triangle3


class Triangle6(Element2D):
    """
    6-node quadratic triangular element (LST - Linear Strain Triangle) for 2D elasticity.

    Higher-order element with quadratic shape functions providing better accuracy
    than the 3-node Triangle3 element, especially for curved boundaries and
    stress concentrations.

    Node numbering (counter-clockwise):
        3
        |\\
        | \\
        6  5
        |   \\
        |    \\
        1--4--2

    Nodes 1, 2, 3: Corner nodes
    Nodes 4, 5, 6: Mid-side nodes (4 on edge 1-2, 5 on edge 2-3, 6 on edge 3-1)

    Natural coordinates: (ξ, η) ∈ [0,1] with ζ = 1-ξ-η
    Uses 3-point Gauss quadrature for exact integration of quadratic terms.
    """

    def __init__(self, nodes: List[Tuple[float, float]], mat: Union[PlaneStress, PlaneStrain], geom: Geometry2D):
        """
        Initialize 6-node quadratic triangle element.

        Args:
            nodes: List of 6 node coordinates in order:
                   [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]
                   where 1,2,3 are corners and 4,5,6 are mid-side nodes
            mat: ConstitutiveLaw object (PlaneStress or PlaneStrain)
            geom: Geometry2D object containing thickness
        """
        super().__init__(nodes, mat, geom)

        if self.nd != 6:
            raise ValueError(f"Triangle6 element requires exactly 6 nodes, got {self.nd}")

        self.area = self._compute_area()

    def _compute_area(self) -> float:
        """
        Compute triangle area using corner nodes only.
        For quadratic triangle, area is based on corner nodes 0, 1, 2.
        """
        x, y = self._xy_arrays()
        # Use corner nodes only (indices 0, 1, 2)
        return 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))

    def N_dN(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Quadratic shape functions and derivatives for 6-node triangle.

        Natural coordinates: (ξ, η) ∈ [0,1] with ζ = 1-ξ-η
        Uses Lagrangian quadratic shape functions.

        Shape functions:
        N1 = ζ(2ζ-1) = (1-ξ-η)(2(1-ξ-η)-1)  [corner node 1]
        N2 = ξ(2ξ-1)                          [corner node 2]
        N3 = η(2η-1)                          [corner node 3]
        N4 = 4ξζ = 4ξ(1-ξ-η)                 [mid-side node 1-2]
        N5 = 4ξη                              [mid-side node 2-3]
        N6 = 4ηζ = 4η(1-ξ-η)                 [mid-side node 3-1]

        Args:
            xi: First natural coordinate (0 to 1)
            eta: Second natural coordinate (0 to 1)

        Returns:
            N: Shape functions at (ξ,η) - array of shape (6,)
            dN_dxi: ∂N/∂ξ - array of shape (6,)
            dN_deta: ∂N/∂η - array of shape (6,)
        """
        zeta = 1.0 - xi - eta

        # Quadratic Lagrangian shape functions
        N = np.array([
            zeta * (2.0 * zeta - 1.0),  # N1
            xi * (2.0 * xi - 1.0),  # N2
            eta * (2.0 * eta - 1.0),  # N3
            4.0 * xi * zeta,  # N4
            4.0 * xi * eta,  # N5
            4.0 * eta * zeta  # N6
        ])

        # Derivatives with respect to ξ
        # Remember: ∂ζ/∂ξ = -1
        dN_dxi = np.array([
            -4.0 * zeta + 1.0,  # ∂N1/∂ξ = ∂/∂ξ[ζ(2ζ-1)]
            4.0 * xi - 1.0,  # ∂N2/∂ξ
            0.0,  # ∂N3/∂ξ
            4.0 * (zeta - xi),  # ∂N4/∂ξ = 4(ζ - ξ)
            4.0 * eta,  # ∂N5/∂ξ
            -4.0 * eta  # ∂N6/∂ξ
        ])

        # Derivatives with respect to η
        # Remember: ∂ζ/∂η = -1
        dN_deta = np.array([
            -4.0 * zeta + 1.0,  # ∂N1/∂η
            0.0,  # ∂N2/∂η
            4.0 * eta - 1.0,  # ∂N3/∂η
            -4.0 * xi,  # ∂N4/∂η
            4.0 * xi,  # ∂N5/∂η
            4.0 * (zeta - eta)  # ∂N6/∂η = 4(ζ - η)
        ])

        return N, dN_dxi, dN_deta

    def quad_rule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        3-point Gauss quadrature for triangles (exact for quadratic elements).

        This rule integrates polynomials up to degree 2 exactly, which is
        sufficient for the quadratic shape functions of Triangle6.

        Returns:
            XI: Array of ξ coordinates for integration points
            ETA: Array of η coordinates for integration points
            W: Array of weights (sum to 0.5 for reference triangle)
        """
        # 3-point symmetric rule (degree 2 exactness)
        # Integration points at mid-side locations
        XI = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        ETA = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        W = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])  # Each weight = 1/6, sum = 0.5

        return XI, ETA, W

    def compute_stress(self, u: np.ndarray, xi: float = 1.0 / 3.0, eta: float = 1.0 / 3.0) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Compute stress and strain at a point in natural coordinates.

        For quadratic elements, stress varies within the element.

        Args:
            u: Displacement vector [u1, v1, u2, v2, u3, v3, u4, v4, u5, v5, u6, v6]
            xi: First natural coordinate (default: 1/3 = centroid)
            eta: Second natural coordinate (default: 1/3 = centroid)

        Returns:
            stress: Stress vector [sigma_xx, sigma_yy, tau_xy]
            strain: Strain vector [eps_xx, eps_yy, gamma_xy]
        """
        N, dN_dxi, dN_deta = self.N_dN(xi, eta)
        J, detJ, Jinv = self.jacobian(dN_dxi, dN_deta)

        grads_nat = np.vstack((dN_dxi, dN_deta))
        grads_xy = Jinv @ grads_nat
        dN_dx, dN_dy = grads_xy[0], grads_xy[1]
        B = self.B_matrix(dN_dx, dN_dy)

        # Compute strain and stress
        strain = B @ u
        stress = self.mat.D @ strain

        return stress, strain
