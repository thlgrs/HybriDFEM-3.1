from typing import Tuple, List, Union

import numpy as np

from Core.Objects.ConstitutiveLaw.Material import PlaneStress, PlaneStrain
from Core.Objects.FEM.Element2D import Element2D, Geometry2D


class Quad4(Element2D):
    """
    4-node isoparametric quadrilateral element for 2D elasticity.

    Node numbering (counter-clockwise from bottom-left):
        3-------2
        |       |
        |   η   |
        |   ↑   |
        |   →ξ  |
        0-------1

    Natural coordinates: ξ, η ∈ [-1, 1]
    Uses bilinear shape functions with 2×2 Gauss quadrature.
    """

    def __init__(self, nodes: List[Tuple[float, float]],
                 mat: Union[PlaneStress, PlaneStrain],
                 geom: Geometry2D):
        """
        Initialize 4-node quadrilateral element.

        Parameters
        ----------
        nodes : List[Tuple[float, float]]
            List of 4 node coordinates in counter-clockwise order:
            [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
            Node 0: bottom-left (ξ=-1, η=-1)
            Node 1: bottom-right (ξ=+1, η=-1)
            Node 2: top-right (ξ=+1, η=+1)
            Node 3: top-left (ξ=-1, η=+1)
        mat : PlaneStress or PlaneStrain
            ConstitutiveLaw model
        geom : Geometry2D
            Geometry parameters (thickness)

        Raises
        ------
        ValueError
            If number of nodes is not 4
        """
        # Call parent constructor
        super().__init__(nodes, mat, geom)

        # Verify exactly 4 nodes
        if self.nd != 4:
            raise ValueError(f"Quad4 requires exactly 4 nodes, got {self.nd}")

        # Compute approximate area (for info/validation)
        self.area = self._compute_approximate_area()

    def _compute_approximate_area(self) -> float:
        """
        Compute approximate area using shoelace formula.

        For quadrilaterals, this gives exact area if the quad is planar.
        """
        x, y = self._xy_arrays()
        # Shoelace formula
        area = 0.5 * abs(
            (x[0]*y[1] - x[1]*y[0]) +
            (x[1]*y[2] - x[2]*y[1]) +
            (x[2]*y[3] - x[3]*y[2]) +
            (x[3]*y[0] - x[0]*y[3])
        )
        return area

    def N_dN(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bilinear shape functions and derivatives for 4-node quad.

        Args:
            xi: First natural coordinate, ξ ∈ [-1, 1]
            eta: Second natural coordinate, η ∈ [-1, 1]

        Returns:
            N: Shape function values at (ξ, η) - shape (4,)
            dN_dxi: ∂N/∂ξ at (ξ, η) - shape (4,)
            dN_deta: ∂N/∂η at (ξ, η) - shape (4,)
        """
        # Bilinear shape functions
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),  # N0: node 0
            (1 + xi) * (1 - eta),  # N1: node 1
            (1 + xi) * (1 + eta),  # N2: node 2
            (1 - xi) * (1 + eta),  # N3: node 3
        ])

        # Derivatives with respect to ξ
        dN_dxi = 0.25 * np.array([
            -(1 - eta),  # ∂N0/∂ξ
            +(1 - eta),  # ∂N1/∂ξ
            +(1 + eta),  # ∂N2/∂ξ
            -(1 + eta),  # ∂N3/∂ξ
        ])

        # Derivatives with respect to η
        dN_deta = 0.25 * np.array([
            -(1 - xi),   # ∂N0/∂η
            -(1 + xi),   # ∂N1/∂η
            +(1 + xi),   # ∂N2/∂η
            +(1 - xi),   # ∂N3/∂η
        ])

        return N, dN_dxi, dN_deta

    def quad_rule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2×2 Gauss quadrature rule (4 integration points).

        Returns:
            XI, ETA: Coordinates of integration points - shape (4,)
            W: Integration weights - shape (4,)
        """
        # Use parent class method (2×2 Gauss rule on square)
        rule = self.gauss_2x2()
        return rule.xi, rule.eta, rule.w

    def compute_stress(self, u: np.ndarray, xi: float = 0.0, eta: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stress and strain at a point in natural coordinates.

        Args:
            u: Element displacement vector [u1, v1, u2, v2, u3, v3, u4, v4]
            xi: First natural coordinate (default: 0.0 = centroid)
            eta: Second natural coordinate (default: 0.0 = centroid)

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

        strain = B @ u
        stress = self.mat.D @ strain
        return stress, strain

    # Optional: Override for different quadrature order
    def quad_rule_3x3(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        3×3 Gauss quadrature (9 points) for higher accuracy.

        Use this for:
        - Distorted elements
        - Nonlinear materials
        - Higher-order interpolation

        To use: Override quad_rule() to call this method instead.
        """
        rule = self.gauss_3x3()
        return rule.xi, rule.eta, rule.w


class Quad8(Element2D):
    """
    8-node serendipity quadrilateral element for 2D elasticity.

    Node numbering (counter-clockwise, corner nodes first, then mid-side):
        3---6---2
        |       |
        7       5
        |       |
        0---4---1

    Natural coordinates: ξ, η ∈ [-1, 1]
    Uses quadratic serendipity shape functions with 3×3 Gauss quadrature.
    """

    def __init__(self, nodes: List[Tuple[float, float]],
                 mat: Union[PlaneStress, PlaneStrain],
                 geom: Geometry2D):
        """
        Initialize 8-node quadrilateral element.

        Args:
            nodes: List of 8 node coordinates (4 corners + 4 mid-side, CCW)
            mat: PlaneStress or PlaneStrain material
            geom: Geometry2D with thickness
        """
        super().__init__(nodes, mat, geom)

        if self.nd != 8:
            raise ValueError(f"Quad8 requires exactly 8 nodes, got {self.nd}")

    def N_dN(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Serendipity shape functions for 8-node quad.

        Args:
            xi, eta: Natural coordinates ∈ [-1, 1]

        Returns:
            N: Shape function values - shape (8,)
            dN_dxi: ∂N/∂ξ - shape (8,)
            dN_deta: ∂N/∂η - shape (8,)
        """
        # Corner nodes (with modifications for mid-side nodes)
        N = np.zeros(8)
        dN_dxi = np.zeros(8)
        dN_deta = np.zeros(8)

        # Corner node 0: (ξ=-1, η=-1)
        N[0] = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
        dN_dxi[0] = 0.25 * (1 - eta) * (2*xi + eta)
        dN_deta[0] = 0.25 * (1 - xi) * (xi + 2*eta)

        # Corner node 1: (ξ=+1, η=-1)
        N[1] = 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1)
        dN_dxi[1] = 0.25 * (1 - eta) * (2*xi - eta)
        dN_deta[1] = 0.25 * (1 + xi) * (-xi + 2*eta)

        # Corner node 2: (ξ=+1, η=+1)
        N[2] = 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1)
        dN_dxi[2] = 0.25 * (1 + eta) * (2*xi + eta)
        dN_deta[2] = 0.25 * (1 + xi) * (xi + 2*eta)

        # Corner node 3: (ξ=-1, η=+1)
        N[3] = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
        dN_dxi[3] = 0.25 * (1 + eta) * (2*xi - eta)
        dN_deta[3] = 0.25 * (1 - xi) * (-xi + 2*eta)

        # Mid-side node 4: (ξ=0, η=-1)
        N[4] = 0.5 * (1 - xi**2) * (1 - eta)
        dN_dxi[4] = -xi * (1 - eta)
        dN_deta[4] = -0.5 * (1 - xi**2)

        # Mid-side node 5: (ξ=+1, η=0)
        N[5] = 0.5 * (1 + xi) * (1 - eta**2)
        dN_dxi[5] = 0.5 * (1 - eta**2)
        dN_deta[5] = -(1 + xi) * eta

        # Mid-side node 6: (ξ=0, η=+1)
        N[6] = 0.5 * (1 - xi**2) * (1 + eta)
        dN_dxi[6] = -xi * (1 + eta)
        dN_deta[6] = 0.5 * (1 - xi**2)

        # Mid-side node 7: (ξ=-1, η=0)
        N[7] = 0.5 * (1 - xi) * (1 - eta**2)
        dN_dxi[7] = -0.5 * (1 - eta**2)
        dN_deta[7] = -(1 - xi) * eta

        return N, dN_dxi, dN_deta

    def quad_rule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        3×3 Gauss quadrature for 8-node quad (9 integration points).

        Higher-order quadrature needed for quadratic shape functions.
        """
        rule = self.gauss_3x3()
        return rule.xi, rule.eta, rule.w

    def compute_stress(self, u: np.ndarray, xi: float = 0.0, eta: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stress and strain at a point in natural coordinates.

        Args:
            u: Element displacement vector (16 components for 8 nodes)
            xi: First natural coordinate (default: 0.0 = centroid)
            eta: Second natural coordinate (default: 0.0 = centroid)

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

        strain = B @ u
        stress = self.mat.D @ strain
        return stress, strain


class Quad9(Element2D):
    """
    9-node Lagrangian quadrilateral element for 2D elasticity.

    Node numbering (compatible with Gmsh quad9 ordering):
        3---6---2
        |       |
        7   8   5
        |       |
        0---4---1

    Natural coordinates: ξ, η ∈ [-1, 1]
    Uses biquadratic Lagrange shape functions with 3×3 Gauss quadrature.
    """

    def __init__(self, nodes: List[Tuple[float, float]],
                 mat: Union[PlaneStress, PlaneStrain],
                 geom: Geometry2D):
        """
        Initialize 9-node Lagrangian quadrilateral element.

        Args:
            nodes: List of 9 node coordinates (4 corners + 4 mid-side + center, CCW)
            mat: PlaneStress or PlaneStrain material
            geom: Geometry2D with thickness
        """
        super().__init__(nodes, mat, geom)

        if self.nd != 9:
            raise ValueError(f"Quad9 requires exactly 9 nodes, got {self.nd}")

    def N_dN(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tensor-product Lagrange shape functions for 9-node quad.

        Args:
            xi, eta: Natural coordinates ∈ [-1, 1]

        Returns:
            N: Shape function values - shape (9,)
            dN_dxi: ∂N/∂ξ - shape (9,)
            dN_deta: ∂N/∂η - shape (9,)
        """
        # 1D Lagrange polynomials and derivatives
        # L0(t) = t(t-1)/2, L1(t) = 1-t², L2(t) = t(t+1)/2
        # dL0/dt = t - 0.5, dL1/dt = -2t, dL2/dt = t + 0.5

        # Evaluate 1D Lagrange polynomials at xi
        L0_xi = xi * (xi - 1) / 2
        L1_xi = 1 - xi ** 2
        L2_xi = xi * (xi + 1) / 2

        # Evaluate 1D Lagrange polynomials at eta
        L0_eta = eta * (eta - 1) / 2
        L1_eta = 1 - eta ** 2
        L2_eta = eta * (eta + 1) / 2

        # Derivatives
        dL0_dxi = xi - 0.5
        dL1_dxi = -2 * xi
        dL2_dxi = xi + 0.5

        dL0_deta = eta - 0.5
        dL1_deta = -2 * eta
        dL2_deta = eta + 0.5

        # Shape functions (tensor product)
        # Node mapping follows Gmsh quad9 convention:
        # 0: (xi=-1, eta=-1), 1: (xi=+1, eta=-1), 2: (xi=+1, eta=+1), 3: (xi=-1, eta=+1)
        # 4: (xi=0, eta=-1),  5: (xi=+1, eta=0),  6: (xi=0, eta=+1),  7: (xi=-1, eta=0)
        # 8: (xi=0, eta=0) - center

        N = np.array([
            L0_xi * L0_eta,  # Node 0: corner (xi=-1, eta=-1)
            L2_xi * L0_eta,  # Node 1: corner (xi=+1, eta=-1)
            L2_xi * L2_eta,  # Node 2: corner (xi=+1, eta=+1)
            L0_xi * L2_eta,  # Node 3: corner (xi=-1, eta=+1)
            L1_xi * L0_eta,  # Node 4: mid-side (xi=0, eta=-1)
            L2_xi * L1_eta,  # Node 5: mid-side (xi=+1, eta=0)
            L1_xi * L2_eta,  # Node 6: mid-side (xi=0, eta=+1)
            L0_xi * L1_eta,  # Node 7: mid-side (xi=-1, eta=0)
            L1_xi * L1_eta,  # Node 8: center (xi=0, eta=0)
        ])

        # Derivatives: ∂N/∂ξ = ∂L_i/∂ξ × L_j(η)
        dN_dxi = np.array([
            dL0_dxi * L0_eta,  # Node 0
            dL2_dxi * L0_eta,  # Node 1
            dL2_dxi * L2_eta,  # Node 2
            dL0_dxi * L2_eta,  # Node 3
            dL1_dxi * L0_eta,  # Node 4
            dL2_dxi * L1_eta,  # Node 5
            dL1_dxi * L2_eta,  # Node 6
            dL0_dxi * L1_eta,  # Node 7
            dL1_dxi * L1_eta,  # Node 8
        ])

        # Derivatives: ∂N/∂η = L_i(ξ) × ∂L_j/∂η
        dN_deta = np.array([
            L0_xi * dL0_deta,  # Node 0
            L2_xi * dL0_deta,  # Node 1
            L2_xi * dL2_deta,  # Node 2
            L0_xi * dL2_deta,  # Node 3
            L1_xi * dL0_deta,  # Node 4
            L2_xi * dL1_deta,  # Node 5
            L1_xi * dL2_deta,  # Node 6
            L0_xi * dL1_deta,  # Node 7
            L1_xi * dL1_deta,  # Node 8
        ])

        return N, dN_dxi, dN_deta

    def quad_rule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        3×3 Gauss quadrature for 9-node quad (9 integration points).

        Same integration rule as Quad8 - sufficient for biquadratic polynomials.
        """
        rule = self.gauss_3x3()
        return rule.xi, rule.eta, rule.w

    def compute_stress(self, u: np.ndarray, xi: float = 0.0, eta: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stress and strain at a point in natural coordinates.

        Args:
            u: Element displacement vector (18 components for 9 nodes)
            xi: First natural coordinate (default: 0.0 = centroid)
            eta: Second natural coordinate (default: 0.0 = centroid)

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

        strain = B @ u
        stress = self.mat.D @ strain
        return stress, strain


# ============================================================================
# HELPER FUNCTIONS FOR MESH GENERATION
# ============================================================================

def create_quad_mesh_rectangular(width: float, height: float,
                                 nx: int, ny: int,
                                 mat: Union[PlaneStress, PlaneStrain],
                                 geom: Geometry2D,
                                 origin: Tuple[float, float] = (0, 0),
                                 element_type: str = 'Quad4') -> List[Union[Quad4, Quad8, Quad9]]:
    """
    Generate a structured quadrilateral mesh for a rectangular domain.

    Args:
        width: Width of rectangle
        height: Height of rectangle
        nx: Number of elements in x-direction
        ny: Number of elements in y-direction
        mat: PlaneStress or PlaneStrain material
        geom: Geometry2D (thickness)
        origin: Bottom-left corner coordinates
        element_type: 'Quad4', 'Quad8', or 'Quad9'

    Returns:
        List of quadrilateral elements
    """
    x0, y0 = origin
    dx = width / nx
    dy = height / ny

    elements = []

    if element_type == 'Quad4':
        # Generate Quad4 elements
        for j in range(ny):
            for i in range(nx):
                # Node coordinates (counter-clockwise)
                x_left = x0 + i * dx
                x_right = x0 + (i + 1) * dx
                y_bottom = y0 + j * dy
                y_top = y0 + (j + 1) * dy

                nodes = [
                    (x_left, y_bottom),   # Node 0
                    (x_right, y_bottom),  # Node 1
                    (x_right, y_top),     # Node 2
                    (x_left, y_top),      # Node 3
                ]

                elem = Quad4(nodes, mat, geom)
                elements.append(elem)

    elif element_type == 'Quad8':
        # Generate Quad8 elements
        for j in range(ny):
            for i in range(nx):
                x_left = x0 + i * dx
                x_right = x0 + (i + 1) * dx
                x_mid = x0 + (i + 0.5) * dx
                y_bottom = y0 + j * dy
                y_top = y0 + (j + 1) * dy
                y_mid = y0 + (j + 0.5) * dy

                nodes = [
                    # Corner nodes
                    (x_left, y_bottom),   # 0
                    (x_right, y_bottom),  # 1
                    (x_right, y_top),     # 2
                    (x_left, y_top),      # 3
                    # Mid-side nodes
                    (x_mid, y_bottom),    # 4
                    (x_right, y_mid),     # 5
                    (x_mid, y_top),       # 6
                    (x_left, y_mid),      # 7
                ]

                elem = Quad8(nodes, mat, geom)
                elements.append(elem)

    elif element_type == 'Quad9':
        # Generate Quad9 elements
        for j in range(ny):
            for i in range(nx):
                x_left = x0 + i * dx
                x_right = x0 + (i + 1) * dx
                x_mid = x0 + (i + 0.5) * dx
                y_bottom = y0 + j * dy
                y_top = y0 + (j + 1) * dy
                y_mid = y0 + (j + 0.5) * dy

                nodes = [
                    # Corner nodes
                    (x_left, y_bottom),  # 0
                    (x_right, y_bottom),  # 1
                    (x_right, y_top),  # 2
                    (x_left, y_top),  # 3
                    # Mid-side nodes
                    (x_mid, y_bottom),  # 4
                    (x_right, y_mid),  # 5
                    (x_mid, y_top),  # 6
                    (x_left, y_mid),  # 7
                    # Center node
                    (x_mid, y_mid),  # 8
                ]

                elem = Quad9(nodes, mat, geom)
                elements.append(elem)

    else:
        raise ValueError(f"Unknown element_type: {element_type}. Use 'Quad4', 'Quad8', or 'Quad9'.")

    return elements


def create_quad_from_corners(corner_nodes: List[Tuple[float, float]],
                             mat: Union[PlaneStress, PlaneStrain],
                             geom: Geometry2D,
                             element_type: str = 'Quad4') -> Union[Quad4, Quad8, Quad9]:
    """
    Create a single quadrilateral element from 4 corner nodes.

    For Quad8/Quad9, mid-side nodes are automatically placed at edge midpoints.

    Args:
        corner_nodes: 4 corner coordinates in counter-clockwise order
        mat: PlaneStress or PlaneStrain material
        geom: Geometry2D (thickness)
        element_type: 'Quad4', 'Quad8', or 'Quad9'

    Returns:
        Quad element of specified type
    """
    if len(corner_nodes) != 4:
        raise ValueError("Need exactly 4 corner nodes")

    if element_type == 'Quad4':
        return Quad4(corner_nodes, mat, geom)

    elif element_type == 'Quad8':
        # Generate mid-side nodes
        p0, p1, p2, p3 = [np.array(p) for p in corner_nodes]

        # Mid-side nodes at edge midpoints
        p4 = (p0 + p1) / 2  # Bottom edge
        p5 = (p1 + p2) / 2  # Right edge
        p6 = (p2 + p3) / 2  # Top edge
        p7 = (p3 + p0) / 2  # Left edge

        nodes = [
            tuple(p0), tuple(p1), tuple(p2), tuple(p3),  # Corners
            tuple(p4), tuple(p5), tuple(p6), tuple(p7),  # Mid-sides
        ]

        return Quad8(nodes, mat, geom)

    elif element_type == 'Quad9':
        # Generate mid-side nodes and center node
        p0, p1, p2, p3 = [np.array(p) for p in corner_nodes]

        # Mid-side nodes at edge midpoints
        p4 = (p0 + p1) / 2  # Bottom edge
        p5 = (p1 + p2) / 2  # Right edge
        p6 = (p2 + p3) / 2  # Top edge
        p7 = (p3 + p0) / 2  # Left edge

        # Center node at element centroid
        p8 = (p0 + p1 + p2 + p3) / 4

        nodes = [
            tuple(p0), tuple(p1), tuple(p2), tuple(p3),  # Corners
            tuple(p4), tuple(p5), tuple(p6), tuple(p7),  # Mid-sides
            tuple(p8),  # Center
        ]

        return Quad9(nodes, mat, geom)

    else:
        raise ValueError(f"Unknown element_type: {element_type}. Use 'Quad4', 'Quad8', or 'Quad9'.")
