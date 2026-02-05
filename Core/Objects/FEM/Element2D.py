from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Union

import numpy as np

from Core.Objects.ConstitutiveLaw.Material import PlaneStress, PlaneStrain
from Core.Objects.FEM.BaseFE import BaseFE


@dataclass
class Geometry2D:
    """
    Geometry parameters for 2D shell elements.

    Attributes:
        t: Thickness of the shell element [m]
    """
    t: float

    def __post_init__(self):
        if self.t <= 0:
            raise ValueError(f"Thickness must be positive, got {self.t}")


@dataclass
class QuadRule:
    # tensor-product rule on [-1,1]x[-1,1]
    xi: np.ndarray
    eta: np.ndarray
    w: np.ndarray


class Element2D(BaseFE):
    """
    Isoparametric 2D element shell: subclasses provide N, dN/dxi, dN/deta,
    natural coordinates of nodes, and a quadrature rule.
    """
    DOFS_PER_NODE = 2  # 2D shell elements: [ux, uy] only (no rotation)

    def __init__(self, nodes: List[Tuple[float, float]], mat: Union[PlaneStrain, PlaneStress], geom: Geometry2D):
        """
        Initialize 2D finite element.
        """

        self.t = float(geom.t)
        self.mat = mat
        self.nd = len(nodes)
        self.dpn = 2  # DOF per node (u, v only)
        self.edof = self.nd * self.dpn
        self.nodes = [tuple(n) for n in nodes]

        # Initialize connectivity
        self.connect = np.zeros(self.nd, dtype=int)
        self.dofs = np.zeros(self.edof, dtype=int)

        # 2D shell elements have no rotation DOFs (only ux, uy)
        self.rotation_dofs = np.array([], dtype=int)

        self.lin_geom = True

    # ----- API each subclass must provide -----
    @abstractmethod
    def N_dN(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (N, dN_dxi, dN_deta) at (xi,eta)
        N: (nd,), dN_dxi: (nd,), dN_deta: (nd,)
        """
        pass

    @abstractmethod
    def quad_rule(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (XI, ETA, W) of quadrature in natural space."""
        pass

    @staticmethod
    def gauss_1x1() -> QuadRule:
        return QuadRule(np.array([0.0]), np.array([0.0]), np.array([2.0]))  # 1D weights=2

    @staticmethod
    def gauss_2x2() -> QuadRule:
        a = 1 / np.sqrt(3)
        pts = np.array([-a, a])
        w = np.array([1.0, 1.0])
        XI, ETA = np.meshgrid(pts, pts, indexing="xy")
        W = np.outer(w, w)
        return QuadRule(XI.ravel(), ETA.ravel(), W.ravel())

    @staticmethod
    def gauss_3x3() -> QuadRule:
        a = np.sqrt(3 / 5)
        pts = np.array([-a, 0.0, a])
        w1 = 5 / 9
        w2 = 8 / 9
        w = np.array([w1, w2, w1])
        XI, ETA = np.meshgrid(pts, pts, indexing="xy")
        W = np.outer(w, w)
        return QuadRule(XI.ravel(), ETA.ravel(), W.ravel())

    # ----- common machinery -----
    def _xy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.array([n[0] for n in self.nodes])
        y = np.array([n[1] for n in self.nodes])
        return x, y

    def jacobian(self, dN_dxi: np.ndarray, dN_deta: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Build Jacobian, detJ, and inverse from natural derivatives.
        """
        x, y = self._xy_arrays()
        J = np.array([[np.dot(dN_dxi, x), np.dot(dN_dxi, y)],
                      [np.dot(dN_deta, x), np.dot(dN_deta, y)]], dtype=float)
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError(f"Non-positive Jacobian determinant: {detJ}")
        Jinv = np.linalg.inv(J)
        return J, detJ, Jinv

    def B_matrix(self, dN_dx: np.ndarray, dN_dy: np.ndarray) -> np.ndarray:
        """
        Construct 3x(2*nd) B-matrix:
        [ dN1/dx  0  dN2/dx  0  pass ]
        [ 0  dN1/dy  0  dN2/dy pass ]
        [ dN1/dy dN1/dx dN2/dy dN2/dx pass ]
        """
        nd = self.nd
        B = np.zeros((3, 2 * nd))
        B[0, 0::2] = dN_dx
        B[1, 1::2] = dN_dy
        B[2, 0::2] = dN_dy
        B[2, 1::2] = dN_dx
        return B

    def Ke(self) -> np.ndarray:
        """
        Element stiffness: Ke = ∫ B^T D B t |J| de dn

        Uses material tangent stiffness (get_k_tan_2D()) instead of elastic
        stiffness (D) to support nonlinear materials. For linear elastic
        materials (PlaneStress), get_k_tan_2D() returns D, so behavior is
        unchanged. For nonlinear materials (SimplePlaneStressBilinear), this
        returns the plastic tangent stiffness when yielded.
        """
        D = self.mat.get_k_tan_2D()
        XI, ETA, W = self.quad_rule()
        ndofs = 2 * self.nd
        Ke = np.zeros((ndofs, ndofs))
        for xi, eta, w in zip(XI, ETA, W):
            N, dN_dxi, dN_deta = self.N_dN(xi, eta)
            J, detJ, Jinv = self.jacobian(dN_dxi, dN_deta)
            # chain rule: [dN/dx; dN/dy] = Jinv @ [dN/dxi; dN/deta]
            grads_nat = np.vstack((dN_dxi, dN_deta))  # 2 x nd
            grads_xy = Jinv @ grads_nat  # 2 x nd
            dN_dx, dN_dy = grads_xy[0], grads_xy[1]
            B = self.B_matrix(dN_dx, dN_dy)
            Ke += self.t * (B.T @ D @ B) * detJ * w
        return Ke

    def Me_consistent(self) -> np.ndarray:
        """
        Consistent mass: Me = ∫ ρ t (N^T N) dA  (lumped is easy too)
        """
        rho = self.mat.rho
        XI, ETA, W = self.quad_rule()
        ndofs = 2 * self.nd
        Me = np.zeros((ndofs, ndofs))
        for xi, eta, w in zip(XI, ETA, W):
            N, dN_dxi, dN_deta = self.N_dN(xi, eta)
            J, detJ, _ = self.jacobian(dN_dxi, dN_deta)
            # build 2D Nbar for u,v
            Nbar = np.zeros((2, 2 * self.nd))
            Nbar[0, 0::2] = N
            Nbar[1, 1::2] = N
            Me += rho * self.t * (Nbar.T @ Nbar) * detJ * w
        return Me

    def make_connect(self, connect: int, node_number: int, structure=None) -> None:
        """
        Map local element node to global structure node and DOFs.

        This method now supports variable DOFs per node:
        - If structure provided: uses structure.node_dof_offsets (flexible DOF system)
        - If structure=None: falls back to 3*connect (backward compatibility)

        Args:
            connect: Global node index in Structure_2D.list_nodes
            node_number: Local node index in this element (0 to nd-1)
            structure: Structure_2D instance (optional, for flexible DOF support)
        """
        # Store global node index
        self.connect[node_number] = connect

        # Compute base DOF index for this node
        if structure is not None and hasattr(structure, 'node_dof_offsets') and len(
                structure.node_dof_offsets) > connect:
            # Variable DOF mode: use node_dof_offsets
            base_dof = structure.node_dof_offsets[connect]
        else:
            # Fallback: assume 3 DOFs per node
            base_dof = 3 * connect

        # Map element DOFs (2 per node: u, v) to global structure DOFs
        self.dofs[2 * node_number] = base_dof  # u component
        self.dofs[2 * node_number + 1] = base_dof + 1  # v component

        # Note: No longer tracking rotation_dofs - variable DOF system handles this automatically

    def get_mass(self, no_inertia=False):
        # FIX: Add no_inertia parameter to match Timoshenko API
        # For 2D shell elements, no_inertia has no effect (no rotational inertia anyway)
        return self.Me_consistent()

    def get_k_glob(self):
        return self.Ke()

    def get_k_glob0(self):
        # For linear elements, K0 = K (no geometric nonlinearity)
        return self.Ke()

    def get_k_glob_LG(self):
        # For linear elements, geometric stiffness not implemented
        return np.zeros((self.edof, self.edof))

    def compute_nodal_stress_strain(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stress and strain at element nodes using extrapolation from Gauss points.

        This method evaluates stress and strain at each Gauss integration point
        and extrapolates them to element nodes using shape function fitting. This
        provides more accurate stress visualization at element corners compared to
        simple averaging.

        For nodal averaging across elements, use Structure_FEM's compute_nodal_von_mises()
        or compute_nodal_strain() methods.

        Args:
            u: Element displacement vector [u1, v1, u2, v2, ...]

        Returns:
            Tuple of:
                nodal_stress: np.ndarray of shape (nd, 3)
                              Stress [sigma_xx, sigma_yy, tau_xy] at each node
                nodal_strain: np.ndarray of shape (nd, 3)
                              Strain [eps_xx, eps_yy, gamma_xy] at each node
        """
        # Get Gauss points for this element
        XI, ETA, W = self.quad_rule()
        n_gauss = len(W)

        # Compute stress and strain at each Gauss point
        gauss_stress = np.zeros((n_gauss, 3))
        gauss_strain = np.zeros((n_gauss, 3))
        for i, (xi, eta) in enumerate(zip(XI, ETA)):
            stress, strain = self.compute_stress(u, xi, eta)
            gauss_stress[i] = stress
            gauss_strain[i] = strain

        # Try extrapolation first, fall back to averaging if it fails
        try:
            nodal_stress = self._extrapolate_to_nodes(gauss_stress, XI, ETA)
            nodal_strain = self._extrapolate_to_nodes(gauss_strain, XI, ETA)
        except (np.linalg.LinAlgError, ValueError):
            # Fall back to averaging if extrapolation fails
            avg_stress = np.mean(gauss_stress, axis=0)
            avg_strain = np.mean(gauss_strain, axis=0)
            nodal_stress = np.tile(avg_stress, (self.nd, 1))
            nodal_strain = np.tile(avg_strain, (self.nd, 1))

        return nodal_stress, nodal_strain

    def _extrapolate_to_nodes(self, gauss_vals: np.ndarray, xi_g: np.ndarray,
                               eta_g: np.ndarray) -> np.ndarray:
        """
        Extrapolate values from Gauss points to element nodes using least-squares.

        Args:
            gauss_vals: Values at Gauss points - shape (n_gauss, n_components)
            xi_g: Natural xi coordinates of Gauss points
            eta_g: Natural eta coordinates of Gauss points

        Returns:
            Extrapolated values at element nodes - shape (nd, n_components)
        """
        n_gauss = len(xi_g)

        # Build matrix M: evaluate shape functions at Gauss points
        # M[i, j] = N_j(xi_g[i], eta_g[i])
        M = np.zeros((n_gauss, self.nd))
        for i in range(n_gauss):
            N, _, _ = self.N_dN(xi_g[i], eta_g[i])
            M[i, :] = N

        # Solve for nodal values using least squares:
        # M * nodal_vals = gauss_vals
        # nodal_vals = pinv(M) * gauss_vals
        nodal_vals, _, _, _ = np.linalg.lstsq(M, gauss_vals, rcond=None)

        return nodal_vals

    def compute_nodal_stress(self, u: np.ndarray) -> np.ndarray:
        """
        Compute stress at element nodes (backward-compatible wrapper).

        For new code, prefer compute_nodal_stress_strain() to get both.

        Args:
            u: Element displacement vector [u1, v1, u2, v2, ...]

        Returns:
            nodal_stress: np.ndarray of shape (nd, 3)
                          Stress [sigma_xx, sigma_yy, tau_xy] at each node
        """
        nodal_stress, _ = self.compute_nodal_stress_strain(u)
        return nodal_stress

    def get_p_glob(self, q_glob):
        """
        Compute internal force vector with proper material state update.

        For linear materials: F = K * u (unchanged behavior)
        For nonlinear materials: Computes strain, updates material state,
        then computes F = integral(B^T * sigma) dV

        This enables material nonlinearity (e.g., SimplePlaneStressBilinear)
        to work correctly with Newton-Raphson iteration.
        """
        # Check if material has update_2D method (nonlinear material)
        if hasattr(self.mat, 'update_2D'):
            return self._get_p_glob_nonlinear(q_glob)
        else:
            # Linear material: F = K * u
            K = self.get_k_glob()
            return K @ q_glob

    def _get_p_glob_nonlinear(self, q_glob):
        """
        Internal force vector for nonlinear materials.

        Computes: F = integral(B^T * sigma) dV

        where sigma is obtained from material after updating with current strain.
        """
        XI, ETA, W = self.quad_rule()
        ndofs = 2 * self.nd
        F_int = np.zeros(ndofs)

        for xi, eta, w in zip(XI, ETA, W):
            N, dN_dxi, dN_deta = self.N_dN(xi, eta)
            J, detJ, Jinv = self.jacobian(dN_dxi, dN_deta)

            # Transform derivatives to physical coordinates
            grads_nat = np.vstack((dN_dxi, dN_deta))
            grads_xy = Jinv @ grads_nat
            dN_dx, dN_dy = grads_xy[0], grads_xy[1]

            # Build B-matrix
            B = self.B_matrix(dN_dx, dN_dy)

            # Compute strain: epsilon = B * u
            strain = B @ q_glob

            # Update material state with current strain
            self.mat.update_2D(strain)

            # Get stress from material (after update)
            stress = self.mat.get_forces_2D()

            # Compute internal force contribution: F += B^T * sigma * t * detJ * w
            F_int += self.t * (B.T @ stress) * detJ * w

        return F_int
