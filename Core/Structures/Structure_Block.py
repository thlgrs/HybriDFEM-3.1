# Standard imports
import warnings
from typing import List, Union

import numpy as np
import scipy as sc
from scipy.spatial import cKDTree

from Core.Objects.DFEM.Block import Block_2D
from Core.Objects.DFEM.ContactFace import CF_2D
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Structures.Structure_2D import Structure_2D


def _solve_modal_for_damping(structure, modes=None, initial=True):
    """Helper function to solve modal analysis for damping calculation."""
    from Core.Solvers.Modal import Modal
    return Modal.solve_modal(structure, modes=modes, save=False, initial=initial)


class Structure_Block(Structure_2D):
    """
    Structure class for discrete element (rigid block) assemblies.

    This class provides core functionality for block assemblies with contact mechanics.
    Each block is a rigid body with 3 DOFs [ux, uy, rz], and contact interfaces
    define interactions between blocks.

    Attributes
    ----------
    list_blocks : List[Block_2D]
        Rigid blocks in the structure
    list_cfs : List[CF_2D]
        Contact faces defining block interfaces

    See Also
    --------
    BeamBlock, ArchBlock, WallBlock, VoronoiBlock : Specialized constructors
    """

    def __init__(self, listBlocks: Union[List[Block_2D], None] = None):
        super().__init__(structure_type="BLOCKS")
        self.list_blocks = listBlocks or []
        self.list_cfs: List[CF_2D] = []

    # ==========================================================================
    # Construction Methods
    # ==========================================================================

    def add_block(self, block: Block_2D):
        """
        Add an existing Block_2D object to the structure.

        Parameters
        ----------
        block : Block_2D
            The pre-constructed block to add.
        """
        if not isinstance(block, Block_2D):
            raise TypeError("Argument must be an instance of Block_2D.")
        self.list_blocks.append(block)

    def add_block_from_vertices(self, vertices, b=1, material=None, ref_point=None):
        """
        Create and add a single block from its vertices.

        Parameters
        ----------
        vertices : array-like
            Block vertices as 2D array.
        b : float, optional
            Out-of-plane thickness [m].
        material : ConstitutiveLaw, optional
            ConstitutiveLaw model for the block.
        ref_point : array-like, optional
            Reference point (defaults to centroid if None).
        """
        new_block = Block_2D(vertices, b=b, material=material, ref_point=ref_point)
        self.list_blocks.append(new_block)

    def add_block_from_dimensions(self, ref_point, l, h, b=1, material=None):
        """
        Create and add a rectangular block from its dimensions.

        Parameters
        ----------
        ref_point : tuple or array-like
            (xc, yc) coordinates of the block's center.
        l : float
            Length of the block (along x-axis).
        h : float
            Height of the block (along y-axis).
        b : float, optional
            Out-of-plane thickness [m].
        material : ConstitutiveLaw, optional
            ConstitutiveLaw model for the block.
        """
        xc, yc = ref_point
        dx, dy = l / 2, h / 2
        vertices = np.array([
            (xc - dx, yc - dy),
            (xc + dx, yc - dy),
            (xc + dx, yc + dy),
            (xc - dx, yc + dy)
        ])
        new_block = Block_2D(vertices, b=b, material=material, ref_point=ref_point)
        self.list_blocks.append(new_block)

    # ==========================================================================
    # Node Generation Methods
    # ==========================================================================

    def _make_nodes_block(self):
        """Generate nodes from block reference points."""
        for block in self.list_blocks:
            dof_count = getattr(block, 'DOFS_PER_NODE', 3)
            index = self._add_node_if_new(block.ref_point, dof_count=dof_count)
            block.make_connect(index, structure=self)

    def make_nodes(self):
        """Generate nodes from blocks and initialize DOFs."""
        self._make_nodes_block()
        self.nb_dofs = self.compute_nb_dofs()
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = len(self.dof_free)

    def detect_interfaces(self, eps=1e-9, margin=0.01):
        """
        Detect collinear contact interfaces between blocks.

        This method uses geometric algorithms to find overlapping edges between blocks.

        Parameters
        ----------
        eps : float, default=1e-9
            Tolerance for collinearity detection
        margin : float, default=0.01
            Margin for bounding circle pre-filtering

        Returns
        -------
        list of dict
            Detected interfaces with block pairs and edge endpoints
        """
        def overlap_colinear(seg1, seg2, eps=1e-9):
            """Check overlap between two colinear segments."""
            p1, p2 = np.asarray(seg1, float)
            q1, q2 = np.asarray(seg2, float)

            v = p2 - p1
            Lv = np.linalg.norm(v)
            if Lv <= eps:
                return False, None  # degenerate segment
            u = v / Lv  # unit direction

            # project all endpoints onto u, using p1 as origin
            tp = np.array([0.0, Lv])  # p1->p1, p1->p2
            tq = np.array([np.dot(q1 - p1, u), np.dot(q2 - p1, u)])
            tq.sort()

            t_start = max(tp.min(), tq.min())
            t_end = min(tp.max(), tq.max())

            if t_end - t_start < -eps:
                return False, None

            a = p1 + t_start * u
            b = p1 + t_end * u
            return True, np.vstack([a, b])

        def are_colinear(p1, p2, q1, q2, eps=1e-9):
            """Check if four points are colinear using cross products."""
            v = p2 - p1
            w1 = q1 - p1
            w2 = q2 - p1

            def cross(a, b): return a[0] * b[1] - a[1] * b[0]

            return (abs(cross(v, w1)) <= eps) and (abs(cross(v, w2)) <= eps)

        def circles_separated_sq(c1, r1, c2, r2, margin=0.01):
            """Check if two circles are separated (no contact)."""
            d2 = np.sum((c1 - c2) ** 2)
            thr = (r1 + r2) ** 2 * (1.0 + margin) ** 2
            return d2 >= thr

        # prefetch
        blocks = self.list_blocks
        B = len(blocks)
        triplets = [blk.compute_triplets() for blk in blocks]

        interfaces = []
        self.interf_counter = 0

        # --- Optimization: Broad-phase using KD-Tree ---
        if B > 0:
            centers = np.array([b.circle_center for b in blocks])
            radii = np.array([b.circle_radius for b in blocks])
            max_r = np.max(radii)

            tree = cKDTree(centers)

            # Query pairs within max possible interaction distance
            # We use a safe upper bound: dist < r_i + r_max + margin
            # This is slightly conservative but safe and much faster than N^2
            # Using query_pairs returns a set of (i, j) tuples where i < j

            # To be strictly correct with varying radii, we need a safe constant radius query 
            # or individual queries. query_pairs uses a fixed radius.
            # Let's use query_pairs with 2*max_r + margin as a conservative bound.
            search_radius = 2.0 * max_r * (1.0 + margin)
            candidate_pairs = tree.query_pairs(r=search_radius)
        else:
            candidate_pairs = []

        for i, j in candidate_pairs:
            cand = blocks[i]
            anta = blocks[j]

            # 1) quick prunes (re-check specific radii condition strictly)
            if cand.connect == anta.connect:
                continue

            # Strict check: d^2 > (r1+r2)^2
            if circles_separated_sq(cand.circle_center, cand.circle_radius,
                                    anta.circle_center, anta.circle_radius,
                                    margin=margin):
                continue

            # 2) test edges on the same line
            ifaces_ij = []
            for t1 in triplets[i]:
                A1, B1, C1 = t1["ABC"]
                P = np.asarray(t1["Vertices"], float)  # shape (2,2)
                for t2 in triplets[j]:
                    if not np.allclose(t1["ABC"], t2["ABC"], rtol=1e-8, atol=eps):
                        continue
                    Q = np.asarray(t2["Vertices"], float)

                    # now both segments lie on the same infinite line; check finite overlap
                    has, seg = overlap_colinear(P, Q, eps=eps)
                    if not has:
                        continue

                    a, b = seg  # endpoints
                    u = (b - a)
                    Lu = np.linalg.norm(u)
                    if Lu <= eps:  # zero-length overlap
                        continue
                    u /= Lu
                    n = np.array([-u[1], u[0]])  # left-hand normal
                    # Decide block A vs B via normal direction
                    if np.dot(cand.ref_point - a, n) > 0:
                        blA, blB = cand, anta
                    else:
                        blA, blB = anta, cand
                    ifaces_ij.append({
                        "Block A": blA,
                        "Block B": blB,
                        "x_e1": a,
                        "x_e2": b,
                        # (optionally keep unit vectors if useful)
                        # "tangent": u, "normal": n
                    })

            self.interf_counter += 1
            if ifaces_ij:
                interfaces.extend(ifaces_ij)

        return interfaces

    def make_cfs(self, lin_geom, nb_cps=2, offset=-1, contact=None, surface=None, weights=None, interfaces=None):
        """
        Create contact faces from detected or provided interfaces.

        Parameters
        ----------
        lin_geom : bool
            Linear geometry flag
        nb_cps : int, default=2
            Number of contact points per interface
        offset : int, default=-1
            Contact offset parameter
        contact : optional
            Contact model to use
        surface : optional
            Surface model to use
        weights : optional
            Weight distribution for contact points
        interfaces : list, optional
            Pre-detected interfaces (if None, will detect automatically)
        """
        if interfaces is None:
            interfaces = self.detect_interfaces()
        for i, face in enumerate(interfaces):
            cf = CF_2D(face, nb_cps, lin_geom, offset=offset, contact=contact, surface=surface, weights=weights)
            self.list_cfs.append(cf)
            cf.bl_A.cfs.append(i)
            cf.bl_B.cfs.append(i)

    # ==========================================================================
    # Matrix Assembly Methods
    # ==========================================================================

    def _get_P_r_block(self):
        """Assemble internal force vector from contact forces."""
        self.dofs_defined()
        if not hasattr(self, "P_r"):
            self.P_r = np.zeros(self.nb_dofs, dtype=float)

        for CF in self.list_cfs:
            qf_glob = np.zeros(6)
            qf_glob[:3] = self.U[CF.bl_A.dofs]
            qf_glob[3:] = self.U[CF.bl_B.dofs]
            pf_glob = CF.get_pf_glob(qf_glob)
            self.P_r[CF.bl_A.dofs] += pf_glob[:3]
            self.P_r[CF.bl_B.dofs] += pf_glob[3:]

    def get_P_r(self):
        """Assemble global internal force vector."""
        self.P_r = np.zeros(self.nb_dofs, dtype=float)
        self._get_P_r_block()

    def _mass_block(self, no_inertia: bool = False):
        """Assemble mass matrix contributions from blocks."""
        for block in getattr(self, "list_blocks", []):
            # block mass matrix must align with block.dofs length
            M_block = block.get_mass(no_inertia=no_inertia)
            dofs = np.asarray(block.dofs, dtype=int)
            self.M[np.ix_(dofs, dofs)] += M_block

    def get_M_str(self, no_inertia: bool = False):
        """Assemble global mass matrix."""
        self.dofs_defined()
        self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._mass_block(no_inertia=no_inertia)
        return self.M

    def _stiffness_block(self):
        """Assemble tangent stiffness matrix from contact stiffness."""
        for CF in getattr(self, "list_cfs", []):
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob = CF.get_kf_glob()

            self.K[np.ix_(dof1, dof1)] += kf_glob[:3, :3]
            self.K[np.ix_(dof1, dof2)] += kf_glob[:3, 3:]
            self.K[np.ix_(dof2, dof1)] += kf_glob[3:, :3]
            self.K[np.ix_(dof2, dof2)] += kf_glob[3:, 3:]

    def _stiffness0_block(self):
        """Assemble initial stiffness matrix from contact stiffness."""
        for CF in getattr(self, "list_cfs", []):
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob0 = CF.get_kf_glob0()

            self.K0[np.ix_(dof1, dof1)] += kf_glob0[:3, :3]
            self.K0[np.ix_(dof1, dof2)] += kf_glob0[:3, 3:]
            self.K0[np.ix_(dof2, dof1)] += kf_glob0[3:, :3]
            self.K0[np.ix_(dof2, dof2)] += kf_glob0[3:, 3:]

    def _stiffness_LG_block(self):
        """Assemble geometric stiffness matrix from contact stiffness."""
        for CF in getattr(self, "list_cfs", []):
            dof1 = CF.bl_A.dofs
            dof2 = CF.bl_B.dofs

            kf_glob_LG = CF.get_kf_glob_LG()

            self.K_LG[np.ix_(dof1, dof1)] += kf_glob_LG[:3, :3]
            self.K_LG[np.ix_(dof1, dof2)] += kf_glob_LG[:3, 3:]
            self.K_LG[np.ix_(dof2, dof1)] += kf_glob_LG[3:, :3]
            self.K_LG[np.ix_(dof2, dof2)] += kf_glob_LG[3:, 3:]

    def get_K_str(self):
        """Assemble global tangent stiffness matrix."""
        self.dofs_defined()
        self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_block()
        return self.K

    def get_K_str0(self):
        """Assemble global initial stiffness matrix."""
        self.dofs_defined()
        self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness0_block()
        return self.K0

    def get_K_str_LG(self):
        """Assemble global geometric stiffness matrix."""
        self.dofs_defined()
        self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_LG_block()
        return self.K_LG

    def set_lin_geom(self, lin_geom=True):
        """Set linear or nonlinear geometry for all contact faces."""
        for cf in self.list_cfs:
            cf.set_lin_geom(lin_geom)

    def get_C_str(self):
        if not (hasattr(self, "K")):
            self.get_K_str()
        # if not (hasattr(self, 'M')): self.get_M_str()

        if not hasattr(self, "damp_coeff"):
            # No damping
            if self.xsi[0] == 0 and self.xsi[1] == 0:
                self.damp_coeff = np.zeros(2)

            elif self.damp_type == "RAYLEIGH":
                try:
                    _solve_modal_for_damping(self, modes=2, initial=True)
                except Exception:
                    _solve_modal_for_damping(self, modes=None, initial=True)

                A = np.array(
                    [
                        [1 / self.eig_vals[0], self.eig_vals[0]],
                        [1 / self.eig_vals[1], self.eig_vals[1]],
                    ]
                )

                if isinstance(self.xsi, float):
                    self.xsi = [self.xsi, self.xsi]
                    self.damp_coeff = 2 * sc.linalg.solve(A, np.array(self.xsi))

                if isinstance(self.xsi, list) and len(self.xsi) == 2:
                    self.damp_coeff = 2 * sc.linalg.solve(A, np.array(self.xsi))

                else:
                    warnings.warn(
                        "Xsi is not a list of two damping ratios for Rayleigh damping"
                    )

            elif self.damp_type == "STIFF":
                if not hasattr(self, "eig_vals"):
                    try:
                        _solve_modal_for_damping(self, modes=1, initial=True)
                    except Exception:
                        _solve_modal_for_damping(self, modes=None, initial=True)
                self.damp_coeff = np.array([0, 2 * self.xsi[0] / self.eig_vals[0]])

            elif self.damp_type == "MASS":
                try:
                    _solve_modal_for_damping(self, modes=1, initial=True)
                except Exception:
                    _solve_modal_for_damping(self, modes=None, initial=True)
                self.damp_coeff = np.array([2 * self.xsi[0] * self.eig_vals[0], 0])

        if self.stiff_type == "INIT":
            if not (hasattr(self, "C")):
                self.get_K_str0()
                self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K0

        elif self.stiff_type == "TAN":
            self.get_K_str()
            self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K

        elif self.stiff_type == "TAN_LG":
            self.get_K_str_LG()
            self.C = self.damp_coeff[0] * self.M + self.damp_coeff[1] * self.K_LG

    def commit(self):
        """Commit converged state for all contact faces."""
        for CF in self.list_cfs:
            CF.commit()

    def revert_commit(self):
        """Revert to last committed state for all contact faces."""
        for CF in self.list_cfs:
            CF.revert_commit()


class BeamBlock(Structure_Block):
    """
    Beam-like assembly of rectangular blocks.

    Automatically generates a sequence of rectangular blocks along a line,
    suitable for modeling beam-like structures or block-masonry beams.

    Parameters
    ----------
    N1 : array-like
        Start point (x, y) of the beam
    N2 : array-like
        End point (x, y) of the beam
    n_blocks : int
        Number of blocks in the beam
    h : float
        Height of each block (perpendicular to beam axis)
    rho : float
        Density for mass calculation
    b : float, optional
        Out-of-plane thickness (default: 1)
    material : ConstitutiveLaw, optional
        ConstitutiveLaw model for blocks
    end_1 : bool
        Include end reference point at start (default: True)
    end_2 : bool
        Include end reference point at end (default: True)
    """

    def __init__(
            self, N1, N2, n_blocks, h, rho, b=1, material=None, end_1=True, end_2=True
    ):
        blockList = []
        lx = N2[0] - N1[0]
        ly = N2[1] - N1[1]
        L = np.sqrt(lx ** 2 + ly ** 2)
        L_b = L / (n_blocks - 1)

        long = np.array([lx, ly]) / L
        tran = np.array([-ly, lx]) / L

        # Loop to create the blocks
        ref_point = N1.copy()

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([ref_point, ref_point, ref_point, ref_point])

            if i == 0:  # First block is half block
                vertices[0] += L_b / 2 * long - h / 2 * tran
                vertices[1] += L_b / 2 * long + h / 2 * tran
                vertices[2] += h / 2 * tran
                vertices[3] += -h / 2 * tran

                if end_1:
                    ref = ref_point
                else:
                    ref = None

            elif i == n_blocks - 1:  # Last block is also a half_block
                vertices[0] += -h / 2 * tran
                vertices[1] += h / 2 * tran
                vertices[2] += h / 2 * tran - L_b / 2 * long
                vertices[3] += -h / 2 * tran - L_b / 2 * long

                if end_2:
                    ref = ref_point
                else:
                    ref = None

            else:
                vertices[0] += -h / 2 * tran + L_b / 2 * long
                vertices[1] += h / 2 * tran + L_b / 2 * long
                vertices[2] += h / 2 * tran - L_b / 2 * long
                vertices[3] += -h / 2 * tran - L_b / 2 * long
                ref = None

            # Create material with rho if not provided
            if material is None:
                block_material = Material(E=1e9, nu=0.3, rho=rho)
            else:
                block_material = material
            blockList.append(Block_2D(vertices, b=b, material=block_material, ref_point=ref))
            ref_point += L_b * long

        super().__init__(blockList)


class TaperedBeamBlock(Structure_Block):
    def __init__(self, N1, N2, n_blocks, h1, h2, rho, b=1, material=None, contact=None, end_1=True, end_2=True):
        """
        Add a tapered beam (varying height) made of blocks.

        Parameters
        ----------
        N1, N2 : array-like
            Start and end points of beam centerline
        n_blocks : int
            Number of blocks
        h1, h2 : float
            Heights at start and end [m]
        rho : float
            Density [kg/m³]
        b : float, optional
            Out-of-plane thickness [m]
        material : ConstitutiveLaw, optional
            ConstitutiveLaw model
        contact : optional
            Contact parameters (deprecated)
        end_1, end_2 : bool, optional
            Whether to include end reference points
        """
        blockList = []
        lx = N2[0] - N1[0]
        ly = N2[1] - N1[1]
        L = np.sqrt(lx ** 2 + ly ** 2)
        L_b = L / (n_blocks - 1)

        heights = np.linspace(h1, h2, n_blocks)
        d_h = (heights[1] - heights[0]) / 2

        long = np.array([lx, ly]) / L
        tran = np.array([-ly, lx]) / L

        # Loop to create the blocks
        ref_point = N1.copy()

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([ref_point, ref_point, ref_point, ref_point])

            if i == 0:  # First block is half block
                h1 = heights[i]
                h2 = heights[i] + d_h
                vertices[0] += L_b / 2 * long - h2 / 2 * tran
                vertices[1] += L_b / 2 * long + h2 / 2 * tran
                vertices[2] += h1 / 2 * tran
                vertices[3] += -h1 / 2 * tran

                if end_1:
                    ref = ref_point
                else:
                    ref = None

            elif i == n_blocks - 1:  # Last block is also a half_block
                h2 = heights[i]
                h1 = heights[i] - d_h
                vertices[0] += -h2 / 2 * tran
                vertices[1] += h2 / 2 * tran
                vertices[2] += h1 / 2 * tran - L_b / 2 * long
                vertices[3] += -h1 / 2 * tran - L_b / 2 * long

                if end_2:
                    ref = ref_point
                else:
                    ref = None

            else:
                h1 = heights[i] - d_h
                h2 = heights[i] + d_h
                vertices[0] += -h2 / 2 * tran + L_b / 2 * long
                vertices[1] += h2 / 2 * tran + L_b / 2 * long
                vertices[2] += h1 / 2 * tran - L_b / 2 * long
                vertices[3] += -h1 / 2 * tran - L_b / 2 * long
                ref = None

            # Create material with rho if not provided
            if material is None:
                block_material = Material(E=1e9, nu=0.3, rho=rho)
            else:
                block_material = material
            blockList.append(Block_2D(vertices, b=b, material=block_material, ref_point=ref))

            ref_point += L_b * long
        super().__init__(blockList)


class ArchBlock(Structure_Block):
    """Structure_Block specialization for arch assemblies."""

    def __init__(self, c, a1, a2, R, n_blocks, h, rho, b=1, material=None, contact=None):
        """
        Add an arch made of blocks.

        Parameters
        ----------
        c : array-like
            Center point of the arch
        a1, a2 : float
            Start and end angles [radians]
        R : float
            Mean radius of arch [m]
        n_blocks : int
            Number of blocks
        h : float
            Radial thickness of arch [m]
        rho : float
            Density [kg/m³]
        b : float, optional
            Out-of-plane thickness [m]
        material : ConstitutiveLaw, optional
            ConstitutiveLaw model
        contact : optional
            Contact parameters (deprecated)
        """
        blockList = []
        d_a = (a2 - a1) / n_blocks
        angle = a1

        R_int = R - h / 2
        R_out = R + h / 2

        for i in np.arange(n_blocks):
            # Initialize array of vertices
            vertices = np.array([c, c, c, c])

            unit_dir_1 = np.array([np.cos(angle), np.sin(angle)])
            unit_dir_2 = np.array([np.cos(angle + d_a), np.sin(angle + d_a)])
            vertices[0] += R_int * unit_dir_1
            vertices[1] += R_out * unit_dir_1
            vertices[2] += R_out * unit_dir_2
            vertices[3] += R_int * unit_dir_2

            # print(vertices)
            # Create material with rho if not provided
            if material is None:
                block_material = Material(E=1e9, nu=0.3, rho=rho)
            else:
                block_material = material
            blockList.append(Block_2D(vertices, b=b, material=block_material))

            angle += d_a
        super().__init__(blockList)


class WallBlock(Structure_Block):
    """Structure_Block specialization for masonry wall assemblies."""

    def __init__(self, c1, l_block, h_block, pattern, rho, b=1, material=None, orientation=None):
        """
        Add a masonry wall with specified pattern.

        Parameters
        ----------
        c1 : array-like
            Origin point of the wall
        l_block : float
            Standard block length [m]
        h_block : float
            Block height [m]
        pattern : list of lists
            Wall pattern where each row is a list of block length multipliers
            Positive values = full block, negative = gap
        rho : float
            Density [kg/m³]
        b : float, optional
            Out-of-plane thickness [m]
        material : ConstitutiveLaw, optional
            ConstitutiveLaw model
        orientation : array-like, optional
            Orientation vector (default is [1, 0])
        """
        blockList = []
        if orientation is not None:
            long = orientation
            tran = np.array([-orientation[1], orientation[0]])
        else:
            long = np.array([1, 0], dtype=float)
            tran = np.array([0, 1], dtype=float)

        for j, line in enumerate(pattern):
            ref_point = (
                    c1 + 0.5 * abs(line[0]) * l_block * long + (j + 0.5) * h_block * tran
            )

            for i, brick in enumerate(line):
                if brick > 0:
                    vertices = np.array([ref_point, ref_point, ref_point, ref_point])
                    vertices[0] += brick * l_block / 2 * long - h_block / 2 * tran
                    vertices[1] += brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[2] += -brick * l_block / 2 * long + h_block / 2 * tran
                    vertices[3] += -brick * l_block / 2 * long - h_block / 2 * tran

                    # Create material with rho if not provided
                    if material is None:
                        block_material = Material(E=1e9, nu=0.3, rho=rho)
                    else:
                        block_material = material
                    blockList.append(Block_2D(vertices, b=b, material=block_material))

                if not i == len(line) - 1:
                    ref_point += 0.5 * l_block * long * (abs(brick) + abs(line[i + 1]))
        super().__init__(blockList)


class VoronoiBlock(Structure_Block):
    """Structure_Block specialization for Voronoi tessellation assemblies."""

    def __init__(self, surface, list_of_points, rho, b=1, material=None):
        """
        Add blocks using Voronoi tessellation within a surface.

        Parameters
        ----------
        surface : array-like
            List of points defining the boundary surface
        list_of_points : array-like
            Points to use as Voronoi cell centers
        rho : float
            Density [kg/m³]
        b : float, optional
            Out-of-plane thickness [m]
        material : ConstitutiveLaw, optional
            ConstitutiveLaw model
        """
        # Surface is a list of points defining the surface to be subdivided into
        # Voronoi cells.

        blockList = []

        def point_in_surface(point, surface):
            # Check if a point lies on the surface
            # Surface is a list of points delimiting the surface
            # Point is a 2D numpy array

            n = len(surface)

            for i in range(n):
                A = surface[i]
                B = surface[(i + 1) % n]
                C = point

                if np.cross(B - A, C - A) < 0:
                    return False

            return True

        for point in list_of_points:
            # Check if all points lie on the surface
            if not point_in_surface(point, surface):
                warnings.warn("Not all points lie on the surface")
                return

        # Create Voronoi cells
        vor = sc.spatial.Voronoi(list_of_points)

        # Create block for each Voronoi region
        # If region is finite, it's easy
        # If region is infinite, delimit it with the edge of the surface
        # Create material with rho if not provided
        if material is None:
            block_material = Material(E=1e9, nu=0.3, rho=rho)
        else:
            block_material = material

        for region in vor.regions[1:]:
            if not -1 in region:
                vertices = np.array([vor.vertices[i] for i in region])
                blockList.append(Block_2D(vertices, b=b, material=block_material))

            else:
                vertices = []
                for i in region:
                    if not i == -1:
                        vertices.append(vor.vertices[i])

                # Find the edges of the surface that intersect the infinite cell
                for i in range(len(vertices)):
                    A = vertices[i]
                    B = vertices[(i + 1) % len(vertices)]

                    for j in range(len(surface)):
                        C = surface[j]
                        D = surface[(j + 1) % len(surface)]

                        if np.cross(B - A, C - A) * np.cross(B - A, D - A) < 0:
                            # Intersection between AB and CD
                            if np.cross(D - C, A - C) * np.cross(D - C, B - C) < 0:
                                # Intersection between CD and AB
                                vertices.insert(
                                    i + 1,
                                    C
                                    + np.cross(D - C, A - C)
                                    / np.cross(D - C, B - C)
                                    * (B - A),
                                )
                                vertices.insert(i + 2, D)
                                break
                blockList.append(Block_2D(np.array(vertices), b=b, material=block_material))
        super().__init__(blockList)
