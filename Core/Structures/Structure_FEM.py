"""
Structure_FEM - Continuous Finite Element Structures
=====================================================

This module implements pure FEM structures for 2D plane stress/strain analysis.
Each node has 2 DOFs: [ux, uy] (translations only, no rotation).

Key Concepts for Students:
--------------------------
1. **Element Assembly**: Elements are stored in list_fes. Each element knows its
   local nodes and stiffness. Global assembly sums contributions at shared nodes.

2. **DOF Numbering**: For a node i, DOFs are at indices:
   - node_dof_offsets[i] = ux (horizontal displacement)
   - node_dof_offsets[i] + 1 = uy (vertical displacement)

3. **Stiffness Assembly Pattern**:
   K_global[np.ix_(elem.dofs, elem.dofs)] += elem.get_K_loc()

4. **Supported Elements**: Triangle3, Triangle6, Quad4, Quad8, Quad9

5. **Mesh Generation**: Use from_rectangular_grid() for structured meshes or
   from_mesh() to import from GMSH.

Typical Usage:
    >>> St = Structure_FEM()
    >>> St.list_fes.append(Triangle3(nodes=..., mat=..., geom=...))
    >>> St.make_nodes()                      # Build DOF system
    >>> St.fix_node(0, [0, 1])              # Fix node 0 in x,y
    >>> St.load_node(5, [0], 1000)          # Apply Fx=1000N at node 5
    >>> St = Static.solve_linear(St)         # Solve
"""

# Standard imports
from typing import List, Tuple, Union

import numpy as np

from Core.Objects.FEM.BaseFE import BaseFE
from Core.Structures.Structure_2D import Structure_2D


class Structure_FEM(Structure_2D):
    """
    Structure class for Finite Element Method (FEM) assemblies.

    This class manages a collection of 2D finite elements (triangles, quads)
    and provides methods for mesh generation, assembly, and analysis.

    Attributes
    ----------
    list_fes : List[BaseFE]
        List of finite elements in the structure.
    fixed_dofs_per_node : bool
        If True, all nodes have 3 DOFs (for hybrid compatibility).
        If False, FEM nodes have 2 DOFs.
    merge_coincident_nodes : bool
        If True, nodes at same position share DOFs.

    See Also
    --------
    Structure_Block : For discrete rigid block structures
    Hybrid : For combined block-FEM structures
    """

    def __init__(self, list_fes: Union[List[BaseFE], None] = None, fixed_dofs_per_node: bool = True,
                 merge_coincident_nodes: bool = True):
        """
        Initialize a Structure_FEM.

        Parameters
        ----------
        list_fes : List[BaseFE], optional
            Initial list of finite elements.
        fixed_dofs_per_node : bool
            If True, all nodes have 3 DOFs for mixed element compatibility.
        merge_coincident_nodes : bool
            If True, nodes at same position are merged.
        """
        super().__init__(structure_type="FEM")
        self.list_fes: List[BaseFE] = list_fes or []
        self.fixed_dofs_per_node = fixed_dofs_per_node
        self.merge_coincident_nodes = merge_coincident_nodes

        if fixed_dofs_per_node:
            self.DOF_PER_NODE = 3  # Override for fixed layout

    @classmethod
    def from_rectangular_grid(cls, nx: int, ny: int, length: float, height: float,
                              element_type: str, order: int, material, geometry):
        """
        Create Structure_FEM from a rectangular grid specification (optimized).

        Parameters
        ----------
        nx, ny : int
            Number of elements in x and y directions.
        length, height : float
            Total dimensions [m].
        element_type : str
            'triangle' or 'quad'.
        order : int
            Element order: 1 (linear), 2 (quadratic), 3 (Q9 only).
        material : PlaneStress or PlaneStrain
            Material model for all elements.
        geometry : Geometry2D
            Geometry properties (thickness).

        Returns
        -------
        Structure_FEM
            Populated structure ready for boundary conditions.
        """
        from Core.Objects.FEM.Triangles import Triangle3, Triangle6
        from Core.Objects.FEM.Quads import Quad4, Quad8, Quad9

        # Element type mapping
        element_map = {
            ('triangle', 1): Triangle3,
            ('triangle', 2): Triangle6,
            ('quad', 1): Quad4,
            ('quad', 2): Quad8,
            ('quad', 3): Quad9,
        }

        ElementClass = element_map.get((element_type, order))
        if ElementClass is None:
            raise ValueError(f"Unsupported element type/order: {element_type}, {order}")

        # Generate full node grid
        mul = 2 if order > 1 else 1
        nnx = nx * mul + 1
        nny = ny * mul + 1

        x = np.linspace(0, length, nnx)
        y = np.linspace(0, height, nny)
        xv, yv = np.meshgrid(x, y)
        all_nodes = np.column_stack([xv.ravel(), yv.ravel()])

        # Compute connectivity vectorially (uses indices into full grid)
        connectivity_full = cls._compute_connectivity(nx, ny, nnx, element_type, order)

        # Find which nodes are actually used (important for Q8 which skips center nodes)
        used_nodes = np.unique(connectivity_full.ravel())
        n_nodes = len(used_nodes)

        # Create mapping from old indices to new indices
        old_to_new = np.full(len(all_nodes), -1, dtype=int)
        old_to_new[used_nodes] = np.arange(n_nodes)

        # Extract only the used nodes
        nodes = all_nodes[used_nodes]

        # Remap connectivity to new node indices
        connectivity = old_to_new[connectivity_full]

        # Create structure instance
        structure = cls(fixed_dofs_per_node=False)

        # Set nodes directly (bypass _add_node_if_new)
        structure.list_nodes = [list(n) for n in nodes]

        # For 2D solid elements, all nodes have 2 DOFs
        structure.node_dof_counts = [2] * n_nodes
        structure.node_dof_offsets = list(range(0, 2 * n_nodes + 1, 2))
        structure.nb_dofs = 2 * n_nodes

        # Create elements with pre-computed connectivity
        n_elements = len(connectivity)
        structure.list_fes = [None] * n_elements

        for idx in range(n_elements):
            node_indices = connectivity[idx]
            element_nodes = [nodes[i].tolist() for i in node_indices]

            # Create element
            elem = ElementClass(nodes=element_nodes, mat=material, geom=geometry)

            # Set connectivity directly (bypass make_connect)
            elem.connect = node_indices.tolist()
            elem.dofs = []
            for node_idx in node_indices:
                start_dof = 2 * node_idx
                elem.dofs.extend([start_dof, start_dof + 1])
            elem.dofs = np.array(elem.dofs, dtype=int)

            structure.list_fes[idx] = elem

        # Initialize state vectors
        structure.U = np.zeros(structure.nb_dofs, dtype=float)
        structure.P = np.zeros(structure.nb_dofs, dtype=float)
        structure.P_fixed = np.zeros(structure.nb_dofs, dtype=float)
        structure.dof_fix = np.array([], dtype=int)
        structure.dof_free = np.arange(structure.nb_dofs, dtype=int)
        structure.nb_dof_fix = 0
        structure.nb_dof_free = structure.nb_dofs

        return structure

    @staticmethod
    def _compute_connectivity(nx: int, ny: int, nnx: int,
                              element_type: str, order: int) -> np.ndarray:
        """Compute element connectivity for structured grids (vectorized)."""
        # Create grid of cell indices
        i_idx = np.arange(nx)
        j_idx = np.arange(ny)
        ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
        ii = ii.ravel()
        jj = jj.ravel()

        if element_type == 'triangle':
            if order == 1:
                base = ii + jj * nnx
                tri1 = np.column_stack([base, base + 1, base + 1 + nnx])
                tri2 = np.column_stack([base, base + 1 + nnx, base + nnx])
                n_cells = nx * ny
                conn = np.empty((2 * n_cells, 3), dtype=np.int64)
                conn[0::2] = tri1
                conn[1::2] = tri2
            else:  # order == 2
                bi = ii * 2
                bj = jj * 2
                base = bi + bj * nnx
                c0, c1, c2, c3 = base, base + 2, base + 2 + 2 * nnx, base + 2 * nnx
                m01, m12, m20 = base + 1, base + 2 + nnx, base + 1 + nnx
                m23, m30 = base + 1 + 2 * nnx, base + nnx
                tri1 = np.column_stack([c0, c1, c2, m01, m12, m20])
                tri2 = np.column_stack([c0, c2, c3, m20, m23, m30])
                n_cells = nx * ny
                conn = np.empty((2 * n_cells, 6), dtype=np.int64)
                conn[0::2] = tri1
                conn[1::2] = tri2
        else:  # quad
            if order == 1:
                base = ii + jj * nnx
                conn = np.column_stack([base, base + 1, base + 1 + nnx, base + nnx])
            elif order == 2:
                bi = ii * 2
                bj = jj * 2
                base = bi + bj * nnx
                conn = np.column_stack([
                    base, base + 2, base + 2 + 2 * nnx, base + 2 * nnx,
                          base + 1, base + 2 + nnx, base + 1 + 2 * nnx, base + nnx
                ])
            else:  # order == 3 (Q9)
                bi = ii * 2
                bj = jj * 2
                base = bi + bj * nnx
                conn = np.column_stack([
                    base, base + 2, base + 2 + 2 * nnx, base + 2 * nnx,
                          base + 1, base + 2 + nnx, base + 1 + 2 * nnx, base + nnx,
                          base + 1 + nnx
                ])

        return conn

    @classmethod
    def from_mesh(cls, mesh, material, geometry, element_class=None, prefer_quad9=False):
        """
        Create Structure_FEM from a Mesh object.

        Parameters
        ----------
        mesh : Mesh
            Mesh instance (already generated or read from file).
        material : PlaneStress or PlaneStrain
            Material model for all elements.
        geometry : Geometry2D
            Geometry properties (thickness).
        element_class : type, optional
            Element class (auto-detected if None).
        prefer_quad9 : bool
            Use Quad9 instead of Quad8 for order=2 quads.

        Returns
        -------
        Structure_FEM
            Populated structure ready for boundary conditions.
        """

        # Validate mesh has been generated or read
        # For in-memory meshes (from batch generation), _nodes/_elements are set directly
        if not mesh.is_in_memory():
            if mesh._mesh is None:
                try:
                    mesh.read_mesh()
                except Exception:
                    raise ValueError(
                        "Mesh has not been generated. Call mesh.generate_mesh() first "
                        "or provide mesh_file to read from."
                    )

        # Auto-detect element class if not provided
        if element_class is None:
            element_class = cls._detect_element_class(mesh, prefer_quad9=prefer_quad9)

        # Get nodes and element connectivity from mesh
        nodes_array = mesh.nodes()  # (n_nodes, 2)
        elements_array = mesh.elements(prefer_quad9=prefer_quad9)  # (n_elements, nodes_per_element)

        if elements_array.size == 0:
            raise ValueError(
                f"No elements found for type '{mesh.element_type}' with order {mesh.order}. "
                "Check that mesh was generated correctly."
            )

        # Create structure instance
        # Mesh-generated elements are always 2D solids with 2 DOFs per node
        structure = cls(fixed_dofs_per_node=False)

        # Create element objects from connectivity
        print(f"Creating {len(elements_array)} {element_class.__name__} elements from mesh...")

        for elem_connectivity in elements_array:
            # Get node coordinates for this element
            elem_node_coords = [tuple(nodes_array[node_id]) for node_id in elem_connectivity]

            # Create element instance
            element = element_class(nodes=elem_node_coords, mat=material, geom=geometry)

            # Add to structure
            structure.add_fe(element)

        print(f"[OK] Created {len(structure.list_fes)} elements")
        print(f"  Element type: {element_class.__name__}")
        print(f"  Nodes per element: {elements_array.shape[1]}")
        print(f"  Total mesh nodes: {len(nodes_array)}")

        return structure

    @staticmethod
    def _detect_element_class(mesh, prefer_quad9=False):
        """Detect appropriate element class from Mesh properties."""
        from Core.Objects.FEM.Triangles import Triangle3, Triangle6
        from Core.Objects.FEM.Quads import Quad4, Quad8, Quad9

        # For quad order=2, check if user wants Quad9 (9-node Lagrangian)
        # GMSH always generates Quad9 for order=2, but we convert to Quad8 by default
        # Only use Quad9 if prefer_quad9=True AND mesh actually has quad9 elements
        if mesh.element_type == "quad" and mesh.order == 2:
            if prefer_quad9 and mesh.has_quad9():
                return Quad9
            else:
                return Quad8

        # Map (element_type, order) -> element class
        element_map = {
            ("triangle", 1): Triangle3,
            ("triangle", 2): Triangle6,
            ("quad", 1): Quad4,
            ("quad", 2): Quad8,
            ("quad", 3): Quad9,
        }

        key = (mesh.element_type, mesh.order)
        element_class = element_map.get(key)

        if element_class is None:
            raise ValueError(
                f"Unsupported element type/order: {mesh.element_type} (order {mesh.order}). "
                f"Supported combinations: {list(element_map.keys())}"
            )

        return element_class

    def add_fe(self, fe: BaseFE):
        """
        Add a finite element to the structure.

        Parameters
        ----------
        fe : BaseFE
            A pre-constructed finite element object.
        """
        if not isinstance(fe, BaseFE):
            raise TypeError(f"Argument must be an instance of FE, got {type(fe).__name__}")
        self.list_fes.append(fe)

    def _make_nodes_fem(self):
        """Generate nodes from finite elements with flexible or fixed DOF strategy."""
        # If merge_coincident_nodes=False, force creation of new nodes
        force_new = not getattr(self, 'merge_coincident_nodes', True)

        for fe in self.list_fes:
            # Choose DOF count strategy
            if self.fixed_dofs_per_node:
                # Fixed: all nodes have same DOF count (3 for 2D structures)
                dof_count = 3
            else:
                # Variable: get DOF count from element type
                dof_count = getattr(fe, 'DOFS_PER_NODE', 3)

            for j, node in enumerate(fe.nodes):
                # Add node if new, or get existing node index
                index = self._add_node_if_new(node, dof_count=dof_count, force_new=force_new)
                # Create element connectivity to global DOFs
                fe.make_connect(index, j, structure=self)

    def make_nodes(self):
        """Generate nodes from finite elements and initialize DOFs."""
        self._make_nodes_fem()

        # Use flexible DOF calculation (supports variable DOFs per node)
        self.nb_dofs = self.compute_nb_dofs()
        self.U = np.zeros(self.nb_dofs, dtype=float)
        self.P = np.zeros(self.nb_dofs, dtype=float)
        self.P_fixed = np.zeros(self.nb_dofs, dtype=float)

        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.arange(self.nb_dofs, dtype=int)

        self.nb_dof_fix = len(self.dof_fix)
        self.nb_dof_free = len(self.dof_free)

    def _get_P_r_fem(self):
        """Assemble internal force vector from FEM elements."""
        self.dofs_defined()
        if not hasattr(self, "P_r"):
            self.P_r = np.zeros(self.nb_dofs, dtype=float)

        for fe in self.list_fes:
            q_glob = self.U[fe.dofs]
            p_glob = fe.get_p_glob(q_glob)
            self.P_r[fe.dofs] += p_glob

    def get_P_r(self):
        """Assemble global internal force vector."""
        self.P_r = np.zeros(self.nb_dofs, dtype=float)
        self._get_P_r_fem()

    def _mass_fem(self, no_inertia: bool = False):
        """Assemble mass matrix contributions from FEM elements."""
        for fe in getattr(self, "list_fes", []):
            mass_fe = fe.get_mass(no_inertia=no_inertia)
            if mass_fe is None:
                continue
            dofs = np.asarray(fe.dofs, dtype=int)
            self.M[np.ix_(dofs, dofs)] += mass_fe

    def get_M_str(self, no_inertia: bool = False):
        """Assemble global mass matrix."""
        self.dofs_defined()
        self.M = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._mass_fem(no_inertia=no_inertia)
        return self.M

    def _stiffness_fem(self):
        """Assemble tangent stiffness matrix contributions from FEM elements."""
        for fe in getattr(self, "list_fes", []):
            k_glob = fe.get_k_glob()
            dofs = np.asarray(fe.dofs, dtype=int)
            self.K[np.ix_(dofs, dofs)] += k_glob

    def _stiffness0_fem(self):
        """Assemble initial stiffness matrix contributions from FEM elements."""
        for fe in getattr(self, "list_fes", []):
            k_glob0 = fe.get_k_glob0()
            dofs = np.asarray(fe.dofs, dtype=int)
            self.K0[np.ix_(dofs, dofs)] += k_glob0

    def _stiffness_LG_fem(self):
        """Assemble geometric stiffness matrix contributions from FEM elements."""
        for fe in getattr(self, "list_fes", []):
            k_glob_LG = fe.get_k_glob_LG()
            dofs = np.asarray(fe.dofs, dtype=int)
            self.K_LG[np.ix_(dofs, dofs)] += k_glob_LG

    def get_K_str(self):
        """Assemble global tangent stiffness matrix."""
        self.dofs_defined()
        self.K = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_fem()
        return self.K

    def get_K_str0(self):
        """Assemble global initial stiffness matrix."""
        self.dofs_defined()
        self.K0 = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness0_fem()
        return self.K0

    def get_K_str_LG(self):
        """Assemble global geometric stiffness matrix."""
        self.dofs_defined()
        self.K_LG = np.zeros((self.nb_dofs, self.nb_dofs), dtype=float)
        self._stiffness_LG_fem()
        return self.K_LG

    def set_lin_geom(self, lin_geom=True):
        """Set linear or nonlinear geometry for all elements."""
        for fe in self.list_fes:
            fe.lin_geom = lin_geom

    def commit(self):
        """Commit the current state of all elements."""
        pass

    def suppress_drilling_dofs(self, tolerance: float = 1e-12) -> int:
        """
        Automatically fix rotational DOFs that have no stiffness contribution.

        Parameters
        ----------
        tolerance : float
            DOFs with diagonal stiffness below this value are fixed.

        Returns
        -------
        int
            Number of drilling DOFs that were suppressed.
        """
        # Ensure K is assembled
        K = getattr(self, 'K', None) or getattr(self, 'K0', None)
        if K is None:
            self.get_K_str0()
            K = self.K0

        diag_K = np.diag(K)
        dofs_to_fix = []

        # Iterate through all nodes using the variable DOF system
        for node_id in range(len(self.list_nodes)):
            # Get DOF count for this node
            if node_id < len(self.node_dof_counts):
                dof_count = self.node_dof_counts[node_id]
            else:
                dof_count = self.DOF_PER_NODE

            # Only check nodes with 3 DOFs (which include rotation)
            if dof_count < 3:
                continue

            # Get the rotation DOF index (3rd DOF, index 2 relative to node start)
            node_dofs = self.get_dofs_from_node(node_id)
            if len(node_dofs) >= 3:
                rot_dof = node_dofs[2]  # θz is the 3rd DOF

                # Check if stiffness is effectively zero
                if abs(diag_K[rot_dof]) < tolerance:
                    # And if it's not already fixed by the user
                    if rot_dof in self.dof_free:
                        dofs_to_fix.append(rot_dof)

        if dofs_to_fix:
            # Remove from free, add to fixed
            self.dof_free = np.setdiff1d(self.dof_free, dofs_to_fix)
            self.dof_fix = np.union1d(self.dof_fix, dofs_to_fix)
            self.nb_dof_free = len(self.dof_free)
            self.nb_dof_fix = len(self.dof_fix)

        return len(dofs_to_fix)

    def revert_commit(self):
        """Revert to the last committed state."""
        pass

    def _compute_nodal_stress_strain(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute averaged nodal stress and strain from element contributions.

        Returns
        -------
        avg_stress : np.ndarray of shape (n_nodes, 3)
            Averaged stress [sigma_xx, sigma_yy, tau_xy] at each node.
        avg_strain : np.ndarray of shape (n_nodes, 3)
            Averaged strain [eps_xx, eps_yy, gamma_xy] at each node.
        """
        n_nodes = len(self.list_nodes)

        # Accumulators for nodal stress and strain averaging
        stress_sum = np.zeros((n_nodes, 3))  # [sigma_xx, sigma_yy, tau_xy]
        strain_sum = np.zeros((n_nodes, 3))  # [eps_xx, eps_yy, gamma_xy]
        count = np.zeros(n_nodes)

        # Collect stress and strain contributions from each element
        for fe in self.list_fes:
            # Extract element displacements from global U
            u_elem = self.U[fe.dofs]

            # Compute stress and strain at element nodes (single call)
            nodal_stress, nodal_strain = fe.compute_nodal_stress_strain(u_elem)

            # Accumulate to global nodes
            for local_idx, global_idx in enumerate(fe.connect):
                stress_sum[global_idx] += nodal_stress[local_idx]
                strain_sum[global_idx] += nodal_strain[local_idx]
                count[global_idx] += 1

        # Average values (avoid division by zero for unused nodes)
        count[count == 0] = 1
        avg_stress = stress_sum / count[:, np.newaxis]
        avg_strain = strain_sum / count[:, np.newaxis]

        return avg_stress, avg_strain

    def compute_nodal_von_mises(self) -> np.ndarray:
        """
        Compute Von Mises stress at all nodes.

        Returns
        -------
        von_mises : np.ndarray of shape (n_nodes,)
            Von Mises stress at each node.
        """
        # Get averaged nodal stress (strain not needed for Von Mises)
        avg_stress, _ = self._compute_nodal_stress_strain()

        # Compute Von Mises stress
        # sigma_vm = sqrt(sigma_xx^2 - sigma_xx*sigma_yy + sigma_yy^2 + 3*tau_xy^2)
        sxx = avg_stress[:, 0]
        syy = avg_stress[:, 1]
        txy = avg_stress[:, 2]
        von_mises = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * txy ** 2)

        return von_mises

    def compute_nodal_directional_stress(self, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normal and shear stress on a plane defined by its tangent angle.

        Parameters
        ----------
        angle : float
            Angle of the plane's tangent vector in radians.

        Returns
        -------
        sigma_n : np.ndarray
            Normal stress acting on the plane.
        tau_nt : np.ndarray
            Shear stress acting along the plane.
        """
        avg_stress, _ = self._compute_nodal_stress_strain()
        sxx = avg_stress[:, 0]
        syy = avg_stress[:, 1]
        txy = avg_stress[:, 2]

        # Convert tangent angle to normal angle for stress transformation
        # Plane is defined by tangent 'angle', so normal is 'angle + pi/2'
        phi = angle + np.pi / 2

        c = np.cos(phi)
        s = np.sin(phi)

        # Normal stress on the plane
        sigma_n = sxx * c ** 2 + syy * s ** 2 + 2 * txy * s * c

        # Shear stress on the plane
        # Note: Sign convention for shear can vary. This computes shear on the face
        # with normal 'phi' in the direction of tangent 'phi - pi/2'.
        tau_nt = -(sxx - syy) * s * c + txy * (c ** 2 - s ** 2)

        return sigma_n, tau_nt

    def compute_nodal_principal_stress(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute principal stresses at all nodes.

        Returns
        -------
        sigma_1, sigma_2 : np.ndarray
            Maximum and minimum principal stresses.
        tau_max : np.ndarray
            Maximum shear stress.
        """
        avg_stress, _ = self._compute_nodal_stress_strain()
        sxx = avg_stress[:, 0]
        syy = avg_stress[:, 1]
        txy = avg_stress[:, 2]

        # Center and Radius of Mohr's circle
        center = (sxx + syy) / 2
        radius = np.sqrt(((sxx - syy) / 2) ** 2 + txy ** 2)

        sigma_1 = center + radius
        sigma_2 = center - radius
        tau_max = radius

        return sigma_1, sigma_2, tau_max