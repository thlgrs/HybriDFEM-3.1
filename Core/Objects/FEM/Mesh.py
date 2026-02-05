from typing import Dict, Optional, Any, TYPE_CHECKING
from typing import Tuple, List

if TYPE_CHECKING:
    pass

try:
    import gmsh
    GMSH_AVAILABLE = True
except (ImportError, OSError):
    GMSH_AVAILABLE = False
    gmsh = None

import matplotlib.pyplot as plt
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False
    meshio = None

import numpy as np
from matplotlib.collections import LineCollection


class Mesh:
    """
    Create or read a 2D mesh (triangles or quads, linear or quadratic),
    expose nodes/elements/physical-edge groups, quick plot, and VTK export.
    """

    def __init__(self,
                 points: Optional[List[Tuple[float, float]]] = None,  # Boxing points
                 mesh_file: Optional[str] = None,
                 element_type: str = "triangle",  # 'triangle'/'tri' or 'quad'
                 element_size: float = 0.1,
                 order: int = 2,  # 1=linear, 2=quadratic
                 name: str = "myMesh",
                 edge_groups: Optional[Dict[str, List[int]]] = None,  # indices into boundary edges (CCW)
                 ):
        if points is None and mesh_file is None:
            raise ValueError("Provide either `points` or `mesh_file`.")
        self.points_list = points
        self.mesh_file = mesh_file
        self.element_type = (
            "triangle" if element_type in ("tri", "triangle") else "quad"
        )
        self.element_size = float(element_size)
        self.order = int(order)
        self.name = str(name)
        self.edge_groups = edge_groups or {}
        self._mesh: Optional[meshio.Mesh] = None
        self.generated = False

        # In-memory mesh data (for batch generation without file I/O)
        self._nodes: Optional[np.ndarray] = None
        self._elements: Optional[np.ndarray] = None

    # -- Mesh generation -----------------------------------------------------
    def generate_mesh(self) -> None:
        """
        Build a polygon from `points_list`, mesh it with Gmsh, create
        physical groups: 'domain' (surface) and named line groups in edge_groups.
        """
        if self.points_list is None:
            raise RuntimeError(
                "Cannot generate: no geometry defined (points_list is None)."
            )

        gmsh_init_here = not gmsh.isInitialized()
        if gmsh_init_here:
            gmsh.initialize()
        try:
            gmsh.model.add(self.name)

            # Points + boundary lines
            pts = [
                gmsh.model.geo.addPoint(x, y, 0.0, self.element_size)
                for x, y in self.points_list
            ]
            lines = [
                gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)])
                for i in range(len(pts))
            ]
            loop = gmsh.model.geo.addCurveLoop(lines)
            surface = gmsh.model.geo.addPlaneSurface([loop])
            gmsh.model.geo.synchronize()

            # Physical groups
            dom_tag = gmsh.model.addPhysicalGroup(2, [surface])
            gmsh.model.setPhysicalName(2, dom_tag, "domain")
            for name, line_indices in (self.edge_groups or {}).items():
                try:
                    phys = gmsh.model.addPhysicalGroup(
                        1, [lines[i] for i in line_indices]
                    )
                    gmsh.model.setPhysicalName(1, phys, name)
                except Exception as e:
                    print(f"[warn] failed creating physical group '{name}': {e}")

            # Meshing options
            if self.element_type == "quad":
                gmsh.model.mesh.setRecombine(2, surface)
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
            gmsh.option.setNumber("Mesh.ElementOrder", self.order)

            gmsh.model.mesh.generate(2)

            filename = self.mesh_file or f"{self.name}.msh"
            gmsh.write(filename)
            self.mesh_file = filename
            self.generated = True

            self._mesh = meshio.read(self.mesh_file)

            if self._mesh.field_data:
                print("\nMeshio Physical Groups:")
                for name, (tag, dim) in self._mesh.field_data.items():
                    print(f"  '{name}': tag={tag}, dim={dim}")
        finally:
            if gmsh_init_here:
                gmsh.finalize()

    def read_mesh(self) -> Any:
        """Read mesh from file or return cached mesh object.

        Note: For batch-generated meshes (in-memory), this method is not
        available. Use nodes() and elements() methods directly instead.
        """
        if self._mesh is None:
            if self.mesh_file is None:
                # Check if this is an in-memory mesh from batch generation
                if self._nodes is not None and self._elements is not None:
                    raise RuntimeError(
                        "This mesh was created in-memory (batch generation). "
                        "Use nodes() and elements() methods directly instead of read_mesh()."
                    )
                raise RuntimeError("No mesh available to read.")
            self._mesh = meshio.read(self.mesh_file)
        return self._mesh

    def is_in_memory(self) -> bool:
        """Check if this mesh uses in-memory data (from batch generation)."""
        return self._nodes is not None and self._elements is not None

    def nodes(self) -> np.ndarray:
        """Return node coordinates as (n_nodes, 2) array."""
        # Check for in-memory data first (from batch generation)
        if self._nodes is not None:
            return self._nodes.copy()
        # Fall back to file-based mesh
        return self.read_mesh().points[:, :2].copy()

    def elements(self, prefer_quad9: bool = False) -> np.ndarray:
        """
        Element connectivities for chosen family/order.

        MeshIO names:
          triangle: 'triangle' (3), 'triangle6' (6)
          quad    : 'quad' (4), 'quad8' (8), 'quad9' (9)

        Note: Gmsh generates quad9 (9-node Lagrangian) for order=2 quads.
        By default, we convert to quad8 (8-node serendipity) by dropping the center node.
        Use prefer_quad9=True to keep the full 9-node elements.

        Parameters
        ----------
        prefer_quad9 : bool, optional
            If True and mesh contains quad9 elements, return them directly
            instead of converting to quad8. Default is False for backward
            compatibility.

        Returns
        -------
        np.ndarray
            Element connectivity array (n_elements × nodes_per_element)
        """
        # Check for in-memory data first (from batch generation)
        if self._elements is not None:
            return self._elements.copy()

        # Fall back to file-based mesh
        md = self.read_mesh().cells_dict
        if self.element_type == "triangle":
            key = "triangle6" if self.order == 2 else "triangle"
            return md.get(key, np.empty((0, 0), dtype=int))
        else:
            if self.order == 1:
                return md.get("quad", np.empty((0, 0), dtype=int))
            else:
                # Order 2: prefer quad9 if requested, otherwise convert to quad8
                if prefer_quad9 and "quad9" in md:
                    return md["quad9"]
                elif "quad8" in md:
                    return md["quad8"]
                elif "quad9" in md:
                    # Convert quad9 to quad8 by dropping center node (index 8)
                    # Gmsh quad9 ordering: corners [0,1,2,3], mid-sides [4,5,6,7], center [8]
                    return md["quad9"][:, :8]  # Keep only first 8 nodes
                else:
                    return np.empty((0, 0), dtype=int)

    def has_quad9(self) -> bool:
        """Check if the mesh contains quad9 elements."""
        # For in-memory meshes, check element type and node count
        if self.is_in_memory():
            return (self.element_type == "quad" and
                    self._elements is not None and
                    self._elements.shape[1] == 9)
        # For file-based meshes, check cells_dict
        md = self.read_mesh().cells_dict
        return "quad9" in md

    def get_boundary_nodes(self, group_name: str) -> List[int]:
        """
        Get list of node IDs on a boundary defined by a physical group.

        Args:
            group_name: Name of the physical group (as defined in edge_groups)

        Returns:
            Sorted list of unique node IDs on the boundary

        Raises:
            ValueError: If the physical group name is not found in the mesh
        """
        mesh = self.read_mesh()

        # Check if mesh has field_data (physical groups)
        if not hasattr(mesh, 'field_data') or mesh.field_data is None:
            raise ValueError(
                "Mesh has no physical groups. Define edge_groups during mesh generation."
            )

        # Find the physical group tag
        if group_name not in mesh.field_data:
            available = list(mesh.field_data.keys())
            raise ValueError(
                f"Physical group '{group_name}' not found in mesh. "
                f"Available groups: {available}"
            )

        # Get the tag and dimension for this physical group
        tag, dim = mesh.field_data[group_name]

        # Physical groups for edges/lines should have dimension 1
        if dim != 1:
            raise ValueError(
                f"Physical group '{group_name}' has dimension {dim} (expected 1 for boundary). "
                "Only line/edge groups can be used for boundary node extraction."
            )

        # Extract nodes from cells with this physical tag
        boundary_nodes = set()

        # Iterate through cell blocks to find matching physical tag
        if hasattr(mesh, 'cell_data') and mesh.cell_data:
            for cell_block_idx, cell_block in enumerate(mesh.cells):
                cell_type = cell_block.type
                cell_data = cell_block.data

                # Check if this cell block has physical tags
                if 'gmsh:physical' in mesh.cell_data:
                    physical_tags = mesh.cell_data['gmsh:physical'][cell_block_idx]

                    # Find cells with matching tag
                    for cell_idx, cell_tag in enumerate(physical_tags):
                        if cell_tag == tag:
                            # Add all nodes from this cell to boundary set
                            cell_nodes = cell_data[cell_idx]
                            boundary_nodes.update(cell_nodes)

        if not boundary_nodes:
            # Fallback: try alternative cell_data structure
            # Some meshio versions use different formats
            for cell_type, cells in mesh.cells_dict.items():
                # Only process line elements (1D boundaries)
                if 'line' in cell_type:
                    # Check if cell_data exists for this cell type
                    cell_data_key = f"gmsh:physical"
                    if hasattr(mesh, 'cell_data_dict') and cell_data_key in mesh.cell_data_dict:
                        if cell_type in mesh.cell_data_dict[cell_data_key]:
                            physical_tags = mesh.cell_data_dict[cell_data_key][cell_type]
                            for cell_idx, cell_tag in enumerate(physical_tags):
                                if cell_tag == tag:
                                    boundary_nodes.update(cells[cell_idx])

        if not boundary_nodes:
            raise ValueError(
                f"No boundary nodes found for physical group '{group_name}'. "
                "This may indicate an issue with mesh generation or physical group tagging."
            )

        # Return sorted list
        return sorted(list(boundary_nodes))

    def get_all_boundary_groups(self) -> Dict[str, int]:
        """
        Get all available boundary physical groups in the mesh.

        Returns:
            Dictionary mapping group name to (tag, dimension)
        """
        mesh = self.read_mesh()

        if not hasattr(mesh, 'field_data') or mesh.field_data is None:
            return {}

        # Return only dimension-1 (line) groups
        return {
            name: (tag, dim)
            for name, (tag, dim) in mesh.field_data.items()
            if dim == 1
        }

    def plot(
            self, save_path: Optional[str] = None, title: Optional[str] = None
    ) -> None:
        mesh = self.read_mesh()
        pts = mesh.points[:, :2]
        segs: List[Tuple[np.ndarray, np.ndarray]] = []

        for cb in mesh.cells:
            t = cb.type
            data = cb.data
            if t == "line":
                for e in data:
                    segs.append((pts[e[0]], pts[e[1]]))
            elif t == "line3":
                for e in data:
                    segs.append((pts[e[0]], pts[e[2]]))
                    segs.append((pts[e[2]], pts[e[1]]))
            elif t == "triangle":
                for e in data:
                    cyc = [0, 1, 2, 0]
                    for i in range(3):
                        segs.append((pts[e[cyc[i]]], pts[e[cyc[i + 1]]]))
            elif t == "triangle6":
                for e in data:
                    segs += [(pts[e[0]], pts[e[3]]), (pts[e[3]], pts[e[1]])]
                    segs += [(pts[e[1]], pts[e[4]]), (pts[e[4]], pts[e[2]])]
                    segs += [(pts[e[2]], pts[e[5]]), (pts[e[5]], pts[e[0]])]
            elif t == "quad":
                for e in data:
                    cyc = [0, 1, 2, 3, 0]
                    for i in range(4):
                        segs.append((pts[e[cyc[i]]], pts[e[cyc[i + 1]]]))
            elif t == "quad8":
                for e in data:
                    segs += [(pts[e[0]], pts[e[4]]), (pts[e[4]], pts[e[1]])]
                    segs += [(pts[e[1]], pts[e[5]]), (pts[e[5]], pts[e[2]])]
                    segs += [(pts[e[2]], pts[e[6]]), (pts[e[6]], pts[e[3]])]
                    segs += [(pts[e[3]], pts[e[7]]), (pts[e[7]], pts[e[0]])]
            elif t == "quad9":
                # Same edge visualization as quad8 (center node not shown as edge)
                for e in data:
                    segs += [(pts[e[0]], pts[e[4]]), (pts[e[4]], pts[e[1]])]
                    segs += [(pts[e[1]], pts[e[5]]), (pts[e[5]], pts[e[2]])]
                    segs += [(pts[e[2]], pts[e[6]]), (pts[e[6]], pts[e[3]])]
                    segs += [(pts[e[3]], pts[e[7]]), (pts[e[7]], pts[e[0]])]

        lc = LineCollection(segs, linewidths=0.5, colors="k")
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(title or f"{self.name} ({self.element_type}, order={self.order})")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=160)
            plt.close(fig)
        else:
            plt.show()

    # -- Batch mesh generation ------------------------------------------------
    @classmethod
    def generate_batch(
            cls,
            surfaces: List[Dict],
            element_type: str = "triangle",
            element_size: float = 0.1,
            order: int = 1,
            name_prefix: str = "surface",
            verbose: bool = False
    ) -> List['Mesh']:
        """
        Generate multiple disconnected meshes in a single GMSH operation.

        Args:
            surfaces: List of surface definitions with 'points' key (required),
                     optional 'element_size', 'edge_groups', 'name'
            element_type: 'triangle' or 'quad' for all surfaces
            element_size: Default target element size
            order: Element order (1=linear, 2=quadratic)
            name_prefix: Prefix for auto-generated surface names
            verbose: If True, print progress information

        Returns:
            List of Mesh objects with in-memory nodes/elements data
        """
        if not GMSH_AVAILABLE:
            raise ImportError("GMSH is not available. Install with: pip install gmsh")

        element_type = "triangle" if element_type in ("tri", "triangle") else "quad"

        # GMSH element type codes
        GMSH_ELEM_TYPES = {
            ('triangle', 1): 2,  # 3-node triangle
            ('triangle', 2): 9,  # 6-node triangle
            ('quad', 1): 3,  # 4-node quad
            ('quad', 2): 10,  # 9-node quad (Lagrangian)
        }
        target_elem_type = GMSH_ELEM_TYPES.get((element_type, order))

        gmsh.initialize()
        try:
            gmsh.model.add("batch_mesh")
            if not verbose:
                gmsh.option.setNumber("General.Terminal", 0)  # Suppress output

            surface_tags = []
            surface_elem_sizes = []

            # Step 1: Create all surfaces in one model
            for idx, surf_def in enumerate(surfaces):
                points = surf_def.get('points')
                if points is None:
                    raise ValueError(f"Surface {idx} missing 'points' key")

                surf_elem_size = surf_def.get('element_size', element_size)
                surface_elem_sizes.append(surf_elem_size)

                # Create points with per-surface element size
                pts = [
                    gmsh.model.geo.addPoint(x, y, 0.0, surf_elem_size)
                    for x, y in points
                ]

                # Create boundary lines
                lines = [
                    gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)])
                    for i in range(len(pts))
                ]

                # Create surface
                loop = gmsh.model.geo.addCurveLoop(lines)
                surface = gmsh.model.geo.addPlaneSurface([loop])
                surface_tags.append(surface)

            # Must synchronize BEFORE creating physical groups
            gmsh.model.geo.synchronize()

            # Now create physical groups for each surface
            for idx, surf_def in enumerate(surfaces):
                phys_tag = idx + 1
                gmsh.model.addPhysicalGroup(2, [surface_tags[idx]], phys_tag)
                surf_name = surf_def.get('name', f"{name_prefix}_{idx}")
                gmsh.model.setPhysicalName(2, phys_tag, surf_name)

            # Step 2: Set meshing options and generate
            if element_type == "quad":
                for surf in surface_tags:
                    gmsh.model.mesh.setRecombine(2, surf)
                gmsh.option.setNumber("Mesh.RecombineAll", 1)

            gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
            gmsh.option.setNumber("Mesh.ElementOrder", order)

            # Generate mesh for ALL surfaces at once
            gmsh.model.mesh.generate(2)

            if verbose:
                print(f"[Mesh.generate_batch] Generated {len(surfaces)} surfaces")

            # Step 3: Extract mesh data per surface
            meshes = []
            for idx, surf_def in enumerate(surfaces):
                phys_tag = idx + 1
                surf_name = surf_def.get('name', f"{name_prefix}_{idx}")

                # Get nodes for this physical group
                # Returns (nodeTags, coord) - 2 values, not 3
                node_tags, coords = gmsh.model.mesh.getNodesForPhysicalGroup(2, phys_tag)

                # Get elements for this surface entity
                elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(
                    2, surface_tags[idx]
                )

                # Create Mesh object with extracted data
                mesh = cls._create_from_gmsh_data(
                    node_tags=node_tags,
                    coords=coords,
                    elem_types=elem_types,
                    elem_node_tags=elem_node_tags,
                    target_elem_type=target_elem_type,
                    element_type=element_type,
                    element_size=surface_elem_sizes[idx],
                    order=order,
                    name=surf_name
                )
                meshes.append(mesh)

                if verbose:
                    print(f"  {surf_name}: {len(mesh.nodes())} nodes, "
                          f"{len(mesh.elements())} elements")

            return meshes

        finally:
            gmsh.finalize()

    @classmethod
    def _create_from_gmsh_data(
            cls,
            node_tags: np.ndarray,
            coords: np.ndarray,
            elem_types: List[int],
            elem_node_tags: List[np.ndarray],
            target_elem_type: int,
            element_type: str,
            element_size: float,
            order: int,
            name: str
    ) -> 'Mesh':
        """
        Create a Mesh instance from GMSH API data (no file needed).

        Args:
            node_tags: GMSH node tags (global IDs)
            coords: Flattened node coordinates [x0, y0, z0, x1, y1, z1, ...]
            elem_types: List of GMSH element type codes per element block
            elem_node_tags: List of node tag arrays per element block
            target_elem_type: GMSH element type code to extract
            element_type: 'triangle' or 'quad'
            element_size: Element size used for this surface
            order: Element order (1 or 2)
            name: Mesh name

        Returns:
            Mesh instance with in-memory nodes/elements data
        """
        # Create mesh instance without calling __init__ validation
        mesh = object.__new__(cls)
        mesh.points_list = None
        mesh.mesh_file = None
        mesh.element_type = element_type
        mesh.element_size = element_size
        mesh.order = order
        mesh.name = name
        mesh.edge_groups = {}
        mesh._mesh = None
        mesh.generated = True

        # Build node mapping: GMSH tag -> 0-based local index
        node_tag_to_local = {tag: i for i, tag in enumerate(node_tags)}

        # Extract node coordinates (reshape from flat array)
        nodes_array = coords.reshape(-1, 3)[:, :2]  # (n_nodes, 2)
        mesh._nodes = nodes_array

        # Find and extract elements of target type
        elements_list = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == target_elem_type:
                # Determine nodes per element from GMSH element type
                nodes_per_elem = {
                    2: 3,  # triangle
                    9: 6,  # triangle6
                    3: 4,  # quad
                    10: 9,  # quad9
                    16: 8,  # quad8
                }.get(etype, 0)

                if nodes_per_elem > 0:
                    # Reshape and renumber to local indices
                    elem_connectivity = enodes.reshape(-1, nodes_per_elem)
                    local_connectivity = np.array([
                        [node_tag_to_local[tag] for tag in elem]
                        for elem in elem_connectivity
                    ], dtype=int)
                    elements_list.append(local_connectivity)

        if elements_list:
            mesh._elements = np.vstack(elements_list)
        else:
            mesh._elements = np.empty((0, 0), dtype=int)

        return mesh
