from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np


@dataclass
class InterfacePoint:
    """
    Represents a point on the mortar interface.

    Attributes
    ----------
    position : np.ndarray
        Physical coordinates (x, y)
    normal : np.ndarray
        Outward normal vector at this point (unit vector)
    tangent : np.ndarray
        Tangent vector at this point (unit vector)
    parametric_coord : float
        Parametric coordinate along interface segment [0, 1]
    """
    position: np.ndarray  # (x, y)
    normal: np.ndarray    # (nx, ny) unit vector
    tangent: np.ndarray   # (tx, ty) unit vector
    parametric_coord: float  # ξ ∈ [0, 1]


@dataclass
class SlaveEdge:
    """
    Represents a slave (FEM) element edge on the mortar interface.

    Attributes
    ----------
    element_id : int
        Index of the FEM element in structure.list_fes
    local_edge_id : int
        Local edge index within the element (0, 1, 2 for triangles)
    global_node_ids : List[int]
        Global node IDs of nodes on this edge (in order along edge)
    local_node_ids : List[int]
        Local node indices within the element (0, 1, 2 for triangles)
    vertices : np.ndarray
        Edge vertex coordinates, shape (2, 2) for linear edges
    edge_length : float
        Physical length of the edge
    """
    element_id: int
    local_edge_id: int
    global_node_ids: List[int]
    local_node_ids: List[int]
    vertices: np.ndarray
    edge_length: float = 0.0

    def __post_init__(self):
        """Compute edge length after initialization."""
        if self.edge_length == 0.0 and len(self.vertices) >= 2:
            self.edge_length = np.linalg.norm(self.vertices[1] - self.vertices[0])


class MortarInterface:
    """
    Represents an interface segment between a Master entity and Slave (FEM) elements.

    In mortar methods, the interface is where coupling constraints are enforced.
    This class stores geometric information about the interface.

    Attributes
    ----------
    master_id : int
        ID of the master entity (e.g. block_id).
        For backward compatibility, this is initially 'block_id'.
    master_type : str
        Type of master entity: 'block' (default) or 'fem'.
    master_face_id : int
        Which face/edge of the master entity.
    master_vertices : np.ndarray
        Coordinates of master segment endpoints.
    fem_element_ids : List[int]
        Indices of FEM elements near this interface
    fem_nodes_on_interface : List[int]
        FEM node IDs that lie on this interface
    interface_length : float
        Physical length of interface segment
    normal : np.ndarray
        Average outward normal (from master into slave)
    centroid : np.ndarray
        Geometric center of interface segment
    """

    def __init__(self, master_id: int, master_face_id: int,
                 face_vertices: np.ndarray, master_type: str = 'block'):
        """
        Initialize mortar interface.

        Parameters
        ----------
        master_id : int
            Index of master entity in structure list
        master_face_id : int
            Which face/edge of the master entity
        face_vertices : np.ndarray
            Face endpoints, shape (2, 2) for 2D: [[x1,y1], [x2,y2]]
        master_type : str
            'block' or 'fem'
        """
        self.master_id = master_id
        self.master_face_id = master_face_id
        self.master_type = master_type
        self.master_vertices = np.array(face_vertices, dtype=float)

        # Initialize empty lists for FEM elements/nodes
        self.fem_element_ids: List[int] = []
        self.fem_nodes_on_interface: List[int] = []

        # For FEM-master interfaces: store master edge nodes
        self.master_global_nodes: List[int] = []
        self.master_local_nodes: List[int] = []

        # Block faces (for FEM-master interfaces where blocks are slave)
        self.block_faces: List[Tuple[int, int, np.ndarray, np.ndarray]] = []

        # Slave (FEM) edge information for true mortar integration
        self.slave_edges: List[SlaveEdge] = []
        self.slave_nodes_sorted: List[int] = []

        # Compute geometric properties
        self._compute_geometry()

    @property
    def block_id(self):
        """Backward compatibility alias."""
        return self.master_id

    @property
    def block_face_id(self):
        """Backward compatibility alias."""
        return self.master_face_id

    @property
    def block_face_vertices(self):
        """Backward compatibility alias."""
        return self.master_vertices

    def _compute_geometry(self):
        """Compute geometric properties of interface segment."""
        v0, v1 = self.master_vertices[0], self.master_vertices[1]

        # Tangent vector (along face)
        tangent = v1 - v0
        self.interface_length = np.linalg.norm(tangent)

        if self.interface_length > 1e-12:
            self.tangent = tangent / self.interface_length
        else:
            self.tangent = np.array([1.0, 0.0])

        # Normal vector (perpendicular to face, pointing outward from master)
        tx, ty = self.tangent
        self.normal = np.array([-ty, tx])

        # Centroid
        self.centroid = 0.5 * (v0 + v1)

    def add_fem_element(self, element_id: int):
        if element_id not in self.fem_element_ids:
            self.fem_element_ids.append(element_id)

    def add_fem_node(self, node_id: int):
        if node_id not in self.fem_nodes_on_interface:
            self.fem_nodes_on_interface.append(node_id)

    def add_block_face(self, block_id: int, face_id: int, v0: np.ndarray, v1: np.ndarray):
        """Add a block face to this interface (for FEM-master interfaces)."""
        self.block_faces.append((block_id, face_id, np.array(v0), np.array(v1)))

    def point_to_interface_distance(self, point: np.ndarray) -> float:
        vec = point - self.centroid
        return np.dot(vec, self.normal)

    def project_point_to_interface(self, point: np.ndarray) -> Tuple[np.ndarray, float]:
        v0 = self.master_vertices[0]
        vec = point - v0
        projection_length = np.dot(vec, self.tangent)
        projection_length = np.clip(projection_length, 0.0, self.interface_length)
        projection = v0 + projection_length * self.tangent
        parametric_coord = projection_length / self.interface_length
        return projection, parametric_coord

    def is_point_on_interface(self, point: np.ndarray, tolerance: float = 1e-6,
                               strict_containment: bool = False) -> bool:
        """
        Check if a point lies on the interface segment.

        Parameters
        ----------
        point : np.ndarray
            Point to check (x, y)
        tolerance : float
            Distance tolerance for proximity check
        strict_containment : bool
            If True, require point to project within [0, 1] parametric bounds.
            If False (default), only check distance to infinite line.
            For non-matching FEM-FEM interfaces, use False.

        Returns
        -------
        bool
            True if point is on interface within tolerance
        """
        projection, xi = self.project_point_to_interface(point)
        distance = np.linalg.norm(point - projection)
        within_tolerance = (distance < tolerance)

        if strict_containment:
            on_segment = (0.0 <= xi <= 1.0)
            return on_segment and within_tolerance
        else:
            # For non-matching meshes, allow small extension beyond segment
            # Use a relative tolerance based on interface length
            xi_tolerance = tolerance / max(self.interface_length, 1e-10)
            on_segment = (-xi_tolerance <= xi <= 1.0 + xi_tolerance)
            return on_segment and within_tolerance

    def parametric_to_physical(self, xi: float) -> np.ndarray:
        v0, v1 = self.master_vertices[0], self.master_vertices[1]
        return (1.0 - xi) * v0 + xi * v1

    def create_interface_point(self, xi: float) -> InterfacePoint:
        position = self.parametric_to_physical(xi)
        return InterfacePoint(
            position=position,
            normal=self.normal.copy(),
            tangent=self.tangent.copy(),
            parametric_coord=xi
        )

    def get_info(self) -> Dict:
        return {
            'master_id': self.master_id,
            'master_type': self.master_type,
            'face_id': self.master_face_id,
            'length': self.interface_length,
            'centroid': self.centroid,
            'normal': self.normal,
            'tangent': self.tangent,
            'n_fem_elements': len(self.fem_element_ids),
            'n_fem_nodes': len(self.fem_nodes_on_interface)
        }

    def detect_slave_edges(self, structure, tolerance: float = 1e-6) -> List[SlaveEdge]:
        """
        Detect slave (FEM) element edges that lie on this interface.
        Identical to previous logic but generalized names.
        """
        self.slave_edges = []

        # Edge definitions
        triangle_edges = [(0, 1), (1, 2), (2, 0)]
        quad_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

        for elem_id in self.fem_element_ids:
            element = structure.list_fes[elem_id]
            n_nodes = element.nd

            if n_nodes == 3:
                edge_connectivity = triangle_edges
            elif n_nodes == 6:
                edge_connectivity = [(0, 3, 1), (1, 4, 2), (2, 5, 0)]
            elif n_nodes == 4:
                edge_connectivity = quad_edges
            elif n_nodes == 8 or n_nodes == 9:
                edge_connectivity = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
            else:
                edge_connectivity = triangle_edges

            for local_edge_id, edge_nodes in enumerate(edge_connectivity):
                if len(edge_nodes) < 2: continue
                n0, n1 = edge_nodes[0], edge_nodes[-1]
                v0 = np.array(element.nodes[n0])
                v1 = np.array(element.nodes[n1])

                if (self.is_point_on_interface(v0, tolerance) and
                    self.is_point_on_interface(v1, tolerance)):

                    global_node_ids = [element.connect[n] for n in edge_nodes]
                    local_node_ids = list(edge_nodes)
                    vertices = np.array([element.nodes[n] for n in edge_nodes])

                    slave_edge = SlaveEdge(
                        element_id=elem_id,
                        local_edge_id=local_edge_id,
                        global_node_ids=global_node_ids,
                        local_node_ids=local_node_ids,
                        vertices=vertices
                    )
                    self.slave_edges.append(slave_edge)

        self._sort_slave_edges()
        self._build_slave_nodes_sorted()
        return self.slave_edges

    def detect_block_slave_edges(self, structure, tolerance: float = 1e-6) -> List[SlaveEdge]:
        """
        Detect block edges as slave edges (for FEM-master interfaces).

        When blocks are finer than FEM, block faces become the slave side
        for mortar integration.

        Parameters
        ----------
        structure : Hybrid
            Hybrid structure
        tolerance : float
            Tolerance for edge detection

        Returns
        -------
        List[SlaveEdge]
            Block edges as SlaveEdge objects
        """
        self.slave_edges = []

        for block_id, face_id, v0, v1 in self.block_faces:
            block = structure.list_blocks[block_id]
            block_node_id = block.connect  # Global node ID for block

            # Check if face endpoints are on the interface
            if not (self.is_point_on_interface(v0, tolerance) and
                    self.is_point_on_interface(v1, tolerance)):
                continue

            # Create SlaveEdge for this block face
            # Note: For blocks, we use negative element_id to distinguish from FEM
            # and store block_id in a custom way
            slave_edge = SlaveEdge(
                element_id=-(block_id + 1),  # Negative to indicate block
                local_edge_id=face_id,
                global_node_ids=[block_node_id],  # Block has single reference node
                local_node_ids=[0],
                vertices=np.array([v0, v1])
            )
            # Store block info for later use
            slave_edge.is_block_edge = True
            slave_edge.block_id = block_id

            self.slave_edges.append(slave_edge)

        self._sort_slave_edges()
        self._build_block_slave_nodes_sorted(structure)
        return self.slave_edges

    def _build_block_slave_nodes_sorted(self, structure):
        """Build sorted list of block nodes for FEM-master interfaces."""
        if not self.slave_edges:
            self.slave_nodes_sorted = []
            return

        # For block slave edges, each edge corresponds to one block
        # We collect unique block node IDs
        node_positions = {}
        for edge in self.slave_edges:
            if hasattr(edge, 'block_id'):
                block = structure.list_blocks[edge.block_id]
                block_node_id = block.connect
                if block_node_id not in node_positions:
                    midpoint = np.mean(edge.vertices[:2], axis=0)
                    _, param = self.project_point_to_interface(midpoint)
                    node_positions[block_node_id] = param

        sorted_items = sorted(node_positions.items(), key=lambda x: x[1])
        self.slave_nodes_sorted = [node_id for node_id, _ in sorted_items]

    def _sort_slave_edges(self):
        if not self.slave_edges: return
        def edge_position(edge: SlaveEdge) -> float:
            midpoint = np.mean(edge.vertices[:2], axis=0)
            _, param = self.project_point_to_interface(midpoint)
            return param
        self.slave_edges.sort(key=edge_position)

    def _build_slave_nodes_sorted(self):
        if not self.slave_edges:
            self.slave_nodes_sorted = []
            return
        node_positions = {}
        for edge in self.slave_edges:
            for i, node_id in enumerate(edge.global_node_ids):
                if node_id not in node_positions:
                    node_pos = edge.vertices[i] if i < len(edge.vertices) else edge.vertices[0]
                    _, param = self.project_point_to_interface(node_pos)
                    node_positions[node_id] = param
        sorted_items = sorted(node_positions.items(), key=lambda x: x[1])
        self.slave_nodes_sorted = [node_id for node_id, _ in sorted_items]

    def get_slave_node_count(self) -> int:
        return len(self.slave_nodes_sorted)

    def __repr__(self):
        n_edges = len(self.slave_edges) if self.slave_edges else 0
        return (f"MortarInterface(master={self.master_id} ({self.master_type}), "
                f"face={self.master_face_id}, length={self.interface_length:.4f}, "
                f"n_slave_edges={n_edges})")
