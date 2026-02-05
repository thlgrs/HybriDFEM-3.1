from typing import List, Tuple, Optional

import numpy as np
from scipy.spatial import cKDTree

from .MortarInterface import MortarInterface


def get_block_faces(block) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract faces (edges) from a 2D block polygon.

    Parameters
    ----------
    block : Block_2D
        Block object with vertices

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of face tuples: [(v0, v1), (v1, v2), ...]
        Each tuple contains start and end vertex coordinates
    """
    vertices = block.v  # Block vertices (counter-clockwise)
    n_vertices = len(vertices)

    faces = []
    for i in range(n_vertices):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % n_vertices]  # Wrap around
        faces.append((v0, v1))

    return faces


def get_fem_element_edges(element) -> List[Tuple[List[int], np.ndarray, np.ndarray]]:
    """
    Extract edges from a FEM element with local node indices.

    Parameters
    ----------
    element : FE
        FEM element with nodes attribute

    Returns
    -------
    List[Tuple[List[int], np.ndarray, np.ndarray]]
        List of (local_node_ids, v0, v1) tuples
    """
    n_nodes = element.nd
    nodes = element.nodes

    if n_nodes == 3:  # Triangle
        edges = [(0, 1), (1, 2), (2, 0)]
    elif n_nodes == 6:  # Quadratic triangle
        edges = [(0, 1), (1, 2), (2, 0)]  # Corner-to-corner only
    elif n_nodes == 4:  # Quad
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    elif n_nodes in [8, 9]:  # Quadratic quad
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Corner-to-corner only
    else:
        edges = [(0, 1)]

    result = []
    for n0, n1 in edges:
        v0 = np.array(nodes[n0])
        v1 = np.array(nodes[n1])
        result.append(([n0, n1], v0, v1))

    return result


def point_to_segment_distance(point: np.ndarray, seg_start: np.ndarray,
                              seg_end: np.ndarray) -> Tuple[float, float]:
    """
    Compute distance from point to line segment.

    Parameters
    ----------
    point : np.ndarray
        Point coordinates (x, y)
    seg_start : np.ndarray
        Segment start (x0, y0)
    seg_end : np.ndarray
        Segment end (x1, y1)

    Returns
    -------
    distance : float
        Perpendicular distance to segment
    projection_param : float
        Parameter t ∈ [0, 1] of closest point on segment
        (t=0 → seg_start, t=1 → seg_end, 0<t<1 → interior)
    """
    # Vector along segment
    seg_vec = seg_end - seg_start
    seg_length_sq = np.dot(seg_vec, seg_vec)

    if seg_length_sq < 1e-16:
        # Degenerate segment (point)
        return np.linalg.norm(point - seg_start), 0.0

    # Vector from segment start to point
    point_vec = point - seg_start

    # Project point onto segment direction
    t = np.dot(point_vec, seg_vec) / seg_length_sq

    # Clamp to segment bounds
    t = np.clip(t, 0.0, 1.0)

    # Closest point on segment
    closest = seg_start + t * seg_vec

    # Distance
    distance = np.linalg.norm(point - closest)

    return distance, t


def find_fem_nodes_near_segment(structure, seg_start: np.ndarray,
                                seg_end: np.ndarray,
                                tolerance: float,
                                n_blocks: int,
                                fem_node_tree: Optional[cKDTree] = None,
                                fem_nodes_array: Optional[np.ndarray] = None) -> List[int]:
    """
    Find FEM nodes near a line segment.

    Parameters
    ----------
    structure : Hybrid
        Hybrid structure
    seg_start : np.ndarray
        Segment start coordinates
    seg_end : np.ndarray
        Segment end coordinates
    tolerance : float
        Distance tolerance
    n_blocks : int
        Number of blocks (to skip block nodes)
    fem_node_tree : cKDTree, optional
        Pre-built KD-tree of FEM node positions for O(log n) queries.
        If None, falls back to O(n) linear scan.
    fem_nodes_array : np.ndarray, optional
        Pre-built array of FEM node coordinates (required if fem_node_tree is provided)

    Returns
    -------
    List[int]
        Node IDs of FEM nodes within tolerance of segment
    """
    # Fast path: use KD-tree if available
    if fem_node_tree is not None and fem_nodes_array is not None:
        # Query sphere around segment midpoint with radius = half-length + tolerance
        seg_mid = (seg_start + seg_end) / 2
        seg_half_len = np.linalg.norm(seg_end - seg_start) / 2
        search_radius = seg_half_len + tolerance

        # Get candidate indices (local indices into fem_nodes_array)
        candidate_local_indices = fem_node_tree.query_ball_point(seg_mid, r=search_radius)

        # Refine with exact point-to-segment distance
        nearby_nodes = []
        for local_idx in candidate_local_indices:
            node_pos = fem_nodes_array[local_idx]
            distance, t = point_to_segment_distance(node_pos, seg_start, seg_end)
            if distance < tolerance:
                # Convert local index to global node ID
                global_node_id = local_idx + n_blocks
                nearby_nodes.append(global_node_id)

        return nearby_nodes

    # Fallback: O(n) linear scan (original behavior)
    fem_node_ids = list(range(n_blocks, len(structure.list_nodes)))
    nearby_nodes = []

    for node_id in fem_node_ids:
        node_pos = structure.list_nodes[node_id]
        distance, t = point_to_segment_distance(node_pos, seg_start, seg_end)
        if distance < tolerance:
            nearby_nodes.append(node_id)

    return nearby_nodes


def find_fem_elements_near_segment(structure, seg_start: np.ndarray,
                                   seg_end: np.ndarray,
                                   tolerance: float,
                                   element_centroid_tree: Optional[cKDTree] = None,
                                   element_centroids: Optional[np.ndarray] = None,
                                   max_element_radius: Optional[float] = None) -> List[int]:
    """
    Find FEM elements whose nodes are near a line segment.

    An element is considered "near" if any of its nodes is within tolerance
    of the segment.

    Parameters
    ----------
    structure : Hybrid
        Hybrid structure
    seg_start : np.ndarray
        Segment start coordinates
    seg_end : np.ndarray
        Segment end coordinates
    tolerance : float
        Distance tolerance
    element_centroid_tree : cKDTree, optional
        Pre-built KD-tree of element centroids for O(log n) queries.
        If None, falls back to O(n) linear scan.
    element_centroids : np.ndarray, optional
        Pre-built array of element centroids (required if element_centroid_tree is provided)
    max_element_radius : float, optional
        Maximum element radius (half-diagonal) for search expansion

    Returns
    -------
    List[int]
        Element indices in structure.list_fes
    """
    # Fast path: use KD-tree if available
    if element_centroid_tree is not None:
        # Query sphere around segment midpoint
        seg_mid = (seg_start + seg_end) / 2
        seg_half_len = np.linalg.norm(seg_end - seg_start) / 2

        # Search radius = segment half-length + element radius + tolerance
        elem_radius = max_element_radius if max_element_radius else tolerance * 10
        search_radius = seg_half_len + elem_radius + tolerance

        # Get candidate element indices
        candidate_indices = element_centroid_tree.query_ball_point(seg_mid, r=search_radius)

        # Refine with exact node-to-segment distance checks
        nearby_elements = []
        for elem_id in candidate_indices:
            element = structure.list_fes[elem_id]
            for node_pos in element.nodes:
                distance, t = point_to_segment_distance(
                    np.array(node_pos), seg_start, seg_end
                )
                if distance < tolerance:
                    nearby_elements.append(elem_id)
                    break  # Element is near, no need to check other nodes

        return nearby_elements

    # Fallback: O(n) linear scan (original behavior)
    nearby_elements = []

    for elem_id, element in enumerate(structure.list_fes):
        # Check if any node of this element is near the segment
        for node_pos in element.nodes:
            distance, t = point_to_segment_distance(
                np.array(node_pos), seg_start, seg_end
            )

            if distance < tolerance:
                nearby_elements.append(elem_id)
                break  # Element is near, no need to check other nodes

    return nearby_elements


def find_block_faces_near_segment(structure, seg_start: np.ndarray,
                                  seg_end: np.ndarray,
                                  tolerance: float) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    Find block faces that overlap with a line segment.

    Parameters
    ----------
    structure : Hybrid
        Hybrid structure
    seg_start : np.ndarray
        Segment start coordinates
    seg_end : np.ndarray
        Segment end coordinates
    tolerance : float
        Distance tolerance

    Returns
    -------
    List[Tuple[int, int, np.ndarray, np.ndarray]]
        List of (block_id, face_id, face_v0, face_v1) for overlapping faces
    """
    nearby_faces = []

    for block_id, block in enumerate(structure.list_blocks):
        faces = get_block_faces(block)

        for face_id, (v0, v1) in enumerate(faces):
            # Check if either endpoint of the block face is near the segment
            dist0, t0 = point_to_segment_distance(v0, seg_start, seg_end)
            dist1, t1 = point_to_segment_distance(v1, seg_start, seg_end)

            # Face is near if either endpoint is within tolerance
            if dist0 < tolerance or dist1 < tolerance:
                nearby_faces.append((block_id, face_id, v0, v1))

    return nearby_faces


def detect_fem_master_interfaces(structure, tolerance: float = 1e-6,
                                 interface_orientation: str = None) -> List[MortarInterface]:
    """
    Detect mortar interfaces with FEM edges as master (for fine-block/coarse-FEM cases).

    This is the reverse of the standard approach: instead of iterating over block
    faces and finding nearby FEM elements, we iterate over FEM element edges and
    find nearby block faces.

    Use this when blocks are finer than FEM at the interface.

    Parameters
    ----------
    structure : Hybrid
        Hybrid structure with blocks and FEM elements
    tolerance : float
        Distance tolerance for proximity detection
    interface_orientation : str, optional
        Filter interfaces by orientation

    Returns
    -------
    List[MortarInterface]
        Detected interfaces with master_type='fem'
    """
    interfaces = []
    n_blocks = len(structure.list_blocks)

    # Track which FEM edges we've already processed to avoid duplicates
    processed_edges = set()

    for elem_id, element in enumerate(structure.list_fes):
        edges = get_fem_element_edges(element)

        for local_edge_id, (local_nodes, v0, v1) in enumerate(edges):
            # Create unique key for this edge (based on position)
            edge_key = (round(v0[0], 6), round(v0[1], 6), round(v1[0], 6), round(v1[1], 6))
            if edge_key in processed_edges:
                continue

            # Find block faces near this FEM edge
            nearby_block_faces = find_block_faces_near_segment(
                structure, v0, v1, tolerance
            )

            if nearby_block_faces:
                # Create interface with FEM as master
                interface = MortarInterface(
                    master_id=elem_id,
                    master_face_id=local_edge_id,
                    face_vertices=np.array([v0, v1]),
                    master_type='fem'
                )

                # Store the global node IDs for the master edge
                interface.master_global_nodes = [element.connect[n] for n in local_nodes]
                interface.master_local_nodes = local_nodes

                # Add block faces as "slave" faces
                for block_id, face_id, bv0, bv1 in nearby_block_faces:
                    interface.add_block_face(block_id, face_id, bv0, bv1)

                interfaces.append(interface)
                processed_edges.add(edge_key)

    # Filter by orientation if specified
    if interface_orientation is not None:
        filtered = []
        for interface in interfaces:
            nx, ny = interface.normal[0], interface.normal[1]
            if interface_orientation == 'horizontal' and abs(ny) > abs(nx):
                filtered.append(interface)
            elif interface_orientation == 'vertical' and abs(nx) > abs(ny):
                filtered.append(interface)
        interfaces = filtered

    return interfaces


def detect_mortar_interfaces(structure, tolerance: float = 1e-6,
                             interface_orientation: str = None) -> List[MortarInterface]:
    """
    Detect mortar interfaces in a hybrid structure.

    This algorithm identifies which block faces are near FEM elements
    and creates MortarInterface objects to represent the coupling regions.

    Algorithm:
    1. Build spatial indices (KD-trees) for fast queries
    2. Pre-compute all block faces
    3. For each face, find nearby FEM nodes and elements using spatial queries
    4. If nearby FEM entities found, create MortarInterface
    5. (Optional) Filter by interface orientation

    Parameters
    ----------
    structure : Hybrid
        Hybrid structure with blocks and FEM elements
    tolerance : float
        Distance tolerance for proximity detection
    interface_orientation : str, optional
        Filter interfaces by orientation:
        - 'horizontal': Keep only horizontal faces (|ny| > |nx|)
        - 'vertical': Keep only vertical faces (|nx| > |ny|)
        - None: Keep all detected interfaces (default)
        Useful for beam-on-column coupling where only horizontal interfaces
        should be coupled.

    Returns
    -------
    List[MortarInterface]
        Detected interfaces

    Notes
    -----
    - Tolerance should be larger than for nodal coupling (e.g., 1e-6 vs 1e-9)
    - Mortar doesn't require exact nodal coincidence
    - Interfaces are created even if only part of face is near FEM
    - For typical beam-on-column structures, use interface_orientation='horizontal'
      to avoid spurious vertical interface detection
    - Uses KD-tree spatial indexing for O(log n) queries instead of O(n)
    """
    interfaces = []
    n_blocks = len(structure.list_blocks)

    # =========================================================================
    # BUILD SPATIAL INDICES (one-time cost, enables O(log n) queries)
    # =========================================================================

    # Build KD-tree for FEM nodes
    fem_node_tree = None
    fem_nodes_array = None
    n_fem_nodes = len(structure.list_nodes) - n_blocks
    if n_fem_nodes > 0:
        fem_nodes_array = np.array(structure.list_nodes[n_blocks:])
        fem_node_tree = cKDTree(fem_nodes_array)

    # Build KD-tree for FEM element centroids
    element_centroid_tree = None
    element_centroids = None
    max_element_radius = None
    if structure.list_fes:
        # Compute centroids
        element_centroids = np.array([
            np.mean(fe.nodes, axis=0) for fe in structure.list_fes
        ])
        element_centroid_tree = cKDTree(element_centroids)

        # Compute max element radius (for search expansion)
        max_element_radius = 0.0
        for fe in structure.list_fes:
            nodes = np.array(fe.nodes)
            centroid = np.mean(nodes, axis=0)
            radii = np.linalg.norm(nodes - centroid, axis=1)
            max_element_radius = max(max_element_radius, np.max(radii))

    # Pre-compute all block faces (avoids recomputing in loop)
    all_block_faces = [get_block_faces(block) for block in structure.list_blocks]

    # =========================================================================
    # INTERFACE DETECTION (now O(log n) per query instead of O(n))
    # =========================================================================

    for block_id, faces in enumerate(all_block_faces):
        for face_id, (v0, v1) in enumerate(faces):
            # Find nearby FEM nodes (O(log n) with KD-tree)
            nearby_nodes = find_fem_nodes_near_segment(
                structure, v0, v1, tolerance, n_blocks,
                fem_node_tree=fem_node_tree,
                fem_nodes_array=fem_nodes_array
            )

            # Find nearby FEM elements (O(log n) with KD-tree)
            nearby_elements = find_fem_elements_near_segment(
                structure, v0, v1, tolerance,
                element_centroid_tree=element_centroid_tree,
                element_centroids=element_centroids,
                max_element_radius=max_element_radius
            )

            # Create interface if FEM entities found
            if nearby_nodes or nearby_elements:
                interface = MortarInterface(
                    master_id=block_id,
                    master_face_id=face_id,
                    face_vertices=np.array([v0, v1]),
                    master_type='block'
                )

                # Add FEM nodes to interface
                for node_id in nearby_nodes:
                    interface.add_fem_node(node_id)

                # Add FEM elements to interface
                for elem_id in nearby_elements:
                    interface.add_fem_element(elem_id)

                interfaces.append(interface)

    # Filter by orientation if specified
    if interface_orientation is not None:
        filtered_interfaces = []

        for interface in interfaces:
            nx, ny = interface.normal[0], interface.normal[1]

            if interface_orientation == 'horizontal':
                # Keep if |ny| > |nx| (normal points mostly up/down)
                if abs(ny) > abs(nx):
                    filtered_interfaces.append(interface)

            elif interface_orientation == 'vertical':
                # Keep if |nx| > |ny| (normal points mostly left/right)
                if abs(nx) > abs(ny):
                    filtered_interfaces.append(interface)

        interfaces = filtered_interfaces

    return interfaces


def validate_interfaces(interfaces: List[MortarInterface],
                        min_length: float = 1e-6) -> List[MortarInterface]:
    """
    Validate and filter detected interfaces.

    Removes interfaces that are too short or have no FEM entities.

    Parameters
    ----------
    interfaces : List[MortarInterface]
        Detected interfaces
    min_length : float
        Minimum interface length to keep

    Returns
    -------
    List[MortarInterface]
        Valid interfaces
    """
    valid_interfaces = []

    for interface in interfaces:
        # Check length
        if interface.interface_length < min_length:
            continue

        # Check FEM entities
        if not interface.fem_element_ids and not interface.fem_nodes_on_interface:
            continue

        # Interface is valid
        valid_interfaces.append(interface)

    return valid_interfaces


def detect_coincident_nodes(structure, tolerance: float = 1e-6) -> List[Tuple[int, int]]:
    """
    Detect pairs of geometrically coincident nodes in the structure.

    Useful for tying matching FEM meshes (Constraint/Lagrange coupling).

    Parameters
    ----------
    structure : Structure
        Structure object containing list_nodes
    tolerance : float
        Distance tolerance for coincidence

    Returns
    -------
    List[Tuple[int, int]]
        List of node pairs [(id1, id2), ...] that are coincident.
        id1 < id2 is guaranteed.
    """
    nodes = structure.list_nodes
    n_nodes = len(nodes)
    coincident_pairs = []

    # Optimization: Use spatial hashing/sorting if N is large.
    # For now, simple sorting by x-coordinate reduces search space.

    # Store nodes with their original IDs: [(x, y, original_id), ...]
    indexed_nodes = []
    for i, coords in enumerate(nodes):
        indexed_nodes.append((coords[0], coords[1], i))

    # Sort by X coordinate
    indexed_nodes.sort(key=lambda x: x[0])

    # Sweep line approach
    for i in range(n_nodes):
        x_i, y_i, id_i = indexed_nodes[i]

        # Check subsequent nodes
        for j in range(i + 1, n_nodes):
            x_j, y_j, id_j = indexed_nodes[j]

            # If x distance exceeds tolerance, we can stop checking this inner loop
            # because the list is sorted by X
            if (x_j - x_i) > tolerance:
                break

            # Check Y distance
            if abs(y_j - y_i) < tolerance:
                # Full distance check (optional, but good for robustness)
                dist = np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2)
                if dist < tolerance:
                    # Found a pair!
                    # Store with sorted IDs to avoid duplicates and ensure consistency
                    p1, p2 = min(id_i, id_j), max(id_i, id_j)
                    coincident_pairs.append((p1, p2))

    return coincident_pairs
