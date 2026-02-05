from typing import List, Tuple, Dict

import numpy as np

from .InterfaceDetection import point_to_segment_distance
from .MortarInterface import MortarInterface


def find_fem_boundary_edges(structure) -> Dict[int, List[Tuple[int, int]]]:
    """
    Identify boundary edges of the FEM mesh.

    Supports all 2D element types:
    - Triangle3 (3 nodes), Triangle6 (6 nodes)
    - Quad4 (4 nodes), Quad8 (8 nodes), Quad9 (9 nodes)

    Returns
    -------
    boundary_edges : Dict[edge_hash, List[elem_id, local_edge_id]]
        Dictionary mapping edge key (sorted corner node pair) to list of
        (element_id, local_edge_id) tuples. Boundary edges appear only once.
    """
    # Edge hash -> list of (element_id, local_edge_id)
    # Boundary edges appear only once.
    edge_occurrences = {}

    # Edge definitions for each element type
    # Format: list of tuples (corner1, corner2) - use corner nodes for edge identification
    triangle_edges = [(0, 1), (1, 2), (2, 0)]
    quad_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    for elem_id, element in enumerate(structure.list_fes):
        n_nodes = element.nd

        # Determine edge connectivity based on element type
        if n_nodes == 3:
            # Triangle3 (linear triangle)
            connectivity = triangle_edges
        elif n_nodes == 6:
            # Triangle6 (quadratic triangle) - use corner nodes only
            connectivity = triangle_edges
        elif n_nodes == 4:
            # Quad4 (linear quadrilateral)
            connectivity = quad_edges
        elif n_nodes == 8:
            # Quad8 (serendipity quadrilateral) - use corner nodes only
            connectivity = quad_edges
        elif n_nodes == 9:
            # Quad9 (biquadratic quadrilateral) - use corner nodes only
            connectivity = quad_edges
        else:
            continue  # Skip unknown element types

        for local_id, (n1_local, n2_local) in enumerate(connectivity):
            n1_global = element.connect[n1_local]
            n2_global = element.connect[n2_local]

            # Unique edge identifier (sorted node pair)
            edge_key = tuple(sorted((n1_global, n2_global)))

            if edge_key not in edge_occurrences:
                edge_occurrences[edge_key] = []
            edge_occurrences[edge_key].append((elem_id, local_id))

    return edge_occurrences


def detect_fem_fem_interfaces(structure, tolerance: float = 1e-4) -> List[MortarInterface]:
    """
    Detect interfaces between two distinct FEM domains (FEM-FEM Mortar).

    Supports all 2D element types:
    - Triangle3 (3 nodes), Triangle6 (6 nodes)
    - Quad4 (4 nodes), Quad8 (8 nodes), Quad9 (9 nodes)

    Strategy:
    1. Find all boundary edges of the FEM mesh(es).
    2. For each boundary edge (potential master), find close slave edges.
    3. Create MortarInterface for each master with slave edges found.

    Parameters
    ----------
    structure : Structure_FEM or Hybrid
        Structure containing FEM elements
    tolerance : float
        Distance tolerance for interface detection

    Returns
    -------
    interfaces : List[MortarInterface]
        Detected FEM-FEM interfaces
    """
    # Edge connectivity for different element types
    triangle_edges = [(0, 1), (1, 2), (2, 0)]
    quad_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    def get_edge_connectivity(n_nodes):
        """Get edge connectivity based on element node count."""
        if n_nodes in (3, 6):
            return triangle_edges
        elif n_nodes in (4, 8, 9):
            return quad_edges
        return None

    # 1. Find Boundary Edges
    edge_occurrences = find_fem_boundary_edges(structure)
    boundary_edges = [occ[0] for key, occ in edge_occurrences.items() if len(occ) == 1]

    interfaces = []

    # 2. Pair detection (Naive O(N^2) for now)
    # Each boundary edge is a candidate master. Find slave edges close to it.

    for i, (master_elem_id, master_local_edge) in enumerate(boundary_edges):
        master_elem = structure.list_fes[master_elem_id]
        n_nodes_master = master_elem.nd

        # Get Master Vertices
        connectivity_master = get_edge_connectivity(n_nodes_master)
        if connectivity_master is None:
            continue

        n1_local, n2_local = connectivity_master[master_local_edge]
        v0 = np.array(master_elem.nodes[n1_local])
        v1 = np.array(master_elem.nodes[n2_local])

        # Create potential interface
        interface = MortarInterface(
            master_id=master_elem_id,
            master_face_id=master_local_edge,
            face_vertices=np.array([v0, v1]),
            master_type='fem'
        )

        # Store master node info for later use
        interface.master_local_nodes = [n1_local, n2_local]
        interface.master_global_nodes = [master_elem.connect[n1_local], master_elem.connect[n2_local]]

        # Search for Slaves
        found_slaves = False

        for j, (slave_elem_id, slave_local_edge) in enumerate(boundary_edges):
            if i == j:
                continue  # Skip self
            if master_elem_id == slave_elem_id:
                continue  # Skip same element

            slave_elem = structure.list_fes[slave_elem_id]
            n_nodes_slave = slave_elem.nd

            # Get slave edge connectivity
            connectivity_slave = get_edge_connectivity(n_nodes_slave)
            if connectivity_slave is None:
                continue

            # Check slave edge vertices
            n1_sl, n2_sl = connectivity_slave[slave_local_edge]
            sv0 = np.array(slave_elem.nodes[n1_sl])
            sv1 = np.array(slave_elem.nodes[n2_sl])

            dist0, _ = point_to_segment_distance(sv0, v0, v1)
            dist1, _ = point_to_segment_distance(sv1, v0, v1)

            if dist0 < tolerance and dist1 < tolerance:
                # Candidate found!
                interface.add_fem_element(slave_elem_id)
                # Add nodes
                for node_idx in [slave_elem.connect[n1_sl], slave_elem.connect[n2_sl]]:
                    interface.add_fem_node(node_idx)
                found_slaves = True

        if found_slaves:
            interfaces.append(interface)

    return interfaces
