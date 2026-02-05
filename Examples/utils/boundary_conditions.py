"""
Boundary Condition Utilities
============================

Common boundary condition patterns for examples:
- Cantilever (fixed base)
- Compression (fixed base, top load)
- Shear (fixed base, lateral load)
- Simply supported

Centralizes node-finding and load application logic.
"""

from typing import List, Tuple, Optional, Set

import numpy as np

from Examples.utils.model_builders import find_nodes_at_base, find_nodes_at_height, find_nodes_at_length


def find_node_sets(St, config) -> dict:
    """
    Find common node sets based on geometry.

    Args:
        St: Structure object
        config: Configuration dictionary with geometry

    Returns:
        Dictionary with node sets: bottom, top, left, right, center_x, center_y
    """
    g = config.get('geometry', {})

    # Get dimensions from config
    width = g.get('width', g.get('x_dim', 1.0))
    height = g.get('height', g.get('y_dim', 1.0))

    # For sliced structures, calculate total height
    if 'n_slices' in g:
        n_slices = g['n_slices']
        start_block = (g.get('start_with', 'block') == 'block')

        if start_block:
            n_block = (n_slices + 1) // 2
            n_fem = n_slices // 2
        else:
            n_fem = (n_slices + 1) // 2
            n_block = n_slices // 2

        height = (n_block * g.get('block_slice_height', 0.1) +
                  n_fem * g.get('fem_slice_height', 0.1) +
                  (n_slices - 1) * g.get('coupling_offset', 0))

    # Tolerance for node finding
    tol = g.get('tolerance', 1e-3)
    center_tol = g.get('center_tolerance', 0.1)

    # Find sets
    return {
        'bottom': set(find_nodes_at_height(St, 0.0, tolerance=tol)),
        'top': set(find_nodes_at_height(St, height, tolerance=tol)),
        'left': set(find_nodes_at_length(St, 0.0, tolerance=tol)),
        'right': set(find_nodes_at_length(St, width, tolerance=tol)),
        'center_x': set(find_nodes_at_length(St, width / 2, tolerance=center_tol)),
        'center_y': set(find_nodes_at_height(St, height / 2, tolerance=center_tol)),
        'width': width,
        'height': height
    }


def get_dofs_per_node(St) -> int:
    """Determine DOFs per node from structure type."""
    has_blocks = hasattr(St, 'list_blocks') and St.list_blocks
    return 3 if has_blocks else 2


def apply_cantilever_bc(
    St, config,
    load_nodes: str = 'top',
    load_distribution: str = 'uniform'
) -> Tuple[any, int]:
    """
    Apply cantilever boundary conditions (fixed base, loaded top).

    Args:
        St: Structure object
        config: Configuration dictionary with loads section
        load_nodes: Where to apply load: 'top', 'top_center', 'top_corner'
        load_distribution: 'uniform' (divide equally) or 'point' (single node)

    Returns:
        (St, control_node)
    """
    node_sets = find_node_sets(St, config)
    l = config.get('loads', config.get('loading', {}))
    dof_count = get_dofs_per_node(St)

    # Fix bottom nodes
    bottom_nodes = list(node_sets['bottom'])
    for node in bottom_nodes:
        if dof_count == 3:
            St.fix_node(node_ids=[node], dofs=[0, 1, 2])
        else:
            St.fix_node(node_ids=[node], dofs=[0, 1])

    print(f"  -> Fixed base: {len(bottom_nodes)} nodes")

    # Determine load nodes
    if load_nodes == 'top_center':
        nodes_to_load = list(node_sets['top'].intersection(node_sets['center_x']))
    elif load_nodes == 'top_corner':
        nodes_to_load = list(node_sets['top'].intersection(node_sets['right']))
    else:  # 'top'
        nodes_to_load = list(node_sets['top'])

    if not nodes_to_load:
        nodes_to_load = list(node_sets['top'])

    # Determine control node (middle of top)
    top_sorted = sorted(node_sets['top'], key=lambda i: St.list_nodes[i][0])
    control_node = top_sorted[len(top_sorted) // 2]

    # Apply loads
    Fx = l.get('Fx', 0)
    Fy = l.get('Fy', 0)
    Mz = l.get('Mz', 0)

    n_load = len(nodes_to_load)

    if load_distribution == 'point':
        n_load = 1
        nodes_to_load = [control_node]

    if Fx != 0:
        St.load_node(node_ids=nodes_to_load, dofs=[0], force=Fx / n_load)
        print(f"  -> Applied Fx = {Fx/1e3:.1f} kN on {n_load} node(s)")

    if Fy != 0:
        St.load_node(node_ids=nodes_to_load, dofs=[1], force=Fy / n_load)
        print(f"  -> Applied Fy = {Fy/1e3:.1f} kN on {n_load} node(s)")

    if Mz != 0 and dof_count == 3:
        St.load_node(node_ids=nodes_to_load, dofs=[2], force=Mz / n_load)
        print(f"  -> Applied Mz = {Mz/1e3:.1f} kNm on {n_load} node(s)")

    return St, control_node


def apply_compression_bc(St, config) -> Tuple[any, int]:
    """
    Apply compression boundary conditions (fixed base, vertical load at top center).

    Args:
        St: Structure object
        config: Configuration dictionary with loads section

    Returns:
        (St, control_node)
    """
    node_sets = find_node_sets(St, config)
    l = config.get('loads', config.get('loading', {}))
    dof_count = get_dofs_per_node(St)

    # Fix bottom nodes fully
    bottom_nodes = list(node_sets['bottom'])
    for node in bottom_nodes:
        if dof_count == 3:
            St.fix_node(node_ids=[node], dofs=[0, 1, 2])
        else:
            St.fix_node(node_ids=[node], dofs=[0, 1])

    print(f"  -> Fixed base: {len(bottom_nodes)} nodes")

    # Load at top center
    nodes_to_load = list(node_sets['top'].intersection(node_sets['center_x']))
    if not nodes_to_load:
        # Fallback to center-most top node
        top_sorted = sorted(node_sets['top'], key=lambda i: St.list_nodes[i][0])
        nodes_to_load = [top_sorted[len(top_sorted) // 2]]

    control_node = nodes_to_load[0]

    # Apply vertical load
    Fy = l.get('Fy', 0)
    n_load = len(nodes_to_load)

    if Fy != 0:
        St.load_node(node_ids=nodes_to_load, dofs=[1], force=Fy / n_load)
        print(f"  -> Applied Fy = {Fy/1e3:.1f} kN on {n_load} node(s)")

    return St, control_node


def apply_shear_bc(St, config) -> Tuple[any, int]:
    """
    Apply pure shear boundary conditions (fixed base, horizontal load at top-left).

    Args:
        St: Structure object
        config: Configuration dictionary with loads section

    Returns:
        (St, control_node)
    """
    node_sets = find_node_sets(St, config)
    l = config.get('loads', config.get('loading', {}))
    dof_count = get_dofs_per_node(St)

    # Fix bottom nodes fully
    bottom_nodes = list(node_sets['bottom'])
    for node in bottom_nodes:
        if dof_count == 3:
            St.fix_node(node_ids=[node], dofs=[0, 1, 2])
        else:
            St.fix_node(node_ids=[node], dofs=[0, 1])

    print(f"  -> Fixed base: {len(bottom_nodes)} nodes")

    # Load at top-left corner
    l_set = find_nodes_at_length(St, 0.05, 0.1)
    nodes_to_load = list(node_sets['top'])  # .intersection(l_set))
    if not nodes_to_load:
        nodes_to_load = [min(node_sets['top'], key=lambda i: St.list_nodes[i][0])]

    # Control at top-right for shear deformation measurement
    control_candidates = list(node_sets['top'].intersection(node_sets['right']))
    if control_candidates:
        control_node = control_candidates[0]
    else:
        control_node = max(node_sets['top'], key=lambda i: St.list_nodes[i][0])

    # Apply horizontal load
    Fx = l.get('Fx', 0)
    n_load = len(nodes_to_load)

    if Fx != 0:
        St.load_node(node_ids=nodes_to_load, dofs=[0], force=Fx / n_load)
        print(f"  -> Applied Fx = {Fx/1e3:.1f} kN on {n_load} node(s)")

    return St, control_node


def apply_tip_load_bc(St, config) -> Tuple[any, int]:
    """
    Apply cantilever beam tip load (fixed left, load at top-right).

    Args:
        St: Structure object
        config: Configuration dictionary with loads section

    Returns:
        (St, control_node)
    """
    g = config.get('geometry', {})
    l = config.get('loads', config.get('loading', {}))

    # Get dimensions
    width = g.get('width', g.get('x_dim', 1.0))
    height = g.get('height', g.get('y_dim', 1.0))

    coords = np.array(St.list_nodes)
    tol = 1e-6

    # Fix left edge (x=0)
    left_indices = np.where(np.abs(coords[:, 0]) < tol)[0]
    St.fix_node(left_indices.tolist(), [0, 1])
    print(f"  -> Fixed left edge: {len(left_indices)} nodes")

    # Find tip node (x=width, y=height)
    tip_indices = np.where(
        (np.abs(coords[:, 0] - width) < tol) &
        (np.abs(coords[:, 1] - height) < tol)
    )[0]

    if len(tip_indices) > 0:
        control_node = int(tip_indices[0])
    else:
        # Fallback: rightmost top node
        top_indices = np.where(np.abs(coords[:, 1] - height) < tol)[0]
        control_node = int(top_indices[np.argmax(coords[top_indices, 0])])

    # Apply load at tip
    Fy = l.get('Fy', 0)
    if Fy != 0:
        St.load_node(int(control_node), 1, Fy)
        print(f"  -> Applied Fy = {Fy/1e3:.1f} kN at Node {control_node}")

    return St, [control_node]


def get_bc_function(bc_type: str):
    """
    Get the appropriate BC function for a given type.

    Args:
        bc_type: 'cantilever', 'compression', 'shear', 'tip_load'

    Returns:
        BC function
    """
    bc_map = {
        'cantilever': apply_cantilever_bc,
        'compression': apply_compression_bc,
        'shear': apply_shear_bc,
        'tip_load': apply_tip_load_bc,
    }

    if bc_type not in bc_map:
        raise ValueError(f"Unknown BC type: {bc_type}. Available: {list(bc_map.keys())}")

    return bc_map[bc_type]


def apply_conditions_from_config(St, config) -> Tuple[any, int]:
    """
    Auto-apply boundary conditions based on config.

    Reads bc.type from config and applies appropriate function.
    Falls back to cantilever if not specified.

    Args:
        St: Structure object
        config: Configuration dictionary

    Returns:
        (St, control_node)
    """
    bc = config.get('bc', {})
    bc_type = bc.get('type', 'cantilever')

    # Handle load 'name' field as BC type (backwards compatibility)
    l = config.get('loads', config.get('loading', {}))
    if 'name' in l:
        bc_type = l['name']

    if bc_type in ['', None]:
        bc_type = 'cantilever'

    print(f"\nApplying Boundary Conditions: {bc_type.upper()}")

    bc_func = get_bc_function(bc_type)
    return bc_func(St, config)
