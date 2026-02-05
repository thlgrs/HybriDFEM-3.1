"""
Low-level Mesh Generation Utilities
====================================

Optimized functions for generating node grids, block geometry, and FEM elements.

Performance notes:
- Nodes are kept as NumPy arrays for fast indexing
- Connectivity is computed vectorially with NumPy
- Element creation is the only sequential part (due to object instantiation)
"""
import numpy as np

from Core.Objects.FEM.Quads import Quad4, Quad8, Quad9
from Core.Objects.FEM.Triangles import Triangle3, Triangle6

# Element type mapping
ELEMENT_MAP = {
    ('triangle', 1): Triangle3,
    ('triangle', 2): Triangle6,
    ('quad', 1): Quad4,
    ('quad', 2): Quad8,
    ('quad', 3): Quad9,
}


def generate_node_grid(nx, ny, length, height, order):
    """
    Generate node grid using vectorized NumPy operations.

    Returns
    -------
    nodes : np.ndarray
        Shape (n_nodes, 2) array of [x, y] coordinates
    nnx : int
        Number of nodes in x direction
    nny : int
        Number of nodes in y direction
    """
    mul = 2 if order > 1 else 1
    nnx = nx * mul + 1
    nny = ny * mul + 1

    x = np.linspace(0, length, nnx)
    y = np.linspace(0, height, nny)
    xv, yv = np.meshgrid(x, y)

    # Keep as NumPy array for fast indexing (no .tolist())
    nodes = np.column_stack([xv.ravel(), yv.ravel()])
    return nodes, nnx, nny


def _compute_tri_connectivity_order1(nx, ny, nnx):
    """
    Vectorized computation of T3 triangle connectivity.

    Returns array of shape (2*nx*ny, 3) with node indices.
    """
    # Create grid of cell indices
    i_idx = np.arange(nx)
    j_idx = np.arange(ny)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()

    # Base node index for each cell
    base = ii + jj * nnx

    # Triangle 1: (0, 1, 1+nnx) -> lower-right
    # Triangle 2: (0, 1+nnx, nnx) -> upper-left
    tri1 = np.column_stack([base, base + 1, base + 1 + nnx])
    tri2 = np.column_stack([base, base + 1 + nnx, base + nnx])

    # Interleave: [tri1_0, tri2_0, tri1_1, tri2_1, ...]
    n_cells = nx * ny
    connectivity = np.empty((2 * n_cells, 3), dtype=np.int64)
    connectivity[0::2] = tri1
    connectivity[1::2] = tri2

    return connectivity


def _compute_tri_connectivity_order2(nx, ny, nnx):
    """
    Vectorized computation of T6 triangle connectivity.

    Returns array of shape (2*nx*ny, 6) with node indices.
    """
    # Create grid of cell indices
    i_idx = np.arange(nx)
    j_idx = np.arange(ny)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()

    # Base node index (step=2 for quadratic)
    bi = ii * 2
    bj = jj * 2
    base = bi + bj * nnx

    # Corner nodes
    c0 = base
    c1 = base + 2
    c2 = base + 2 + 2 * nnx
    c3 = base + 2 * nnx

    # Mid-edge nodes
    m01 = base + 1
    m12 = base + 2 + nnx
    m20 = base + 1 + nnx  # diagonal midpoint
    m23 = base + 1 + 2 * nnx
    m30 = base + nnx

    # Triangle 1: c0, c1, c2, m01, m12, m20
    # Triangle 2: c0, c2, c3, m20, m23, m30
    tri1 = np.column_stack([c0, c1, c2, m01, m12, m20])
    tri2 = np.column_stack([c0, c2, c3, m20, m23, m30])

    n_cells = nx * ny
    connectivity = np.empty((2 * n_cells, 6), dtype=np.int64)
    connectivity[0::2] = tri1
    connectivity[1::2] = tri2

    return connectivity


def _compute_quad_connectivity_order1(nx, ny, nnx):
    """Vectorized Q4 connectivity."""
    i_idx = np.arange(nx)
    j_idx = np.arange(ny)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()

    base = ii + jj * nnx

    # Q4: counter-clockwise from bottom-left
    connectivity = np.column_stack([
        base,
        base + 1,
        base + 1 + nnx,
        base + nnx
    ])
    return connectivity


def _compute_quad_connectivity_order2(nx, ny, nnx):
    """Vectorized Q8 connectivity."""
    i_idx = np.arange(nx)
    j_idx = np.arange(ny)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()

    bi = ii * 2
    bj = jj * 2
    base = bi + bj * nnx

    # Q8: 4 corners + 4 mid-edges
    connectivity = np.column_stack([
        base,  # 0: bottom-left
        base + 2,  # 1: bottom-right
        base + 2 + 2 * nnx,  # 2: top-right
        base + 2 * nnx,  # 3: top-left
        base + 1,  # 4: mid-bottom
        base + 2 + nnx,  # 5: mid-right
        base + 1 + 2 * nnx,  # 6: mid-top
        base + nnx  # 7: mid-left
    ])
    return connectivity


def _compute_quad_connectivity_order3(nx, ny, nnx):
    """Vectorized Q9 connectivity (Q8 + center node)."""
    i_idx = np.arange(nx)
    j_idx = np.arange(ny)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
    ii = ii.ravel()
    jj = jj.ravel()

    bi = ii * 2
    bj = jj * 2
    base = bi + bj * nnx

    # Q9: 4 corners + 4 mid-edges + center
    connectivity = np.column_stack([
        base,  # 0: bottom-left
        base + 2,  # 1: bottom-right
        base + 2 + 2 * nnx,  # 2: top-right
        base + 2 * nnx,  # 3: top-left
        base + 1,  # 4: mid-bottom
        base + 2 + nnx,  # 5: mid-right
        base + 1 + 2 * nnx,  # 6: mid-top
        base + nnx,  # 7: mid-left
        base + 1 + nnx  # 8: center
    ])
    return connectivity


def create_triangle_elements(nodes, width_nodes, nx, ny, order, mat, geom):
    """
    Create triangular elements with vectorized connectivity computation.

    Parameters
    ----------
    nodes : np.ndarray or list
        Node coordinates, shape (n_nodes, 2)
    width_nodes : int
        Number of nodes in x direction (nnx)
    nx, ny : int
        Number of elements in each direction
    order : int
        1 for T3, 2 for T6
    mat, geom : object
        Material and geometry objects

    Returns
    -------
    list
        List of Triangle element objects
    """
    ElementClass = ELEMENT_MAP[('triangle', order)]

    # Ensure nodes is array for fast indexing
    if isinstance(nodes, list):
        nodes = np.array(nodes)

    # Compute all connectivity at once
    if order == 1:
        conn = _compute_tri_connectivity_order1(nx, ny, width_nodes)
    else:
        conn = _compute_tri_connectivity_order2(nx, ny, width_nodes)

    # Create elements (this loop is unavoidable due to object creation)
    n_elements = len(conn)
    elements = [None] * n_elements  # Pre-allocate

    for idx in range(n_elements):
        node_indices = conn[idx]
        element_nodes = [nodes[i].tolist() for i in node_indices]
        elements[idx] = ElementClass(nodes=element_nodes, mat=mat, geom=geom)

    return elements


def create_quad_elements(nodes, width_nodes, nx, ny, order, mat, geom):
    """
    Create quadrilateral elements with vectorized connectivity computation.

    Parameters
    ----------
    nodes : np.ndarray or list
        Node coordinates, shape (n_nodes, 2)
    width_nodes : int
        Number of nodes in x direction (nnx)
    nx, ny : int
        Number of elements in each direction
    order : int
        1 for Q4, 2 for Q8, 3 for Q9
    mat, geom : object
        Material and geometry objects

    Returns
    -------
    list
        List of Quad element objects
    """
    ElementClass = ELEMENT_MAP[('quad', order)]

    # Ensure nodes is array for fast indexing
    if isinstance(nodes, list):
        nodes = np.array(nodes)

    # Compute all connectivity at once
    if order == 1:
        conn = _compute_quad_connectivity_order1(nx, ny, width_nodes)
    elif order == 2:
        conn = _compute_quad_connectivity_order2(nx, ny, width_nodes)
    else:  # order == 3 (Q9)
        conn = _compute_quad_connectivity_order3(nx, ny, width_nodes)

    # Create elements
    n_elements = len(conn)
    elements = [None] * n_elements  # Pre-allocate

    for idx in range(n_elements):
        node_indices = conn[idx]
        element_nodes = [nodes[i].tolist() for i in node_indices]
        elements[idx] = ElementClass(nodes=element_nodes, mat=mat, geom=geom)

    return elements


def generate_block_slice(nx, ny, total_width, total_height, y_offset=0.0,
                         interface_top=False, interface_bottom=False, ref_strategy='edge'):
    """
    Generate block grid for a horizontal slice with optional interface layers.

    For constraint/lagrange/penalty coupling, interface layers use staggered blocks
    (nx+1 blocks instead of nx) so that block reference points align with FEM nodes.

    Args:
        nx: Number of blocks in x direction
        ny: Number of block rows in the slice
        total_width: Total width of the slice
        total_height: Total height of the slice
        y_offset: Vertical offset for the slice
        interface_top: If True, top row uses staggered interface blocks
        interface_bottom: If True, bottom row uses staggered interface blocks
        ref_strategy: Reference point strategy ('edge' or 'center')

    Returns:
        flat_vertices: List of vertex arrays for each block
        flat_refs: List of reference point arrays for each block
    """
    flat_vertices = []
    flat_refs = []
    block_width = total_width / nx
    block_height = total_height / ny

    def add_regular_row(j, is_bottom_edge, is_top_edge):
        """Add a regular row of nx blocks."""
        y_bot = y_offset + j * block_height
        y_top = y_offset + (j + 1) * block_height

        for i in range(nx):
            x_left = i * block_width
            x_right = (i + 1) * block_width
            vertices = np.array([[x_left, y_bot], [x_right, y_bot], [x_right, y_top], [x_left, y_top]])

            ref_x = (x_left + x_right) / 2
            ref_y = (y_bot + y_top) / 2

            if ref_strategy == 'edge':
                if is_bottom_edge:
                    ref_y = y_bot
                elif is_top_edge:
                    ref_y = y_top
                if i == 0:
                    ref_x = x_left
                elif i == nx - 1:
                    ref_x = x_right

            flat_vertices.append(vertices)
            flat_refs.append(np.array([ref_x, ref_y]))

    def add_interface_row(j, is_top_interface):
        """Add an interface row of nx+1 staggered blocks."""
        y_bot = y_offset + j * block_height
        y_top = y_offset + (j + 1) * block_height

        for k in range(nx + 1):
            node_x = k * block_width
            if k == 0:
                x_start, x_end = 0.0, block_width / 2.0
            elif k == nx:
                x_start, x_end = total_width - block_width / 2.0, total_width
            else:
                x_start, x_end = node_x - block_width / 2.0, node_x + block_width / 2.0

            vertices = np.array([[x_start, y_bot], [x_end, y_bot], [x_end, y_top], [x_start, y_top]])

            ref_x = (x_start + x_end) / 2.0
            ref_y = y_top if is_top_interface else y_bot

            if ref_strategy == 'edge':
                if k == 0:
                    ref_x = x_start
                elif k == nx:
                    ref_x = x_end

            flat_vertices.append(vertices)
            flat_refs.append(np.array([ref_x, ref_y]))

    # Generate rows
    for j in range(ny):
        is_bottom_row = (j == 0)
        is_top_row = (j == ny - 1)

        if is_bottom_row and interface_bottom:
            add_interface_row(j, is_top_interface=False)
        elif is_top_row and interface_top:
            add_interface_row(j, is_top_interface=True)
        else:
            # Regular row - check if edges should be at boundaries
            is_bottom_edge = is_bottom_row and not interface_bottom
            is_top_edge = is_top_row and not interface_top
            add_regular_row(j, is_bottom_edge, is_top_edge)

    return flat_vertices, flat_refs


def generate_block_slice_vertical(nx, ny, total_width, total_height, x_offset=0.0,
                                  interface_left=False, interface_right=False, ref_strategy='edge'):
    """
    Generate block grid for a vertical slice with optional interface layers on left/right.

    For constraint/lagrange/penalty coupling, interface layers use staggered blocks
    (ny+1 blocks instead of ny) so that block reference points align with FEM nodes.

    Args:
        nx: Number of block columns in the slice
        ny: Number of blocks in y direction
        total_width: Total width of the slice
        total_height: Total height of the slice
        x_offset: Horizontal offset for the slice
        interface_left: If True, left column uses staggered interface blocks
        interface_right: If True, right column uses staggered interface blocks
        ref_strategy: Reference point strategy ('edge' or 'center')

    Returns:
        flat_vertices: List of vertex arrays for each block
        flat_refs: List of reference point arrays for each block
    """
    flat_vertices = []
    flat_refs = []
    block_width = total_width / nx
    block_height = total_height / ny

    def add_regular_column(i, is_left_edge, is_right_edge):
        """Add a regular column of ny blocks."""
        x_left = x_offset + i * block_width
        x_right = x_offset + (i + 1) * block_width

        for j in range(ny):
            y_bot = j * block_height
            y_top = (j + 1) * block_height
            vertices = np.array([[x_left, y_bot], [x_right, y_bot], [x_right, y_top], [x_left, y_top]])

            ref_x = (x_left + x_right) / 2
            ref_y = (y_bot + y_top) / 2

            if ref_strategy == 'edge':
                if is_left_edge:
                    ref_x = x_left
                elif is_right_edge:
                    ref_x = x_right
                if j == 0:
                    ref_y = y_bot
                elif j == ny - 1:
                    ref_y = y_top

            flat_vertices.append(vertices)
            flat_refs.append(np.array([ref_x, ref_y]))

    def add_interface_column(i, is_left_interface):
        """Add an interface column of ny+1 staggered blocks."""
        x_left = x_offset + i * block_width
        x_right = x_offset + (i + 1) * block_width

        for k in range(ny + 1):
            node_y = k * block_height
            if k == 0:
                y_start, y_end = 0.0, block_height / 2.0
            elif k == ny:
                y_start, y_end = total_height - block_height / 2.0, total_height
            else:
                y_start, y_end = node_y - block_height / 2.0, node_y + block_height / 2.0

            vertices = np.array([[x_left, y_start], [x_right, y_start], [x_right, y_end], [x_left, y_end]])

            ref_x = x_left if is_left_interface else x_right
            ref_y = (y_start + y_end) / 2.0

            if ref_strategy == 'edge':
                if k == 0:
                    ref_y = y_start
                elif k == ny:
                    ref_y = y_end

            flat_vertices.append(vertices)
            flat_refs.append(np.array([ref_x, ref_y]))

    # Generate columns
    for i in range(nx):
        is_left_col = (i == 0)
        is_right_col = (i == nx - 1)

        if is_left_col and interface_left:
            add_interface_column(i, is_left_interface=True)
        elif is_right_col and interface_right:
            add_interface_column(i, is_left_interface=False)
        else:
            # Regular column - check if edges should be at boundaries
            is_left_edge = is_left_col and not interface_left
            is_right_edge = is_right_col and not interface_right
            add_regular_column(i, is_left_edge, is_right_edge)

    return flat_vertices, flat_refs


def generate_block_grid(nx, ny, total_width, total_height, coupling_method='mortar', ref_strategy='edge'):
    """Generate block grid vertices and refs (interface at top if not mortar)."""
    flat_vertices = []
    flat_refs = []
    block_width = total_width / nx
    block_height = total_height / ny

    if coupling_method == 'mortar':
        rows = range(ny)
        interface_layer = False
    else:
        rows = range(ny - 1)
        interface_layer = True

    # Regular blocks
    for j in rows:
        for i in range(nx):
            x_left = i * block_width
            x_right = (i + 1) * block_width
            y_bot = j * block_height
            y_top = (j + 1) * block_height
            vertices = np.array([[x_left, y_bot], [x_right, y_bot], [x_right, y_top], [x_left, y_top]])

            ref_x = (x_left + x_right) / 2
            ref_y = (y_bot + y_top) / 2

            if ref_strategy == 'edge':
                if j == 0:
                    ref_y = y_bot
                elif j == ny - 1 and not interface_layer:
                    ref_y = y_top

                if i == 0:
                    ref_x = x_left
                elif i == nx - 1:
                    ref_x = x_right

            flat_vertices.append(vertices)
            flat_refs.append(np.array([ref_x, ref_y]))

    # Interface layer (staggered)
    if interface_layer:
        j = ny - 1
        y_bot = j * block_height
        y_top = (j + 1) * block_height
        for k in range(nx + 1):
            node_x = k * block_width
            if k == 0:
                x_start, x_end = 0.0, block_width / 2.0
            elif k == nx:
                x_start, x_end = total_width - block_width / 2.0, total_width
            else:
                x_start, x_end = node_x - block_width / 2.0, node_x + block_width / 2.0

            vertices = np.array([[x_start, y_bot], [x_end, y_bot], [x_end, y_top], [x_start, y_top]])

            ref_x = (x_start + x_end) / 2.0
            ref_y = (y_bot + y_top) / 2.0

            if ref_strategy == 'edge':
                ref_y = y_top
                if k == 0:
                    ref_x = x_start
                elif k == nx:
                    ref_x = x_end

            flat_vertices.append(vertices)
            flat_refs.append(np.array([ref_x, ref_y]))

    return flat_vertices, flat_refs


def generate_block_grid_from_base(nx, ny, total_width, total_height, y_start, coupling_method='mortar',
                                  ref_strategy='edge'):
    """Generate block grid starting from y_start with interface at bottom."""
    flat_vertices = []
    flat_refs = []
    block_width = total_width / nx
    block_height = total_height / ny

    if coupling_method == 'mortar':
        rows = range(ny)
        interface_layer = False
        start_row = 0
    else:
        rows = range(1, ny)
        interface_layer = True
        start_row = 1

    # Interface layer at BOTTOM
    if interface_layer:
        j = 0
        y_bot = y_start
        y_top = y_start + block_height
        for k in range(nx + 1):
            node_x = k * block_width
            if k == 0:
                x_start, x_end = 0.0, block_width / 2.0
            elif k == nx:
                x_start, x_end = total_width - block_width / 2.0, total_width
            else:
                x_start, x_end = node_x - block_width / 2.0, node_x + block_width / 2.0

            vertices = np.array([[x_start, y_bot], [x_end, y_bot], [x_end, y_top], [x_start, y_top]])

            ref_x = (x_start + x_end) / 2.0
            ref_y = (y_bot + y_top) / 2.0

            if ref_strategy == 'edge':
                ref_y = y_bot
                if k == 0:
                    ref_x = x_start
                elif k == nx:
                    ref_x = x_end

            flat_vertices.append(vertices)
            flat_refs.append(np.array([ref_x, ref_y]))

    # Regular blocks
    for j in rows:
        for i in range(nx):
            x_left = i * block_width
            x_right = (i + 1) * block_width
            y_bot = y_start + j * block_height
            y_top = y_start + (j + 1) * block_height
            vertices = np.array([[x_left, y_bot], [x_right, y_bot], [x_right, y_top], [x_left, y_top]])

            ref_x = (x_left + x_right) / 2
            ref_y = (y_bot + y_top) / 2

            if ref_strategy == 'edge':
                if j == 0 and not interface_layer:
                    ref_y = y_bot
                if j == ny - 1:
                    ref_y = y_top

                if i == 0:
                    ref_x = x_left
                elif i == nx - 1:
                    ref_x = x_right

            flat_vertices.append(vertices)
            flat_refs.append(np.array([ref_x, ref_y]))

    return flat_vertices, flat_refs
