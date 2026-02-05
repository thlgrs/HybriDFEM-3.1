"""
High-level Model Builders
==========================

Functions to build complete Structure objects (Block, FEM, Hybrid) from configuration.
"""

from Core.Objects.ConstitutiveLaw.Contact import Contact
from Core.Objects.ConstitutiveLaw.Material import PlaneStress, Material

from Core.Objects.DFEM.Block import Block_2D
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Structures.Structure_Block import Structure_Block
from Core.Structures.Structure_FEM import Structure_FEM
from Core.Structures.Structure_Hybrid import Hybrid
from Examples.utils.mesh_generation import (
    generate_node_grid, create_triangle_elements, create_quad_elements,
    generate_block_grid, generate_block_grid_from_base,
    generate_block_slice, generate_block_slice_vertical
)


def setup_contact(St, kn, ks, lin_geom=True, nb_cps=20, contact_law=None):
    """Auto-detect interfaces and create contact faces."""
    St.detect_interfaces(eps=1e-9, margin=0.01)
    if contact_law is None:
        contact_law = Contact(k_n=kn, k_s=ks)
    St.make_cfs(lin_geom=lin_geom, nb_cps=nb_cps, contact=contact_law)
    return St


def find_nodes_at_length(St, target_x, tolerance=1e-9):
    """Find node indices at a specific y-coordinate."""
    return sorted([i for i, n in enumerate(St.list_nodes) if abs(n[0] - target_x) < tolerance])

def find_nodes_at_height(St, target_y, tolerance=1e-9):
    """Find node indices at a specific y-coordinate."""
    return sorted([i for i, n in enumerate(St.list_nodes) if abs(n[1] - target_y) < tolerance])


def find_nodes_at_base(St, tolerance=1e-9):
    """Find node indices at y=0."""
    return find_nodes_at_height(St, 0.0, tolerance)


def create_block_column(config):
    """Create a 2D grid of rigid blocks with contact."""
    g_conf = config['geometry']
    c_conf = config['contact']
    m_conf = config['material']

    print("\n" + "=" * 60)
    print("   BUILDING STRUCTURE: Rigid Block Grid")
    print("=" * 60)

    nx = g_conf['nx']
    ny = g_conf['ny']

    # ConstitutiveLaw
    mat = Material(E=m_conf['E'], nu=m_conf['nu'], rho=m_conf.get('rho', 2400))

    # Generate block vertices and reference points
    flat_vertices, flat_refs = generate_block_grid(
        nx=nx, ny=ny,
        total_width=g_conf['width'],
        total_height=g_conf['height'],
        ref_strategy='edge'
    )

    # Build structure
    St = Structure_Block()
    for verts, ref in zip(flat_vertices, flat_refs):
        St.add_block_from_vertices(
            vertices=verts,
            b=g_conf['thickness'],
            material=mat,
            ref_point=ref
        )

    St.make_nodes()
    print(f"[OK] Generated {len(St.list_nodes)} Nodes, {St.nb_dofs} DOFs")

    # Setup contact
    St = setup_contact(St, kn=c_conf['kn'], ks=c_conf['ks'], lin_geom=c_conf['LG'], nb_cps=c_conf['nb_cps'])
    print(f"[OK] Created {len(St.list_cfs)} Contact Faces")

    return St


def create_fem_column(config):
    """Create a pure FEM column structure."""
    g_conf, m_conf, e_conf = config['geometry'], config['material'], config['elements']

    print(f"\nBuilding FEM: {e_conf['type']} Order {e_conf['order']} ({g_conf['nx']}x{g_conf['ny']})")

    St = Structure_FEM(fixed_dofs_per_node=False)
    mat = PlaneStress(E=m_conf['E'], nu=m_conf['nu'], rho=m_conf.get('rho', 2400))
    geom = Geometry2D(t=g_conf['thickness'])

    # Dimensions
    width = g_conf.get('width', g_conf.get('x_dim', 1.0))  # Handle variations
    height = g_conf.get('height', g_conf.get('y_dim', 1.0))

    nodes_flat, nnx, nny = generate_node_grid(
        g_conf['nx'], g_conf['ny'], width, height, e_conf['order']
    )

    if e_conf['type'] == 'triangle':
        elements = create_triangle_elements(nodes_flat, nnx, g_conf['nx'], g_conf['ny'], e_conf['order'], mat, geom)
    else:
        elements = create_quad_elements(nodes_flat, nnx, g_conf['nx'], g_conf['ny'], e_conf['order'], mat, geom)

    for elem in elements:
        St.list_fes.append(elem)

    St.make_nodes()
    return St


def _create_fem_mesh_shifted(St, width, height, x_start, y_start, nx, ny, order, elem_type, mat, geom):
    """Helper to add FEM elements to a Hybrid structure with X and Y offset."""
    nodes_flat, nnx, nny = generate_node_grid(nx, ny, width, height, order)

    # Shift coordinates
    nodes_shifted = [[n[0] + x_start, n[1] + y_start] for n in nodes_flat]

    if elem_type == 'triangle':
        elements = create_triangle_elements(nodes_shifted, nnx, nx, ny, order, mat, geom)
    else:
        elements = create_quad_elements(nodes_shifted, nnx, nx, ny, order, mat, geom)

    for elem in elements:
        St.add_fe(elem)


def _create_fem_mesh_on_hybrid(St, width, y_start, height, nx, ny, order, elem_type, mat, geom):
    """Helper to add FEM elements to a Hybrid structure."""
    _create_fem_mesh_shifted(St, width, height, 0.0, y_start, nx, ny, order, elem_type, mat, geom)


def create_hybrid_column(config, fem_location='top'):
    """Create a hybrid column structure."""
    g, m, c, cp, e = config['geometry'], config['material'], config['contact'], config['coupling'], config['elements']
    print(f"\nBuilding Hybrid Column ({cp['method'].upper()}), FEM on {fem_location.upper()}")

    St = Hybrid()
    mat_block = Material(**m['block'])
    mat_fem = PlaneStress(**m['fem'])
    geom = Geometry2D(t=g['thickness'])

    offset = g.get('coupling_offset', 0.0)

    if fem_location == 'top':
        # Blocks at bottom
        verts, refs = generate_block_grid(
            g['nx'], g['ny_blocks'], g['width'], g['block_height'], cp['method']
        )
        for v, r in zip(verts, refs):
            St.add_block(Block_2D(vertices=v, b=g['thickness'], material=mat_block, ref_point=r))

        # FEM at top
        y_start = g['block_height'] + offset
        _create_fem_mesh_on_hybrid(
            St, g['width'], y_start, g['fem_height'],
            g['nx'], g['ny_fem'], e['order'], e['type'], mat_fem, geom
        )

    elif fem_location == 'bottom':
        # FEM at bottom
        _create_fem_mesh_on_hybrid(
            St, g['width'], 0.0, g['fem_height'],
            g['nx'], g['ny_fem'], e['order'], e['type'], mat_fem, geom
        )

        # Blocks at top
        y_start = g['fem_height'] + offset
        verts, refs = generate_block_grid_from_base(
            g['nx'], g['ny_blocks'], g['width'], g['block_height'], y_start, cp['method']
        )
        for v, r in zip(verts, refs):
            St.add_block(Block_2D(vertices=v, b=g['thickness'], material=mat_block, ref_point=r))

    St.make_nodes()

    # Contact
    setup_contact(St, c['kn'], c['ks'], c['LG'], c.get('nb_cps', 20))

    # Coupling
    _enable_coupling(St, cp)

    return St


def create_hybrid_beam_slices(config):
    """
    Create a beam with alternating Block and FEM vertical slices.

    For constraint/lagrange/penalty coupling, block slices adjacent to FEM slices
    use staggered interface layers so that block reference points align with FEM nodes.

    Configuration requires:
    'geometry': {
        'height': float,
        'thickness': float,
        'n_slices': int,
        'block_slice_width': float,
        'fem_slice_width': float,
        'ny': int,                      # Global vertical refinement
        'nx_block_slice': int,          # Default 2
        'nx_fem_slice': int,            # FEM refinement in x
        'coupling_offset': float,
        'start_with': 'block' or 'fem'
    }
    """
    g, m, c, cp, e = config['geometry'], config['material'], config['contact'], config['coupling'], config['elements']

    print(f"\n" + "=" * 60)
    print(f"   BUILDING HYBRID BEAM: {g['n_slices']} Slices")
    print(f"   Start: {g.get('start_with', 'block')}, Method: {cp['method']}")
    print("=" * 60)

    St = Hybrid()
    mat_block = Material(**m['block'])
    mat_fem = PlaneStress(**m['fem'])
    geom = Geometry2D(t=g['thickness'])

    current_x = 0.0
    offset = g.get('coupling_offset', 1e-6)
    ny = g['ny']
    height = g['height']

    n_slices = g['n_slices']
    start_with = g.get('start_with', 'block')

    # Determine if we need interface alignment (for non-mortar coupling)
    needs_interface = cp['method'] != 'mortar'

    # Pre-compute slice types for neighbor detection
    def is_block_slice(idx):
        if start_with == 'block':
            return idx % 2 == 0
        else:
            return idx % 2 != 0

    for i in range(n_slices):
        is_block = is_block_slice(i)

        if is_block:
            # Block Slice (vertical layer)
            width = g['block_slice_width']
            nx = g.get('nx_block_slice', 2)

            # Determine interface positions based on FEM neighbors
            has_fem_left = (i > 0) and not is_block_slice(i - 1)
            has_fem_right = (i < n_slices - 1) and not is_block_slice(i + 1)

            # Use interface layers only for non-mortar coupling
            interface_left = needs_interface and has_fem_left
            interface_right = needs_interface and has_fem_right

            # Generate blocks with proper interface alignment
            verts, refs = generate_block_slice_vertical(
                nx, ny, width, height,
                x_offset=current_x,
                interface_left=interface_left,
                interface_right=interface_right
            )

            # Add blocks (already positioned with x_offset)
            for v, r in zip(verts, refs):
                St.add_block(Block_2D(vertices=v, b=g['thickness'], material=mat_block, ref_point=r))

            current_x += width

        else:
            # FEM Slice (vertical layer)
            width = g['fem_slice_width']
            nx = g['nx_fem_slice']

            _create_fem_mesh_shifted(
                St, width, height, current_x, 0.0,
                nx, ny, e['order'], e['type'], mat_fem, geom
            )

            current_x += width

        if i < n_slices - 1:
            current_x += offset

    St.make_nodes()

    # Setup Contact
    setup_contact(St, c['kn'], c['ks'], c['LG'], c.get('nb_cps', 20))

    # Enable Coupling
    _enable_coupling(St, cp)

    return St


def _enable_coupling(St, cp_conf):
    method = cp_conf['method']
    if method == 'constraint':
        St.enable_block_fem_coupling(method='constraint', tolerance=cp_conf['tolerance'])
    elif method == 'penalty':
        St.enable_block_fem_coupling(method='penalty', penalty=cp_conf['penalty_stiffness'],
                                     tolerance=cp_conf['tolerance'])
    elif method == 'lagrange':
        St.enable_block_fem_coupling(method='lagrange', tolerance=cp_conf['tolerance'])
    elif method == 'mortar':
        St.enable_block_fem_coupling(method='mortar', integration_order=cp_conf['integration_order'],
                                     interface_tolerance=cp_conf['tolerance'],
                                     interface_orientation=cp_conf.get('interface_orientation', 'horizontal'))


def create_hybrid_column_slices(config):
    """
    Create a column with alternating Block and FEM horizontal slices stacked vertically.

    For constraint/lagrange/penalty coupling, block slices adjacent to FEM slices
    use staggered interface layers so that block reference points align with FEM nodes.

    Configuration requires:
    'geometry': {
        'width': float,                 # Column width
        'thickness': float,             # Out-of-plane thickness
        'n_slices': int,                # Number of alternating slices
        'block_slice_height': float,    # Height of block slices
        'fem_slice_height': float,      # Height of FEM slices
        'nx': int,                      # Global horizontal refinement
        'ny_block_slice': int,          # Blocks per slice vertically (default 2)
        'ny_fem_slice': int,            # FEM elements per slice vertically
        'coupling_offset': float,       # Gap between slices
        'start_with': 'block' or 'fem'  # First slice type at bottom
    }
    """
    g, m, c, cp, e = config['geometry'], config['material'], config['contact'], config['coupling'], config['elements']

    print(f"\n" + "=" * 60)
    print(f"   BUILDING HYBRID COLUMN: {g['n_slices']} Slices")
    print(f"   Start: {g.get('start_with', 'block')}, Method: {cp['method']}")
    print("=" * 60)

    St = Hybrid()
    mat_block = Material(**m['block'])
    mat_fem = PlaneStress(**m['fem'])
    geom = Geometry2D(t=g['thickness'])

    current_y = 0.0
    offset = g.get('coupling_offset', 1e-6)
    nx = g['nx']
    width = g['width']

    n_slices = g['n_slices']
    start_with = g.get('start_with', 'block')

    # Determine if we need interface alignment (for non-mortar coupling)
    needs_interface = cp['method'] != 'mortar'

    # Pre-compute slice types for neighbor detection
    def is_block_slice(idx):
        if start_with == 'block':
            return idx % 2 == 0
        else:
            return idx % 2 != 0

    for i in range(n_slices):
        is_block = is_block_slice(i)

        if is_block:
            # Block Slice (horizontal layer)
            height = g['block_slice_height']
            ny = g.get('ny_block_slice', 2)

            # Determine interface positions based on FEM neighbors
            has_fem_below = (i > 0) and not is_block_slice(i - 1)
            has_fem_above = (i < n_slices - 1) and not is_block_slice(i + 1)

            # Use interface layers only for non-mortar coupling
            interface_bottom = needs_interface and has_fem_below
            interface_top = needs_interface and has_fem_above

            # Generate blocks with proper interface alignment
            verts, refs = generate_block_slice(
                nx, ny, width, height,
                y_offset=current_y,
                interface_top=interface_top,
                interface_bottom=interface_bottom
            )

            # Add blocks (already positioned with y_offset)
            for v, r in zip(verts, refs):
                St.add_block(Block_2D(vertices=v, b=g['thickness'], material=mat_block, ref_point=r))

            current_y += height

        else:
            # FEM Slice (horizontal layer)
            height = g['fem_slice_height']
            ny = g['ny_fem_slice']

            _create_fem_mesh_shifted(
                St, width, height, 0.0, current_y,
                nx, ny, e['order'], e['type'], mat_fem, geom
            )

            current_y += height

        if i < n_slices - 1:
            current_y += offset

    St.make_nodes()

    # Setup Contact
    setup_contact(St, c['kn'], c['ks'], c['LG'], c.get('nb_cps', 20))

    # Enable Coupling
    _enable_coupling(St, cp)

    return St
