"""
FEM Square Example - Gmsh Mesh Generation
==========================================

Pure FEM square plate using Gmsh for automatic mesh generation.

Combines:
- Square geometry from fem_square.py (1m x 1m)
- Gmsh mesh generation from cantilever_gmsh.py
- Compression and shear boundary conditions

Configuration:
- Geometry: 1m x 1m x 0.25m square plate
- Material: E = 10 GPa, nu = 0.0
- Loading: Compression (Fy = -1000 kN) or Shear (Fx = 1000 kN)
- BC: Fixed bottom edge
"""

import sys
from pathlib import Path

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Library Imports ---
from Core.Objects.FEM.Mesh import Mesh
from Core.Structures.Structure_FEM import Structure_FEM
from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
from Examples.utils import find_nodes_at_length
from Examples.utils.helpers import create_config
from Examples.utils.boundary_conditions import find_node_sets
from Examples.utils.runner import run_example, run_examples

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_ROOT = str(project_root / "Examples" / 'Results' / 'FEM' / 'square_gmsh')

BASE_CONFIG = {
    'geometry': {
        'width': 1.0,
        'height': 1.0,
        'thickness': 0.25,
    },
    'elements': {
        'size': 0.01,  # Element size for Gmsh (m)
        'type': 'triangle',
        'order': 1,
    },
    'material': {
        'E': 10e9,
        'nu': 0.0,
        'rho': 0,
    },
    'loads': {
        'name': 'compression',
        'Fx': 0,
        'Fy': -1e6,
    },
    'solver': {
        'name': 'linear',
    },
    'io': {
        'filename': 'fem_square_gmsh',
        'dir': RESULTS_ROOT,
        'show_nodes': False,
        'scale': 50.0,
        'figsize': (4, 4),
    }
}

# =============================================================================
# LOAD DEFINITIONS
# =============================================================================

COMPRESSION = {
    'name': 'compression',
    'Fx': 0,
    'Fy': -1e6,
}

SHEAR = {
    'name': 'shear',
    'Fx': 1e6,
    'Fy': 0,
}


# =============================================================================
# MODEL CREATION
# =============================================================================

def generate_fem_mesh_gmsh(config):
    """Generate square mesh using Gmsh."""
    g = config['geometry']
    e = config['elements']

    width = g['width']
    height = g['height']

    # Define square corners (counter-clockwise)
    points = [
        (0, 0),  # Bottom-left
        (width, 0),  # Bottom-right
        (width, height),  # Top-right
        (0, height),  # Top-left
    ]

    # Define edge groups for boundary conditions
    edge_groups = {
        "bottom": [0],  # Fixed edge
        "right": [1],
        "top": [2],  # Load edge
        "left": [3],
    }

    mesh = Mesh(
        points=points,
        element_type=e['type'],
        element_size=e['size'],
        order=e['order'],
        name="square_mesh",
        edge_groups=edge_groups,
    )

    mesh.generate_mesh()
    return mesh


def create_model_gmsh(config):
    """Create FEM structure from Gmsh mesh."""
    g = config['geometry']
    m = config['material']
    e = config['elements']

    print("  Generating FEM mesh with Gmsh...")
    mesh = generate_fem_mesh_gmsh(config)

    mat_fem = PlaneStress(**m)
    geom = Geometry2D(t=g['thickness'])

    # Check if user wants Quad9 elements (9-node Lagrangian)
    prefer_quad9 = e.get('prefer_quad9', False)

    St = Structure_FEM.from_mesh(mesh, mat_fem, geom, prefer_quad9=prefer_quad9)
    St.make_nodes()

    print(f"  -> Created mesh: {len(St.list_fes)} elements, {len(St.list_nodes)} nodes")
    return St


def apply_conditions(St, config):
    """Apply BCs for hybrid sliced structure based on bc.type."""
    node_sets = find_node_sets(St, config)
    l = config.get('loads', {})
    bc_type = l.get('name', 'compression')

    print(f"\nApplying Conditions: {bc_type.upper()}")
    print(f"  Total Height ~ {node_sets['height']:.4f} m")

    # Fix bottom nodes (blocks have 3 DOFs, FEM has 2)
    bottom_nodes = list(node_sets['bottom'])
    for node in bottom_nodes:
        dofs = St.get_dofs_from_node(node)
        St.fix_node(node_ids=[node], dofs=list(range(len(dofs))))
    print(f"  -> Fixed base: {len(bottom_nodes)} nodes")
    l_set = find_nodes_at_length(St, 0.05, 0.1)
    if bc_type == 'shear':
        # Shear: load at top-left, control at top-right
        top_left = list(node_sets['top'])  # .intersection(l_set))
        if not top_left:
            top_left = [min(node_sets['top'], key=lambda i: St.list_nodes[i][0])]
        nodes_to_load = top_left

        control_candidates = list(node_sets['top'].intersection(node_sets['right']))
        control_node = control_candidates[0] if control_candidates else list(node_sets['top'])[-1]

        Fx = l.get('Fx', 0)
        if Fx != 0:
            St.load_node(node_ids=nodes_to_load, dofs=[0], force=Fx / len(nodes_to_load))
            print(f"  -> Applied Fx = {Fx / 1e3:.1f} kN on {len(nodes_to_load)} nodes")

    else:  # compression
        # Compression: load at top-center, control at same
        top_center = list(node_sets['top'].intersection(node_sets['center_x']))
        if not top_center:
            top_sorted = sorted(node_sets['top'], key=lambda i: St.list_nodes[i][0])
            top_center = [top_sorted[len(top_sorted) // 2]]
        nodes_to_load = top_center
        control_node = top_center[0]

        Fy = l.get('Fy', 0)
        if Fy != 0:
            St.load_node(node_ids=nodes_to_load, dofs=[1], force=Fy / len(nodes_to_load))
            print(f"  -> Applied Fy = {Fy / 1e3:.1f} kN on {len(nodes_to_load)} nodes")

    return St, control_node


# =============================================================================
# RUN FUNCTIONS
# =============================================================================

def run_config(config):
    """Run a single configuration using standard runner."""
    return run_example(config, create_model_gmsh, apply_conditions)


def run_configs(configs):
    """Run multiple configurations using standard runner."""
    return run_examples(configs, create_model_gmsh, apply_conditions)


# =============================================================================
# CONFIGURATION VARIANTS
# =============================================================================

def element_configs(load_type, sizes=None):
    """Generate configs for all element types with a given load type."""
    if sizes is None:
        sizes = [0.1]

    load = COMPRESSION if load_type == 'compression' else SHEAR
    prefix = 'Compression' if load_type == 'compression' else 'Shear'

    configs = []
    for size in sizes:
        # Triangle 3-node (linear)
        configs.append(create_config(
            BASE_CONFIG, f'{prefix}_T3_gmsh_{size}',
            loads=load,
            elements={'size': size, 'type': 'triangle', 'order': 1}
        ))
        # Triangle 6-node (quadratic)
        configs.append(create_config(
            BASE_CONFIG, f'{prefix}_T6_gmsh_{size}',
            loads=load,
            elements={'size': size, 'type': 'triangle', 'order': 2}
        ))
        # Quad 4-node (linear)
        configs.append(create_config(
            BASE_CONFIG, f'{prefix}_Q4_gmsh_{size}',
            loads=load,
            elements={'size': size, 'type': 'quad', 'order': 1}
        ))
        # Quad 8-node (quadratic, serendipity) - converted from Gmsh quad9
        configs.append(create_config(
            BASE_CONFIG, f'{prefix}_Q8_gmsh_{size}',
            loads=load,
            elements={'size': size, 'type': 'quad', 'order': 2}
        ))
        # Quad 9-node (quadratic, Lagrangian) - native Gmsh quad9
        configs.append(create_config(
            BASE_CONFIG, f'{prefix}_Q9_gmsh_{size}',
            loads=load,
            elements={'size': size, 'type': 'quad', 'order': 2, 'prefer_quad9': True}
        ))

    return configs


# Pre-built config lists
COMPRESSION_CONFIGS = element_configs('compression', [0.025])
SHEAR_CONFIGS = element_configs('shear', [0.025])
ALL_CONFIGS = COMPRESSION_CONFIGS + SHEAR_CONFIGS

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run all configurations
    run_configs(SHEAR_CONFIGS)
