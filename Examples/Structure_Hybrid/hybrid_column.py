"""
Hybrid Column Example - Linear Static Analysis
===============================================

Demonstrates hybrid block-FEM structure with alternating slices.

Structure Type: Hybrid (Block + FEM with coupling)
Analysis: Linear static

Configuration:
- Geometry: 1m x 4m x 1m column with 40 alternating slices
- Material: E = 200 GPa, nu = 0.3
- Contact: kn = ks = 200e9 N/m
- Coupling: Constraint, Penalty, Lagrange, or Mortar methods
- Loading: Fx = 500 kN, Fy = -500 kN at top
- BC: Fixed base (cantilever)
"""

import sys
from pathlib import Path

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Imports ---
from Examples.utils.helpers import create_config
from Examples.utils.runner import run_example, run_examples
from Examples.utils.model_builders import create_hybrid_column_slices
from Examples.utils.boundary_conditions import find_node_sets


# =============================================================================
# CONFIGURATION
# =============================================================================

EXAMPLE_ROOT = str(project_root / "Examples" / 'Results' / 'Hybrid' / 'hybrid_column' / 'illustration')

BASE_CONFIG = {
    'geometry': {
        'width': 0.2,
        'thickness': 0.2,
        'n_slices': 2,
        'start_with': 'block',
        'block_slice_height': 0.4,
        'fem_slice_height': 0.8,
        'nx': 1,
        'ny_block_slice': 4,
        'ny_fem_slice': 8,
        'coupling_offset': 1e-6,
    },
    'elements': {
        'type': 'quad',
        'order': 1,
    },
    'material': {
        'block': {'E': 200e9, 'nu': 0.3, 'rho': 0},
        'fem': {'E': 200e9, 'nu': 0.3, 'rho': 0},
    },
    'contact': {
        'kn': 200e9,
        'ks': 200e9,
        'LG': True,
        'nb_cps': 20,
    },
    'coupling': {
        'method': 'constraint',
        'tolerance': 0.001,
        'integration_order': 2,
        'penalty_stiffness': 1e12,
        'interface_orientation': 'horizontal',
    },
    'loads': {
        'Fx': 5e5,
        'Fy': -5e5,
    },
    'bc': {
        'type': 'cantilever',
    },
    'solver': {
        'name': 'linear',
    },
    'io': {
        'filename': 'hybrid_column',
        'dir': EXAMPLE_ROOT,
        'show_nodes': False,
        'scale': 50,
        'figsize': (4, 7),
    }
}


# =============================================================================
# MODEL GENERATION
# =============================================================================

def create_model(config):
    """Create hybrid column with alternating block/FEM slices."""
    return create_hybrid_column_slices(config)


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

def apply_conditions(St, config):
    """Apply cantilever BCs for hybrid sliced structure."""
    node_sets = find_node_sets(St, config)
    l = config.get('loads', {})
    bc_type = config.get('bc', {}).get('type', 'cantilever')

    print(f"\nApplying Conditions: {bc_type.upper()}")
    print(f"  Total Height ~ {node_sets['height']:.4f} m")

    # Fix bottom nodes (blocks have 3 DOFs, FEM has 2)
    bottom_nodes = list(node_sets['bottom'])
    for node in bottom_nodes:
        dofs = St.get_dofs_from_node(node)
        St.fix_node(node_ids=[node], dofs=list(range(len(dofs))))
    print(f"  -> Fixed base: {len(bottom_nodes)} nodes")

    # Find control node
    top_nodes = list(node_sets['top'])
    center_candidates = list(node_sets['top'].intersection(node_sets['center_x']))
    if center_candidates:
        control_node = center_candidates[0]
    else:
        top_sorted = sorted(top_nodes, key=lambda i: St.list_nodes[i][0])
        control_node = top_sorted[len(top_sorted) // 2]

    # Apply loads uniformly to top
    n_load = len(top_nodes)
    Fx = l.get('Fx', 0)
    Fy = l.get('Fy', 0)

    if Fx != 0:
        St.load_node(node_ids=top_nodes, dofs=[0], force=Fx / n_load)
        print(f"  -> Applied Fx = {Fx/1e3:.1f} kN on {n_load} nodes")

    if Fy != 0:
        St.load_node(node_ids=top_nodes, dofs=[1], force=Fy / n_load)
        print(f"  -> Applied Fy = {Fy/1e3:.1f} kN on {n_load} nodes")

    return St, control_node


# =============================================================================
# CONFIGURATION VARIANTS - By Coupling Method
# =============================================================================

# Element configurations
def make_configs(base, method, prefix):
    """Generate configs for all element types with a coupling method."""
    coupling = {'method': method}
    if method == 'mortar':
        # Higher integration order for quadratic elements
        return [
            create_config(base, f'{prefix}_T3', coupling=coupling, elements={'type': 'triangle', 'order': 1}),
            create_config(base, f'{prefix}_T6', coupling={**coupling, 'integration_order': 3}, elements={'type': 'triangle', 'order': 2}),
            create_config(base, f'{prefix}_Q4', coupling=coupling, elements={'type': 'quad', 'order': 1}),
            create_config(base, f'{prefix}_Q8', coupling={**coupling, 'integration_order': 3}, elements={'type': 'quad', 'order': 2}),
        ]
    return [
        create_config(base, f'{prefix}_T3', coupling=coupling, elements={'type': 'triangle', 'order': 1}),
        create_config(base, f'{prefix}_T6', coupling=coupling, elements={'type': 'triangle', 'order': 2}),
        create_config(base, f'{prefix}_Q4', coupling=coupling, elements={'type': 'quad', 'order': 1}),
        create_config(base, f'{prefix}_Q8', coupling=coupling, elements={'type': 'quad', 'order': 2}),
    ]


CONSTRAINT = make_configs(BASE_CONFIG, 'constraint', 'Hybrid_Column_Constraint')
PENALTY = make_configs(BASE_CONFIG, 'penalty', 'Hybrid_Column_Penalty')
LAGRANGE = make_configs(BASE_CONFIG, 'lagrange', 'Hybrid_Column_Lagrange')
MORTAR = make_configs(BASE_CONFIG, 'mortar', 'Hybrid_Column_Mortar')

ALL_CONFIGS = CONSTRAINT + PENALTY + LAGRANGE + MORTAR


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_config(config):
    """Run a single configuration."""
    return run_example(config, create_model, apply_conditions)


def run_configs(configs):
    """Run multiple configurations."""
    return run_examples(configs, create_model, apply_conditions)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_configs(ALL_CONFIGS)
