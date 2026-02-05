"""
Hybrid Beam Example - Alternating Vertical Slices
===================================================
Demonstrates a hybrid beam with alternating Block and FEM vertical slices.

Structure:
- Beam of length L, height H.
- Alternating slices: Block (2 blocks wide) | FEM (refined) | Block | ...
- Interface: Vertical lines.
- Coupling: Constraint, Penalty, Lagrange, or Mortar.

Configuration allows control over:
- Slice widths and counts.
- Refinement (ny global, nx_fem local).
- Materials and Element types.
"""

import os
import sys
from pathlib import Path

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Library Imports ---
from Examples.utils.model_builders import create_hybrid_beam_slices, find_nodes_at_length, find_nodes_at_height
from Examples.utils.visualization import visualize
from Examples.utils.solvers import run_solver
from Examples.utils.helpers import create_config

# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_ROOT = str(project_root / "Examples" / 'Results' / 'Hybrid' / 'hybrid_beam')

# Default Configuration
BASE_CONFIG = {
    'geometry': {
        'height': 0.5,
        'thickness': 0.2,  # Out-of-plane thickness (m)
        'n_slices': 3,  # Number of alternating slices
        'start_with': 'block',  # First slice type
        'block_slice_width': 1,  # Width of block slices (m)
        'fem_slice_width': 1,  # Width of FEM slices (m)

        # Mesh Refinement
        'ny': 5,  # Global vertical refinement (elements/blocks high)
        'nx_block_slice': 10,  # Fixed: Block slices are 2 blocks wide
        'nx_fem_slice': 10,  # Refinement of FEM slices (elements wide)

        'coupling_offset': 1e-6,  # Gap between slices
    },
    'elements': {
        'type': 'quad',  # 'triangle' or 'quad'
        'order': 1,  # 1 (linear), 2 (quadratic)
    },
    'material': {
        'block': {'E': 30e9, 'nu': 0, 'rho': 0},
        'fem': {'E': 30e9, 'nu': 0, 'rho': 0},
    },
    'contact': {
        'kn': 30e9,
        'ks': 30e9,
        'LG': True,
        'nb_cps': 20
    },
    'coupling': {
        'method': 'constraint',
        'tolerance': 1e-4,
        'integration_order': 3,
        'penalty_stiffness': 1e12,
        'interface_orientation': 'vertical',  # Vertical slices -> vertical interfaces
    },
    'loads': {
        'Type': 'Point',  # 'TipShear' or 'Distributed'
        'Fy': -100e3,  # Vertical load (N)
    },
    'bc': {
        'type': 'cantilever',  # 'cantilever' (fix left) or 'simply_supported'
    },
    'io': {
        'filename': 'hybrid_beam',
        'dir': RESULTS_ROOT,
        'show_nodes': False,
        'scale': 20,
        'figsize': (8, 4),
    }
}

gemetry2 = {
    'height': 0.5,
    'thickness': 0.2,
    'n_slices': 3,  # Fixed: 4 slices (block-fem-block-fem)
    'start_with': 'block',
    'block_slice_width': 1,  # Fixed: 0.625 m per slice
    'fem_slice_width': 1,  # Fixed: 0.625 m per slice
    # Total length = 4 * 0.625 = 2.5 m

    # Mesh Refinement (these will vary)
    'ny': 4,
    'nx_block_slice': 2,
    'nx_fem_slice': 2,

    'coupling_offset': 1e-6,
}
# =============================================================================
# MODEL GENERATION
# =============================================================================

def create_model(config):
    return create_hybrid_beam_slices(config)


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

def apply_conditions(St, config):
    g = config['geometry']
    l_conf = config['loads']
    bc_type = config['bc']['type']

    # Calculate total length to find right-most nodes
    # Formula: n_block * w_block + n_fem * w_fem + gaps
    n_slices = g['n_slices']
    start_block = (g['start_with'] == 'block')

    # Count types
    if start_block:
        n_block = (n_slices + 1) // 2
        n_fem = n_slices // 2
    else:
        n_fem = (n_slices + 1) // 2
        n_block = n_slices // 2

    total_length = (n_block * g['block_slice_width'] +
                    n_fem * g['fem_slice_width'] +
                    (n_slices - 1) * g.get('coupling_offset', 0))
    total_height = g['height']
    print(f"\nApplying Conditions (Type: {bc_type})")
    print(f"  Total Length ~ {total_length:.4f} m")

    # 1. Supports
    tol = 1e-3
    left_set = set(find_nodes_at_length(St, 0.0, tolerance=tol))
    right_set = set(find_nodes_at_length(St, total_length, tolerance=tol))
    center_set = set(find_nodes_at_length(St, total_length / 2, tolerance=5e-2))
    bottom_set = set(find_nodes_at_height(St, 0.0, tolerance=tol))
    top_set = set(find_nodes_at_height(St, total_height, tolerance=tol))

    if bc_type == 'cantilever':
        # Fix Left Edge fully
        nodes_to_fix = list(left_set)
        for node in nodes_to_fix:
            St.fix_node(node_ids=[node], dofs=[0, 1, 2])
        print(f"  -> Fixed Left Edge ({len(nodes_to_fix)} nodes)")
        nodes_to_load = list(right_set.intersection(top_set))
        control_node = nodes_to_load[0]

    elif bc_type == 'simply_supported':
        nodes_to_fix = list(bottom_set.intersection(left_set.union(right_set)))
        for node in nodes_to_fix:
            St.fix_node(node_ids=[node], dofs=[0, 1])
        nodes_to_load = list(top_set)
        control_node = list(top_set.intersection(center_set))[0]

    else:
        nodes_to_fix = list(left_set) + list(right_set)
        for node in nodes_to_fix:
            St.fix_node(node_ids=[node], dofs=[0, 1])
        nodes_to_load = list(top_set)
        control_node = list(top_set.intersection(center_set))[0]

    if l_conf.get('Fx', False):
        Fx = l_conf['Fx'] / len(nodes_to_load)
        St.load_node(node_ids=nodes_to_load, dofs=[0], force=Fx)
    if l_conf.get('Fy', False):
        Fy = l_conf['Fy'] / len(nodes_to_load)
        St.load_node(node_ids=nodes_to_load, dofs=[1], force=Fy)

    return St, control_node


# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

def analyze_results(St, config, control_node):
    lines = []

    def log(text=""): print(text); lines.append(text)

    log("\n" + "=" * 60)
    log("   RESULTS ANALYSIS")
    log("=" * 60)

    # Expand displacement if constraint
    if hasattr(St, 'coupling_T') and St.coupling_T is not None:
        St.U = St.coupling_T @ St.U
        log("  (Expanded from reduced DOF system)")

    # Displacements
    dofs = St.get_dofs_from_node(control_node)
    ux = St.U[dofs[0]]
    uy = St.U[dofs[1]]

    log(f"\n**Control Node {control_node} (Right-Tip):**")
    log(f"  - ux = {ux:.6e} m")
    log(f"  - uy = {uy:.6e} m")

    # Save
    io_conf = config['io']
    os.makedirs(io_conf['dir'], exist_ok=True)
    md_path = os.path.join(io_conf['dir'], io_conf['filename'] + ".md")
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n[OK] Saved to {md_path}")


# =============================================================================
# EXECUTION
# =============================================================================

def run_config(config):
    name = config['io']['filename']
    print(f"\nRunning: {name}")
    St = create_model(config)
    St, c_node = apply_conditions(St, config)
    St = run_solver(St, config)
    analyze_results(St, config, c_node)
    visualize(St, config)
    return St


def run_configs(configs):
    for config in configs:
        run_config(config)

# --- Configuration Groups ---


CONSTRAINT_T3 = create_config(BASE_CONFIG, 'Hybrid_Beam_Constraint_T3', elements={'type': 'triangle', 'order': 1},
                              coupling={'method': 'constraint'})
CONSTRAINT_T6 = create_config(BASE_CONFIG, 'Hybrid_Beam_Constraint_T6', elements={'type': 'triangle', 'order': 2},
                              coupling={'method': 'constraint'})
CONSTRAINT_Q4 = create_config(BASE_CONFIG, 'Hybrid_Beam_Constraint_Q4', elements={'type': 'quad', 'order': 1},
                              coupling={'method': 'constraint'})
CONSTRAINT_Q8 = create_config(BASE_CONFIG, 'Hybrid_Beam_Constraint_Q8', elements={'type': 'quad', 'order': 2},
                              coupling={'method': 'constraint'})

PENALTY_T3 = create_config(BASE_CONFIG, 'Hybrid_Beam_Penalty_T3', elements={'type': 'triangle', 'order': 1},
                           coupling={'method': 'penalty'})
PENALTY_T6 = create_config(BASE_CONFIG, 'Hybrid_Beam_Penalty_T6', elements={'type': 'triangle', 'order': 2},
                           coupling={'method': 'penalty'})
PENALTY_Q4 = create_config(BASE_CONFIG, 'Hybrid_Beam_Penalty_Q4', elements={'type': 'quad', 'order': 1},
                           coupling={'method': 'penalty'})
PENALTY_Q8 = create_config(BASE_CONFIG, 'Hybrid_Beam_Penalty_Q8', elements={'type': 'quad', 'order': 2},
                           coupling={'method': 'penalty'})

LAGRANGE_T3 = create_config(BASE_CONFIG, 'Hybrid_Beam_Lagrange_T3', elements={'type': 'triangle', 'order': 1},
                            coupling={'method': 'lagrange'})
LAGRANGE_T6 = create_config(BASE_CONFIG, 'Hybrid_Beam_Lagrange_T6', elements={'type': 'triangle', 'order': 2},
                            coupling={'method': 'lagrange'})
LAGRANGE_Q4 = create_config(BASE_CONFIG, 'Hybrid_Beam_Lagrange_Q4', elements={'type': 'quad', 'order': 1},
                            coupling={'method': 'lagrange'})
LAGRANGE_Q8 = create_config(BASE_CONFIG, 'Hybrid_Beam_Lagrange_Q8', elements={'type': 'quad', 'order': 2},
                            coupling={'method': 'lagrange'})

MORTAR_T3 = create_config(BASE_CONFIG, 'Hybrid_Beam_Mortar_T3', elements={'type': 'triangle', 'order': 1},
                          coupling={'method': 'mortar'})
MORTAR_T6 = create_config(BASE_CONFIG, 'Hybrid_Beam_Mortar_T6', elements={'type': 'triangle', 'order': 2},
                          coupling={'method': 'mortar', 'integration_order': 3})
MORTAR_Q4 = create_config(BASE_CONFIG, 'Hybrid_Beam_Mortar_Q4', elements={'type': 'quad', 'order': 1},
                          coupling={'method': 'mortar'})
MORTAR_Q8 = create_config(BASE_CONFIG, 'Hybrid_Beam_Mortar_Q8', elements={'type': 'quad', 'order': 2},
                          coupling={'method': 'mortar', 'integration_order': 3})

CONSTRAINT = [CONSTRAINT_T3, CONSTRAINT_T6, CONSTRAINT_Q4, CONSTRAINT_Q8, ]
PENALTY = [PENALTY_T3, PENALTY_T6, PENALTY_Q4, PENALTY_Q8, ]
LAGRANGE = [LAGRANGE_T3, LAGRANGE_T6, LAGRANGE_Q4, LAGRANGE_Q8, ]
MORTAR = [MORTAR_T3, MORTAR_T6, MORTAR_Q4, MORTAR_Q8, ]
ALL_CONFIGS = CONSTRAINT + PENALTY + LAGRANGE + MORTAR

if __name__ == "__main__":
    # run_config(LAGRANGE_Q4)
    run_configs(ALL_CONFIGS)
