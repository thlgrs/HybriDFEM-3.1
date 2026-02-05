import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Library Imports ---
from Examples.utils import find_nodes_at_length, find_nodes_at_height
from Examples.utils.helpers import create_config
from Examples.utils.model_builders import create_fem_column as create_model
from Examples.utils.visualization import visualize
from Examples.utils.solvers import run_solver

RESULTS_ROOT = str(project_root / "Examples" / 'Results' / 'FEM' / 'cantilever_mid')
BASE_CONFIG = {
    'geometry': {
        'x_dim': 3,
        'y_dim': 0.5,
        'thickness': 0.2,
        'nx': 100,
        'ny': 25,
    },
    'elements': {
        'type': 'triangle',  # 'triangle' or 'quad'
        'order': 1,  # 1 (linear), 2 (quadratic), 3 (Quad9 only)
    },
    'material': {
        'E': 30e9,  # Steel
        'nu': 0.0,
        'rho': 0,
    },
    'loading': {
        'Fy': -100e3,  # Downward point load (N)
    },
    'solver': {
        'name': 'linear',
        'optimized': True
    },
    'io': {
        'dir': RESULTS_ROOT,
        'show_nodes': False,
        'scale': 20.0,
        'figsize': (10, 6)
    },
}

def apply_conditions(St, config):
    geom = config['geometry']
    L, H = geom.get('x_dim'), geom.get('y_dim')
    F = config['loading']['Fy']
    tol = 1e-6
    nx, ny = geom.get('nx'), geom.get('ny')
    b_set = set(find_nodes_at_height(St, 0, tolerance=tol))
    l_set = set(find_nodes_at_length(St, 0, tolerance=tol))
    r_set = set(find_nodes_at_length(St, L, tolerance=tol))
    t_set = set(find_nodes_at_height(St, H, tolerance=tol))
    c_set = set(find_nodes_at_length(St, L / 2, tolerance=tol + L / nx))
    m_set = set(find_nodes_at_height(St, H / 2, tolerance=tol + H / ny))
    St.fix_node(list(l_set), [0, 1])
    St.fix_node(list(b_set.intersection(c_set)), [1])
    St.load_node(list(t_set.intersection(r_set)), dofs=[1], force=F)
    return St, list(m_set.intersection(r_set))


def analyze_results(St, config, tip_nodes):
    """Analyze and display results, including reaction forces. Save to markdown."""

    io_conf = config['io']
    g_conf = config['geometry']
    m_conf = config['material']
    e_conf = config['elements']
    l_conf = config['loading']

    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log("\n" + "=" * 60)
    log("   RESULTS ANALYSIS")
    log("=" * 60)

    # --- Model Summary ---
    lines.append("")
    lines.append("## Model Information")
    lines.append(f"- **Element Type**: {e_conf['type']} (order {e_conf['order']})")
    lines.append(f"- **Mesh**: {g_conf['nx']}x{g_conf['ny']} cells")
    lines.append(f"- **Nodes**: {len(St.list_nodes)}, **DOFs**: {St.nb_dofs}")
    lines.append(f"- **Material**: E = {m_conf['E'] / 1e9:.1f} GPa, ν = {m_conf['nu']}")
    lines.append(f"- **Geometry**: L = {g_conf['x_dim']} m, H = {g_conf['y_dim']} m, t = {g_conf['thickness']} m")
    lines.append(f"- **Loading**: Fy = {l_conf['Fy'] / 1000:.1f} kN at tip")

    # 1. Displacements
    log("\n## Displacements")
    if tip_nodes:
        tip_node = tip_nodes[0]
        tip_dofs = St.get_dofs_from_node(tip_node)
        log(f"\n**Displacement at Tip (Node {tip_node}):**")
        log(f"  - ux = {St.U[tip_dofs[0]]:.6e} m")
        log(f"  - uy = {St.U[tip_dofs[1]]:.6e} m")

    log("\n**Maximum Displacements:**")
    ux_all = St.U[0::2]
    uy_all = St.U[1::2]
    log(f"  - Max |ux| = {np.max(np.abs(ux_all)):.6e} m")
    log(f"  - Max |uy| = {np.max(np.abs(uy_all)):.6e} m")
    log(f"  - Min uy   = {np.min(uy_all):.6e} m")

    if tip_nodes:
        tip_dofs = St.get_dofs_from_node(tip_nodes[0])
        log(f"  - delta_FEM = {abs(St.U[tip_dofs[1]]):.6e} m")

    # 3. Reaction Forces
    log("\n## Reaction Forces")
    if len(St.dof_fix) > 0:
        reactions = St.P[St.dof_fix]
        Rx_total = 0.0
        Ry_total = 0.0

        for i, dof in enumerate(St.dof_fix):
            # Even = X, Odd = Y
            if dof % 2 == 0:
                Rx_total += reactions[i]
            else:
                Ry_total += reactions[i]

        log(f"\n**Total Reaction Forces at Supports:**")
        log(f"  - Rx (horizontal) = {Rx_total:+.2f} N ({Rx_total / 1000:+.3f} kN)")
        log(f"  - Ry (vertical)   = {Ry_total:+.2f} N ({Ry_total / 1000:+.3f} kN)")

        applied_Fy = l_conf['Fy']
        err = abs(Ry_total + applied_Fy)
        log(f"\n**Equilibrium Check:**")
        log(f"  - Applied Fy = {applied_Fy:+.2f} N")
        log(f"  - Sum Ry     = {Ry_total:+.2f} N")
        log(f"  - Error      = {err:.2e} N {'[OK]' if err < 1.0 else '[WARN]'}")
    else:
        log("  No fixed DOFs found.")

    # --- Save ---
    os.makedirs(io_conf['dir'], exist_ok=True)
    md_path = os.path.join(io_conf['dir'], io_conf['filename'] + ".md")

    md_content = [f"# Analysis Results: {io_conf['filename']}", ""]
    md_content.extend(lines[3:])

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    print(f"\n[OK] Results saved to: {md_path}")

def run_config(config):
    structure = create_model(config)
    structure, tip_node_list = apply_conditions(structure, config)
    solved_structure = run_solver(structure, config)
    analyze_results(solved_structure, config, tip_node_list)
    visualize(solved_structure, config)

def run_configs(configs):
    # Loop over the standard configs
    for config in configs:
        run_config(config)


# Standard element configurations
T3 = create_config(BASE_CONFIG, 'Fem_Beam_T3', elements={'type': 'triangle', 'order': 1})
T6 = create_config(BASE_CONFIG, 'Fem_Beam_T6', elements={'type': 'triangle', 'order': 2})
Q4 = create_config(BASE_CONFIG, 'Fem_Beam_Q4', elements={'type': 'quad', 'order': 1})
Q8 = create_config(BASE_CONFIG, 'Fem_Beam_Q8', elements={'type': 'quad', 'order': 2})
Q9 = create_config(BASE_CONFIG, 'Fem_Beam_Q9', elements={'type': 'quad', 'order': 3})

ALL_CONFIGS = [T3, T6, Q4, Q8, Q9]

ELEMENT_SIZES = np.geomspace(
    BASE_CONFIG['geometry']['y_dim'],  # 0.2 m (coarsest)
    BASE_CONFIG['geometry']['y_dim'] / 20,  # 0.004 m (finest)
    num=20
)

if __name__ == "__main__":
    run_configs(ALL_CONFIGS)

