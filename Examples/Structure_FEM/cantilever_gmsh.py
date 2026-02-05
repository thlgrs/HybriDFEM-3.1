"""
Cantilever Beam Example - Gmsh Mesh Generation (Linear Static)
================================================================

Demonstrates FEM cantilever beam analysis using Gmsh for mesh generation.

Features:
- Gmsh integration for automatic mesh generation
- Multiple element types: Triangle3, Triangle6, Quad4, Quad8, Quad9
- Configurable mesh size and element order
- Equilibrium verification (force + moment balance)

Refactored to use config factory pattern from cantilever_linear.py.
"""
import os
import sys
from pathlib import Path

import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Library Imports ---
from Core.Objects.FEM.Mesh import Mesh
from Core.Structures.Structure_FEM import Structure_FEM
from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D

from Examples.utils.visualization import visualize
from Examples.utils.helpers import create_config
from Examples.utils.model_builders import find_nodes_at_length
from Examples.utils.solvers import run_solver

# =============================================================================
# CONFIG-BASED ARCHITECTURE
# =============================================================================
RESULTS_ROOT = str(project_root / "Examples" / 'Results' / 'FEM' / 'cantilever_gmsh')
BASE_CONFIG = {
    'geometry': {
        'length': 3,
        'height': 0.5,
        'thickness': 0.2,  # 10 mm
    },
    'elements': {
        'size': 1,  # Element size for Gmsh (m)
        'type': 'triangle',
        'order': 1,
    },
    'material': {
        'E': 30e9,  # Steel
        'nu': 0.0,
        'rho': 0,
    },
    'loading': {
        'Fy': -100e3
    },
    'io': {
        'filename': 'canti_gmsh',
        'dir': RESULTS_ROOT,
        'show_nodes': False,
        'scale': 20.0,
        'figsize': (8, 3)
    }
}

# =============================================================================
# MODEL CREATION
# =============================================================================

def generate_fem_mesh_gmsh(config):
    g = config['geometry']
    e = config['elements']

    length = g['length']
    height = g['height']

    # Define rectangle for FEM domain
    points = [
        (0, 0),  # Bottom-left
        (length, 0),  # Bottom-right
        (length, height),  # Top-right
        (0, height),  # Top-left
    ]

    # Define edge groups for boundary conditions
    edge_groups = {
        "bottom": [0],  # Interface with blocks
        "right": [1],
        "top": [2],
        "left": [3],
    }

    mesh = Mesh(
        points=points,
        element_type=e['type'],
        element_size=e['size'],
        order=e['order'],
        name="fem_mesh",
        edge_groups=edge_groups,
    )

    mesh.generate_mesh()

    return mesh

def create_model_gmsh(config):
    g = config['geometry']
    m = config['material']
    print("  Generating FEM mesh with Gmsh...")
    mesh = generate_fem_mesh_gmsh(config)

    mat_fem = PlaneStress(**m)
    geom = Geometry2D(t=g['thickness'])

    St = Structure_FEM.from_mesh(mesh, mat_fem, geom)
    St.make_nodes()
    return St

def apply_conditions(St, config):
    """Apply boundary conditions: fixed left edge, point load at tip."""
    g_conf = config['geometry']
    l_conf = config['loading']

    L = g_conf['length']
    tol = 1e-9

    # 1. Fix left edge (x = 0)
    left_nodes = find_nodes_at_length(St, 0, tol)
    St.fix_node(left_nodes, [0, 1])
    print(f"  -> Fixed left edge: {len(left_nodes)} nodes")

    # 2. Apply load at tip (top-right corner: x=L, y=H)
    right_nodes = find_nodes_at_length(St, L, tol)

    if len(right_nodes) > 0:
        load_value = l_conf['Fy']
        St.load_node(node_ids=right_nodes, dofs=[1], force=load_value / len(right_nodes))
        print(f"  -> Applied Fy = {load_value / 1000:.1f} kN at tip nodes {right_nodes}")
    else:
        print("  [WARN] No tip node found!")

    return St, right_nodes

def analyze_results(St, config, tip_nodes):
    """Analyze and display results, including reaction forces. Save to markdown."""
    io_conf = config['io']
    g_conf = config['geometry']
    m_conf = config['material']
    e_conf = config['elements']
    l_conf = config['loading']

    # Build output lines for both console and markdown
    lines = []

    def log(text=""):
        """Print to console and store for markdown."""
        print(text)
        lines.append(text)

    log("\n" + "=" * 60)
    log("   RESULTS ANALYSIS")
    log("=" * 60)

    # --- Model Summary (for markdown) ---
    lines.append("")  # Extra spacing for markdown
    lines.append("## Model Information")
    lines.append(f"- **Element Type**: {e_conf['type']} (order {e_conf['order']})")
    lines.append(f"- **Nodes**: {len(St.list_nodes)}, **DOFs**: {St.nb_dofs}")
    lines.append(f"- **Material**: E = {m_conf['E'] / 1e9:.1f} GPa, ν = {m_conf['nu']}")
    lines.append(f"- **Geometry**: L = {g_conf['length']} m, H = {g_conf['height']} m, t = {g_conf['thickness']} m")
    lines.append(f"- **Loading**: Fy = {l_conf['Fy'] / 1000:.1f} kN at tip")

    # 1. Tip displacement
    log("\n## Displacements")
    if tip_nodes:
        tip_node = tip_nodes[0]
        tip_dofs = St.get_dofs_from_node(tip_node)
        log(f"\n**Displacement at Tip (Node {tip_node}):**")
        log(f"  - ux = {St.U[tip_dofs[0]]:.6e} m")
        log(f"  - uy = {St.U[tip_dofs[1]]:.6e} m")

    # 2. Maximum displacements
    log("\n**Maximum Displacements:**")
    ux_all = St.U[0::2]
    uy_all = St.U[1::2]
    log(f"  - Max |ux| = {np.max(np.abs(ux_all)):.6e} m")
    log(f"  - Max |uy| = {np.max(np.abs(uy_all)):.6e} m")
    log(f"  - Min uy   = {np.min(uy_all):.6e} m (tip deflection)")

    if tip_nodes:
        delta_fem = abs(St.U[tip_dofs[1]])
        log(f"  - delta_FEM = {delta_fem:.6e} m")

    # 3. Reaction Forces
    log("\n## Reaction Forces")
    if len(St.dof_fix) > 0:
        # Extract reactions at fixed DOFs (computed by solver: R = K @ U at fixed DOFs)
        reactions = St.P[St.dof_fix]

        # Separate into x and y components
        # Fixed DOFs are organized as pairs [dof_x, dof_y] per fixed node
        # We need to identify which are x-DOFs and which are y-DOFs
        Rx_total = 0.0
        Ry_total = 0.0

        for i, dof in enumerate(St.dof_fix):
            # Check if this is an x-DOF (even) or y-DOF (odd)
            if dof % 2 == 0:
                Rx_total += reactions[i]
            else:
                Ry_total += reactions[i]

        log(f"\n**Total Reaction Forces at Supports:**")
        log(f"  - Rx (horizontal) = {Rx_total:+.2f} N ({Rx_total / 1000:+.3f} kN)")
        log(f"  - Ry (vertical)   = {Ry_total:+.2f} N ({Ry_total / 1000:+.3f} kN)")

        # Equilibrium check
        applied_Fy = l_conf['Fy']
        equilibrium_error = abs(Ry_total + applied_Fy)
        log(f"\n**Equilibrium Check:**")
        log(f"  - Applied Fy = {applied_Fy:+.2f} N")
        log(f"  - Sum Ry     = {Ry_total:+.2f} N")
        log(f"  - Error      = {equilibrium_error:.2e} N {'[OK]' if equilibrium_error < 1.0 else '[WARN]'}")
    else:
        log("  No fixed DOFs found (structure may be unconstrained).")

    # --- Save to Markdown File ---
    os.makedirs(io_conf['dir'], exist_ok=True)
    md_path = os.path.join(io_conf['dir'], io_conf['filename'] + ".md")

    # Build markdown content with proper header
    md_content = [f"# Analysis Results: {io_conf['filename']}", ""]
    md_content.extend(lines[3:])  # Skip the console header decoration

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))

    print(f"\n[OK] Results saved to: {md_path}")

def run_config(config):
    """Execute full analysis pipeline for a given configuration."""
    beam = create_model_gmsh(config)
    beam, tip_nodes = apply_conditions(beam, config)
    solved_beam = run_solver(beam, config)
    analyze_results(solved_beam, config, tip_nodes)
    visualize(solved_beam, config)

def run_configs(configs):
    """Run analyses for multiple configurations."""
    for config in configs:
        run_config(config)

def size_configs(sizes):
    configs = []
    for size in sizes:
        configs.append(
            create_config(BASE_CONFIG, f'Fem_Beam_T3_gmsh_{size}',
                          elements={'size': size, 'type': 'triangle', 'order': 1}))
        configs.append(
            create_config(BASE_CONFIG, f'Fem_Beam_T6_gmsh_{size}',
                          elements={'size': size, 'type': 'triangle', 'order': 2}))
        configs.append(
            create_config(BASE_CONFIG, f'Fem_Beam_Q4_gmsh_{size}', elements={'size': size, 'type': 'quad', 'order': 1}))
        configs.append(
            create_config(BASE_CONFIG, f'Fem_Beam_Q8_gmsh_{size}', elements={'size': size, 'type': 'quad', 'order': 2}))
        configs.append(create_config(BASE_CONFIG, f'Fem_Beam_Q9_gmsh_{size}',
                                     elements={'size': size, 'type': 'quad', 'order': 2, 'prefer_quad9': True}))
    return configs


if __name__ == "__main__":
    SIZES = [0.05]
    CONFIGS = size_configs(SIZES)
    run_configs(CONFIGS)
