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

import os
import sys
from pathlib import Path

import numpy as np

from Core.Objects.ConstitutiveLaw.Material import PlaneStrain, PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
# --- Library Imports ---
from Core.Objects.FEM.Mesh import Mesh
from Core.Objects.FEM.Quads import Quad4, Quad8
from Core.Objects.FEM.Triangles import Triangle3, Triangle6
from Core.Structures.Structure_FEM import Structure_FEM
from Examples.utils import (
    find_nodes_at_height,
    find_nodes_at_length,
    plot_deformed,
    plot_displacement,
    plot_initial,
)
from Examples.utils.boundary_conditions import apply_compression_bc, apply_shear_bc
from Examples.utils.helpers import create_config
from Examples.utils.solvers import run_solver
from Examples.utils.visualization import visualize

# =============================================================================
# CONFIGURATION
# =============================================================================
# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
RESULTS_ROOT = str(project_root / "Examples" / "Results" / "FEM" / "I_beam_gmsh")

BASE_CONFIG = {
    "geometry": {
        "width": 1.0,
        "height": 1.0,
        "thickness": 0.25,
    },
    "elements": {
        "size": 0.01,  # Element size for Gmsh (m)
        "type": "triangle",
        "order": 1,
    },
    "material": {
        "E": 10e9,
        "nu": 0.0,
        "rho": 0,
    },
    "loads": {
        "name": "compression",
        "Fx": 0,
        "Fy": -1e6,
    },
    "solver": {
        "name": "linear",
    },
    "io": {
        "filename": "fem_complex_gmsh",
        "dir": RESULTS_ROOT,
        "show_nodes": False,
        "scale": 50.0,
        "figsize": (4, 4),
    },
}

# =============================================================================
# LOAD DEFINITIONS
# =============================================================================

COMPRESSION = {
    "name": "compression",
    "Fx": 0,
    "Fy": -1e6,
}

SHEAR = {
    "name": "shear",
    "Fx": 1e6,
    "Fy": 0,
}


# =============================================================================
# MODEL CREATION
# =============================================================================


def generate_fem_mesh_gmsh(config):
    """Generate square mesh using Gmsh."""
    g = config["geometry"]
    e = config["elements"]

    width = g["width"]
    height = g["height"]

    # Define square corners (counter-clockwise)
    points = [
        (0, 0),  # Bottom-left
        (width, 0),
        (width, height / 10),
        (2 * width / 3, height / 3),
        (2 * width / 3, 2 * height / 3),
        (width, 9 * height / 10),
        (width, height),
        (0, height),
        (0, 9 * height / 10),
        (width / 3, 2 * height / 3),
        (width / 3, height / 3),
        (0, height / 10),
    ]

    # Define edge groups for boundary conditions
    edge_groups = {
        "bottom": [0],
        "top": [6],
    }

    mesh = Mesh(
        points=points,
        element_type=e["type"],
        element_size=e["size"],
        order=e["order"],
        name="square_mesh",
        edge_groups=edge_groups,
    )

    mesh.generate_mesh()
    return mesh


def create_model_gmsh(config):
    """Create FEM structure from Gmsh mesh."""
    g = config["geometry"]
    m = config["material"]
    e = config["elements"]

    print("  Generating FEM mesh with Gmsh...")
    mesh = generate_fem_mesh_gmsh(config)

    mat_fem = PlaneStrain(**m)
    geom = Geometry2D(t=g["thickness"])

    # Check if user wants Quad9 elements (9-node Lagrangian)
    prefer_quad9 = e.get("prefer_quad9", False)

    St = Structure_FEM.from_mesh(mesh, mat_fem, geom, prefer_quad9=prefer_quad9)
    St.make_nodes()

    print(f"  -> Created mesh: {len(St.list_fes)} elements, {len(St.list_nodes)} nodes")
    return St


def apply_conditions(St, config):
    """Apply BCs based on load type in config."""
    geom = config["geometry"]
    L, H = geom.get("height"), geom.get("width")
    F = config["loads"]["Fy"]
    tol = 1e-6
    b_set = set(find_nodes_at_height(St, 0, tolerance=tol))
    l_set = set(find_nodes_at_length(St, 0, tolerance=tol))
    r_set = set(find_nodes_at_length(St, L, tolerance=tol))
    t_set = set(find_nodes_at_height(St, H, tolerance=tol))

    St.fix_node(list(b_set), [1])
    St.load_node(list(t_set), dofs=[1], force=F / len(t_set))
    return St, list(t_set)[len(t_set) // 2]


def analyze_results(St, config, control_node):
    """Analyze and display results, including reaction forces."""
    io_conf = config["io"]
    g_conf = config["geometry"]
    m_conf = config["material"]
    e_conf = config["elements"]
    l_conf = config["loads"]

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
    lines.append(f"- **Element Size**: {e_conf['size']} m")
    lines.append(f"- **Nodes**: {len(St.list_nodes)}, **DOFs**: {St.nb_dofs}")
    lines.append(f"- **Material**: E = {m_conf['E'] / 1e9:.1f} GPa, ν = {m_conf['nu']}")
    lines.append(
        f"- **Geometry**: W = {g_conf['width']} m, H = {g_conf['height']} m, t = {g_conf['thickness']} m"
    )

    bc_type = l_conf.get("name", "compression")
    if bc_type == "shear":
        lines.append(f"- **Loading**: Fx = {l_conf.get('Fx', 0) / 1000:.1f} kN (shear)")
    else:
        lines.append(
            f"- **Loading**: Fy = {l_conf.get('Fy', 0) / 1000:.1f} kN (compression)"
        )

    # --- Displacements ---
    log("\n## Displacements")

    if control_node is not None:
        control_dofs = St.get_dofs_from_node(control_node)
        log(f"\n**Displacement at Control Node {control_node}:**")
        log(f"  - ux = {St.U[control_dofs[0]]:.6e} m")
        log(f"  - uy = {St.U[control_dofs[1]]:.6e} m")

    log("\n**Maximum Displacements:**")
    ux_all = St.U[0::2]
    uy_all = St.U[1::2]
    log(f"  - Max |ux| = {np.max(np.abs(ux_all)):.6e} m")
    log(f"  - Max |uy| = {np.max(np.abs(uy_all)):.6e} m")

    # --- Reaction Forces ---
    log("\n## Reaction Forces")
    if len(St.dof_fix) > 0:
        reactions = St.P[St.dof_fix]

        Rx_total = 0.0
        Ry_total = 0.0

        for i, dof in enumerate(St.dof_fix):
            if dof % 2 == 0:
                Rx_total += reactions[i]
            else:
                Ry_total += reactions[i]

        log(f"\n**Total Reaction Forces at Supports:**")
        log(f"  - Rx (horizontal) = {Rx_total:+.2f} N ({Rx_total / 1000:+.3f} kN)")
        log(f"  - Ry (vertical)   = {Ry_total:+.2f} N ({Ry_total / 1000:+.3f} kN)")

        # Equilibrium check
        applied_Fx = l_conf.get("Fx", 0)
        applied_Fy = l_conf.get("Fy", 0)

        log(f"\n**Equilibrium Check:**")
        if bc_type == "shear":
            equilibrium_error = abs(Rx_total + applied_Fx)
            log(f"  - Applied Fx = {applied_Fx:+.2f} N")
            log(f"  - Sum Rx     = {Rx_total:+.2f} N")
        else:
            equilibrium_error = abs(Ry_total + applied_Fy)
            log(f"  - Applied Fy = {applied_Fy:+.2f} N")
            log(f"  - Sum Ry     = {Ry_total:+.2f} N")

        log(
            f"  - Error      = {equilibrium_error:.2e} N {'[OK]' if equilibrium_error < 1.0 else '[WARN]'}"
        )

    # --- Save to Markdown ---
    os.makedirs(io_conf["dir"], exist_ok=True)
    md_path = os.path.join(io_conf["dir"], io_conf["filename"] + ".md")

    md_content = [f"# Analysis Results: {io_conf['filename']}", ""]
    md_content.extend(lines[3:])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))

    print(f"\n[OK] Results saved to: {md_path}")


# =============================================================================
# RUN FUNCTIONS
# =============================================================================


def run_config(config):
    """Execute full analysis pipeline for a given configuration."""
    St = create_model_gmsh(config)
    St, control_node = apply_conditions(St, config)
    St = run_solver(St, config)
    analyze_results(St, config, control_node)
    visualize(St, config)
    return St


def run_configs(configs):
    """Run analyses for multiple configurations."""
    results = []
    for config in configs:
        results.append(run_config(config))
    return results


# =============================================================================
# CONFIGURATION VARIANTS
# =============================================================================


def element_configs(load_type, sizes=None):
    """Generate configs for all element types with a given load type."""
    if sizes is None:
        sizes = [0.1]

    load = COMPRESSION
    prefix = "I_beam"

    configs = []
    for size in sizes:
        # Triangle 3-node (linear)
        configs.append(
            create_config(
                BASE_CONFIG,
                f"{prefix}_T3_gmsh_{size}",
                loads=load,
                elements={"size": size, "type": "triangle", "order": 1},
            )
        )
        # Triangle 6-node (quadratic)
        configs.append(
            create_config(
                BASE_CONFIG,
                f"{prefix}_T6_gmsh_{size}",
                loads=load,
                elements={"size": size, "type": "triangle", "order": 2},
            )
        )
        # Quad 4-node (linear)
        configs.append(
            create_config(
                BASE_CONFIG,
                f"{prefix}_Q4_gmsh_{size}",
                loads=load,
                elements={"size": size, "type": "quad", "order": 1},
            )
        )
        # Quad 8-node (quadratic, serendipity) - converted from Gmsh quad9
        configs.append(
            create_config(
                BASE_CONFIG,
                f"{prefix}_Q8_gmsh_{size}",
                loads=load,
                elements={"size": size, "type": "quad", "order": 2},
            )
        )
        # Quad 9-node (quadratic, Lagrangian) - native Gmsh quad9
        configs.append(
            create_config(
                BASE_CONFIG,
                f"{prefix}_Q9_gmsh_{size}",
                loads=load,
                elements={
                    "size": size,
                    "type": "quad",
                    "order": 2,
                    "prefer_quad9": True,
                },
            )
        )

    return configs


# Pre-built config lists
COMPRESSION_CONFIGS = element_configs("compression", [0.025])
ALL_CONFIGS = COMPRESSION_CONFIGS

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Run all configurations
    run_configs(ALL_CONFIGS)
