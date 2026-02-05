"""
Convergence Study for FEM Elements - Square Geometry
=====================================================

This script performs a mesh convergence analysis on a 1m x 1m square under:
1. Compression loading (uniform vertical load at top)
2. Shear loading (uniform horizontal load at top)

The analysis:
1. Computes a reference solution with a fine mesh
2. Runs simulations at various element sizes
3. Plots relative displacement error vs element size (log-log scale)
4. Estimates convergence rates from the slope

Configurable for all element types: T3, T6, Q4, Q8

Optimized L2 error: Uses hash-based node matching (O(n) instead of O(n²) KD-tree).
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Solvers.Static import StaticLinear
# --- Library Imports ---
from Core.Structures.Structure_FEM import Structure_FEM
from Examples.utils.visualization import plot_displacement, plot_stress

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# --- Plot Configuration ---
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SquareConfig:
    """Configuration for square convergence study."""

    # Geometry
    L: float = 1.0 / 2  # Side length [m]
    t: float = 0.25 / 2  # Thickness [m]

    # Material
    E: float = 10e9  # Young's modulus [Pa]
    nu: float = 0.0  # Poisson's ratio (0 for simplicity)

    # Loading
    F_total: float = 1e6  # Total applied force [N]
    load_segment: float = 0.05  # Length of load application segment [m] (5 cm)

    # Reference mesh refinement (must fit in memory for dense solver)
    n_ref: int = 75  # Elements per side for reference solution


# Element types to study
ELEMENT_TYPES = {
    "T3": {"type": "triangle", "order": 1, "color": "#1f77b4", "marker": "o"},
    "T6": {"type": "triangle", "order": 2, "color": "#ff7f0e", "marker": "s"},
    "Q4": {"type": "quad", "order": 1, "color": "#2ca02c", "marker": "^"},
    "Q8": {"type": "quad", "order": 2, "color": "#d62728", "marker": "D"},
}

# Output directory
OUTPUT_DIR = project_root / "Examples" / "Results" / "FEM" / "convergence_square"


# =============================================================================
# Core Functions (Optimized L2 Error)
# =============================================================================


def create_square_mesh(
        n: int, elem_type: str, order: int, cfg: SquareConfig
) -> Structure_FEM:
    """
    Create a square FEM mesh using the optimized factory method.

    This uses Structure_FEM.from_rectangular_grid() which:
    - Computes connectivity vectorially with NumPy
    - Sets nodes directly (no hash-based lookup overhead)
    - Pre-allocates element DOF arrays

    Performance: ~3-5x faster than element-by-element construction.
    """
    mat = PlaneStress(E=cfg.E, nu=cfg.nu, rho=0)
    geom = Geometry2D(t=cfg.t)

    return Structure_FEM.from_rectangular_grid(
        nx=n,
        ny=n,
        length=cfg.L,
        height=cfg.L,
        element_type=elem_type,
        order=order,
        material=mat,
        geometry=geom,
    )


def solve_square(
        n: int,
        elem_type: str,
        order: int,
        loading: Literal["compression", "shear"],
        cfg: SquareConfig,
) -> Tuple[Structure_FEM, np.ndarray]:
    """
    Build, solve, and return structure with displacement vector.

    Loading is applied as pointwise forces on a 5cm segment at top left corner:
    - Compression: vertical load (Fy) at top left (x <= load_segment)
    - Shear: horizontal load (Fx) at top left (x <= load_segment)

    Returns
    -------
    St : Structure_FEM
        Solved structure (contains list_nodes for coordinate lookup)
    U : np.ndarray
        Displacement vector
    """
    St = create_square_mesh(n, elem_type, order, cfg)
    coords = np.array(St.list_nodes)
    tol = 1e-9

    # Fix bottom edge
    bottom_nodes = np.where(np.abs(coords[:, 1]) < tol)[0]
    St.fix_node(bottom_nodes.tolist(), [0, 1])

    # Find top edge nodes
    top_mask = np.abs(coords[:, 1] - cfg.L) < tol

    # Both loading cases apply forces at top left corner (x <= load_segment)
    # This ensures at least the corner node (x=0) is always included
    load_mask = top_mask & (coords[:, 0] <= cfg.load_segment + tol)
    load_nodes = np.where(load_mask)[0]
    n_load = len(load_nodes)

    if n_load == 0:
        raise ValueError(f"No nodes found in load segment. This should not happen.")

    force_per_node = cfg.F_total / n_load

    if loading == "compression":
        # Vertical load (downward)
        for node in load_nodes:
            St.load_node(int(node), 1, -force_per_node)
    else:  # shear
        # Horizontal load (rightward)
        for node in load_nodes:
            St.load_node(int(node), 0, force_per_node)

    # Solve
    St = StaticLinear.solve(St, optimized=True)

    return St, St.U.copy()


def build_node_hash_map(nodes: list, precision: int = 6) -> dict:
    """
    Build a hash map from node coordinates to node index.

    Uses rounded coordinates as keys for O(1) lookup.

    Parameters
    ----------
    nodes : list
        List of [x, y] coordinates
    precision : int
        Decimal places for rounding (default 6 = micrometer precision)

    Returns
    -------
    dict
        {(x_rounded, y_rounded): node_index}
    """
    return {
        (round(n[0], precision), round(n[1], precision)): i for i, n in enumerate(nodes)
    }


def compute_l2_error_fast(
        U_h: np.ndarray,
        nodes_h: list,
        U_ref: np.ndarray,
        nodes_ref: list,
        node_map_ref: dict,
        precision: int = 6,
) -> float:
    """
    Compute relative L2 error using hash-based node matching.

    For each node in the coarse mesh, finds the matching node in the
    reference mesh using O(1) hash lookup, then computes the L2 error.

    This is O(n_coarse) instead of O(n_coarse * n_fine) for KD-tree.

    Parameters
    ----------
    U_h : np.ndarray
        Coarse mesh displacement vector
    nodes_h : list
        Coarse mesh node coordinates
    U_ref : np.ndarray
        Reference mesh displacement vector
    nodes_ref : list
        Reference mesh node coordinates
    node_map_ref : dict
        Hash map from reference coordinates to node indices
    precision : int
        Decimal places for coordinate matching

    Returns
    -------
    float
        Relative L2 error: ||U_h - U_ref||_matched / ||U_ref||
    """
    diff_sq_sum = 0.0
    matched_count = 0

    for i, node in enumerate(nodes_h):
        key = (round(node[0], precision), round(node[1], precision))

        if key in node_map_ref:
            j = node_map_ref[key]

            # Get displacements
            ux_h, uy_h = U_h[2 * i], U_h[2 * i + 1]
            ux_ref, uy_ref = U_ref[2 * j], U_ref[2 * j + 1]

            # Accumulate squared difference
            diff_sq_sum += (ux_h - ux_ref) ** 2 + (uy_h - uy_ref) ** 2
            matched_count += 1

    if matched_count == 0:
        return np.nan

    # Compute norms
    error_norm = np.sqrt(diff_sq_sum)
    ref_norm = np.linalg.norm(U_ref)

    return error_norm / ref_norm if ref_norm > 0 else 0.0


def compute_reference_solution(
        elem_config: dict, loading: str, cfg: SquareConfig, elem_name: str
) -> Tuple[Structure_FEM, np.ndarray, dict]:
    """
    Compute reference solution with fine mesh and generate visualization.

    Returns
    -------
    St_ref : Structure_FEM
        Reference structure
    U_ref : np.ndarray
        Reference displacement vector
    node_map : dict
        Hash map for fast node lookup
    """
    n_ref = cfg.n_ref

    print(f"  Reference ({n_ref}x{n_ref})...", end=" ", flush=True)
    St_ref, U_ref = solve_square(
        n_ref, elem_config["type"], elem_config["order"], loading, cfg
    )

    # Build hash map for fast lookup (done once)
    node_map = build_node_hash_map(St_ref.list_nodes)

    # Report stats
    ux_max = np.max(np.abs(U_ref[0::2]))
    uy_max = np.max(np.abs(U_ref[1::2]))
    print(f"|ux|={ux_max:.4e}, |uy|={uy_max:.4e} ({St_ref.nb_dofs} DOFs)")

    # Generate stress and displacement plots for reference solution
    ref_dir = OUTPUT_DIR / "reference"
    viz_config = {
        "io": {
            "dir": str(ref_dir),
            "filename": f"ref_{elem_name}_{loading}",
            "figsize": (12, 10),
            "show_nodes": False,
        }
    }

    try:
        plt.close("all")
        plot_stress(St_ref, viz_config)
    except Exception as e:
        print(f"    -> Warning (stress): {e}")

    try:
        plt.close("all")
        plot_displacement(St_ref, viz_config)
        print(f"    -> Saved reference plots to {ref_dir}")
    except Exception as e:
        print(f"    -> Warning (displacement): {e}")

    return St_ref, U_ref, node_map


# =============================================================================
# Main Convergence Study
# =============================================================================


def run_convergence_study(refinements: np.ndarray, loading: str, cfg: SquareConfig):
    """
    Run convergence study for all element types.

    Parameters
    ----------
    refinements : np.ndarray
        Array of mesh refinement levels (elements per side), e.g., [2, 4, 8, 16, ...]
    loading : str
        'compression' or 'shear'
    cfg : SquareConfig
        Configuration parameters

    Returns
    -------
    dict
        Results with 'refinements', 'element_sizes', 'errors', etc.
    """
    # Compute element sizes from refinements
    element_sizes = cfg.L / refinements

    print("=" * 70)
    print(f"  MESH CONVERGENCE STUDY: {loading.upper()}")
    print("=" * 70)
    print(f"\n  Geometry: {cfg.L}m x {cfg.L}m square, t={cfg.t}m")
    print(f"  Material: E={cfg.E / 1e9:.1f} GPa, nu={cfg.nu}")
    load_dir = "Fy (vertical)" if loading == "compression" else "Fx (horizontal)"
    print(
        f"  Loading:  F={cfg.F_total / 1e6:.2f} MN {load_dir} at top left (x <= {cfg.load_segment * 100:.0f}cm)"
    )
    print(
        f"  Refinements: {len(refinements)} levels (n={refinements.min()} to {refinements.max()})"
    )

    results = {
        "refinements": refinements,
        "element_sizes": element_sizes,
        "loading": loading,
        "errors": {},
        "reference": {},
        "mesh_info": {},
    }

    for name, config in ELEMENT_TYPES.items():
        print(
            f"\n--- {name} ({config['type'].capitalize()}, Order {config['order']}) ---"
        )

        # Step 1: Compute reference solution (once per element type)
        St_ref, U_ref, node_map_ref = compute_reference_solution(
            config, loading, cfg, name
        )
        results["reference"][name] = U_ref

        # Step 2: Compute solutions at each refinement level
        errors = []
        mesh_info = []

        for n, h in zip(refinements, element_sizes):
            # Solve
            St_h, U_h = solve_square(n, config["type"], config["order"], loading, cfg)

            # Compute L2 error using hash-based matching
            rel_error = compute_l2_error_fast(
                U_h, St_h.list_nodes, U_ref, St_ref.list_nodes, node_map_ref
            )

            errors.append(rel_error)
            mesh_info.append({"n": n, "h": h, "ndofs": St_h.nb_dofs})

            print(
                f"  n={n:3d} (h={h:.4f}m, {St_h.nb_dofs:5d} DOFs): L2 error={rel_error:.4e}"
            )

        results["errors"][name] = np.array(errors)
        results["mesh_info"][name] = mesh_info

    return results


def estimate_convergence_rate(h: np.ndarray, errors: np.ndarray) -> float:
    """Estimate convergence rate from log-log slope."""
    mask = errors > 0
    if np.sum(mask) < 2:
        return np.nan

    log_h = np.log(h[mask])
    log_e = np.log(errors[mask])

    coeffs = np.polyfit(log_h, log_e, 1)
    return coeffs[0]


def plot_convergence(results: dict, output_dir: Path, cfg: SquareConfig):
    """Generate convergence plot."""
    os.makedirs(output_dir, exist_ok=True)

    h = results["element_sizes"]
    loading = results["loading"]

    fig, ax = plt.subplots(figsize=(6, 6))

    for name, config in ELEMENT_TYPES.items():
        errors = results["errors"][name]
        rate = estimate_convergence_rate(h, errors)

        ax.loglog(
            h,
            errors,
            marker=config["marker"],
            markersize=6,
            linestyle="-",
            linewidth=1.5,
            color=config["color"],
            label=f"{name} (rate $\\approx$ {rate:.2f})",
        )

    # Reference slopes
    h_ref = np.array([h.min(), h.max()])
    valid_errors = [
        results["errors"][name]
        for name in ELEMENT_TYPES
        if len(results["errors"][name]) > 0 and results["errors"][name][len(h) // 2] > 0
    ]

    if valid_errors:
        mid_error = np.median([e[len(h) // 2] for e in valid_errors])
        mid_h = h[len(h) // 2]

    ax.set_xlabel(r"Element size $h$ [m]", fontsize=12)
    ax.set_ylabel(
        r"Relative $L_2$ error $\|u_h - u_{ref}\|_2 / \|u_{ref}\|_2$", fontsize=12
    )
    ax.set_title(f"Mesh Convergence: Square under {loading.capitalize()}", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    filename = f"convergence_{loading}.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Plot saved: {output_dir / filename}")


def plot_comparison(
        results_compression: dict, results_shear: dict, output_dir: Path, cfg: SquareConfig
):
    """Generate combined comparison plot for both loading cases."""
    os.makedirs(output_dir, exist_ok=True)

    h = results_compression["element_sizes"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, results, title in zip(
            axes, [results_compression, results_shear], ["Compression", "Shear"]
    ):
        for name, config in ELEMENT_TYPES.items():
            errors = results["errors"][name]
            rate = estimate_convergence_rate(h, errors)

            ax.loglog(
                h,
                errors,
                marker=config["marker"],
                markersize=6,
                linestyle="-",
                linewidth=1.5,
                color=config["color"],
                label=f"{name} ({rate:.2f})",
            )

        # Reference slopes
        h_ref = np.array([h.min(), h.max()])
        valid_errors = [
            results["errors"][name]
            for name in ELEMENT_TYPES
            if len(results["errors"][name]) > 0
               and results["errors"][name][len(h) // 2] > 0
        ]

        if valid_errors:
            mid_error = np.median([e[len(h) // 2] for e in valid_errors])
            mid_h = h[len(h) // 2]

        ax.set_xlabel(r"Element size $h$ [m]", fontsize=12)
        ax.set_ylabel(r"Relative $L_2$ error", fontsize=12)
        ax.set_title(f"{title} Loading", fontsize=14)
        ax.legend(fontsize=9, title="Element (rate)")
        ax.grid(True, which="both", alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Comparison plot saved: {output_dir / 'convergence_comparison.png'}")


def print_summary(results_compression: dict, results_shear: dict, cfg: SquareConfig):
    """Print summary table."""
    h = results_compression["element_sizes"]

    print("\n" + "=" * 70)
    print("  SUMMARY: Convergence Rates")
    print("=" * 70)

    print(f"\n{'Element':<8} {'Compression Rate':>18} {'Shear Rate':>18}")
    print("-" * 48)

    for name in ELEMENT_TYPES:
        rate_comp = estimate_convergence_rate(h, results_compression["errors"][name])
        rate_shear = estimate_convergence_rate(h, results_shear["errors"][name])
        print(f"{name:<8} {rate_comp:>18.2f} {rate_shear:>18.2f}")

    print("-" * 48)

    # Note: With concentrated point loading, simple analytical solutions don't apply
    # The uniform compression analytical reference is kept for scale comparison only
    delta_uniform = cfg.F_total / (cfg.L * cfg.t * cfg.E) * cfg.L
    print(f"\n(Reference) Uniform 1D compression: delta = {delta_uniform * 1e6:.4f} um")
    print(
        f"(Note: Point loading creates stress concentration - larger local displacements)"
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Configuration
    cfg = SquareConfig()

    # Refinement levels: elements per side (no duplicates guaranteed)
    # Logarithmic spacing: 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64
    REFINEMENTS = np.unique(np.geomspace(1, 50, num=40).astype(int))

    # Run convergence study
    print("\n" + "=" * 70)
    print("   SQUARE FEM CONVERGENCE ANALYSIS")
    print("=" * 70)

    results_compression = run_convergence_study(REFINEMENTS, "compression", cfg)
    plot_convergence(results_compression, OUTPUT_DIR, cfg)

    results_shear = run_convergence_study(REFINEMENTS, "shear", cfg)
    plot_convergence(results_shear, OUTPUT_DIR, cfg)

    # Combined comparison plot
    plot_comparison(results_compression, results_shear, OUTPUT_DIR, cfg)

    # Summary
    print_summary(results_compression, results_shear, cfg)

    print("\n" + "=" * 70)
    print("  [OK] Convergence study complete!")
    print("=" * 70)
