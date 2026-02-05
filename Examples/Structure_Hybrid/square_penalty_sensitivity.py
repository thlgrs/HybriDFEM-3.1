"""
Penalty Coupling Sensitivity Analysis - Square Geometry
========================================================

This script analyzes the influence of the penalty stiffness parameter (alpha)
on the coupling between blocks and FEM in a square hybrid structure under compression.

Geometry: 1m x 1m square with alternating horizontal block/FEM slices (SQUARE config)
Loading: Uniform compression from top
Analysis: Parametric study varying penalty stiffness alpha

The analysis measures:
1. Displacement error at the control point compared to a reference solution
2. Force equilibrium error (unbalanced forces as % of applied load)
"""

import multiprocessing
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Examples.utils.visualization import visualize

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Library Imports ---
from Examples.utils.model_builders import (
    create_hybrid_column_slices,
    find_nodes_at_height,
    find_nodes_at_length
)
from Examples.utils.solvers import run_solver
from Examples.utils.helpers import create_config

# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_ROOT = str(project_root / "Examples" / 'Results' / 'Hybrid' / 'square_penalty_sensitivity')

# SQUARE geometry configuration from hybrid_column.py
BASE_CONFIG = {
    'geometry': {
        'width': 1.0,  # Square: 1m width
        'thickness': 0.5,  # Out-of-plane thickness (m)
        'n_slices': 20,  # Number of alternating slices
        'start_with': 'block',  # First slice type (at bottom)
        'block_slice_height': 0.05,  # Height of block slices (m) -> 20 * 0.05 = 1m total
        'fem_slice_height': 0.05,  # Height of FEM slices (m)

        # Mesh Refinement
        'nx': 20,  # Global horizontal refinement (elements/blocks wide)
        'ny_block_slice': 2,  # Block slices are 2 blocks high
        'ny_fem_slice': 2,  # FEM slices are 2 elements high

        'coupling_offset': 1e-6,  # Small gap between slices for coupling
    },
    'elements': {
        'type': 'quad',  # Quad elements for FEM
        'order': 1,  # Linear elements
    },
    'material': {
        'block': {'E': 10e9, 'nu': 0.0, 'rho': 0},  # Steel-like, nu=0 for simplicity
        'fem': {'E': 10e9, 'nu': 0.0, 'rho': 0},
    },
    'contact': {
        'kn': 10e9,  # Normal contact stiffness (N/m)
        'ks': 10e9,  # Shear contact stiffness (N/m)
        'LG': True,  # Linear geometry
        'nb_cps': 20,  # Contact points per face
    },
    'coupling': {
        'method': 'penalty',  # Penalty coupling method
        'tolerance': 0.001,  # Interface detection tolerance (m)
        'integration_order': 2,  # Gauss quadrature order
        'penalty_stiffness': 1e12,  # Default penalty stiffness (will be varied)
        'interface_orientation': 'horizontal',  # Horizontal slice interfaces
    },
    'bc': {
        'type': 'compression',  # Compression boundary condition
    },
    'loads': {
        'Fy': -1e6,  # Compressive load at top (1 MN downward)
        'Fx': 0,
    },
    'io': {
        'filename': 'square_penalty_sensitivity',
        'dir': RESULTS_ROOT,
        'show_nodes': False,
        'figsize': (8, 8),
        'scale': 100,
    }
}


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

def apply_conditions(St, config):
    """
    Apply compression boundary conditions:
    - Fix all base nodes (y=0) in all DOFs
    - Apply distributed compressive load at top
    """
    g = config['geometry']
    l_conf = config['loads']

    # Calculate total height
    n_slices = g['n_slices']
    start_block = (g['start_with'] == 'block')

    if start_block:
        n_block = (n_slices + 1) // 2
        n_fem = n_slices // 2
    else:
        n_fem = (n_slices + 1) // 2
        n_block = n_slices // 2

    total_height = (n_block * g['block_slice_height'] +
                    n_fem * g['fem_slice_height'] +
                    (n_slices - 1) * g.get('coupling_offset', 0))

    print(f"\nApplying Compression BCs")
    print(f"  Total Height ~ {total_height:.4f} m")

    tol = 1e-3

    # Find node sets
    bottom_set = set(find_nodes_at_height(St, 0.0, tolerance=tol))
    top_set = set(find_nodes_at_height(St, total_height, tolerance=tol))
    center_x_set = set(find_nodes_at_length(St, g['width'] / 2, tolerance=5e-2))

    # Fix base nodes fully
    for node in bottom_set:
        St.fix_node(node_ids=[node], dofs=[0, 1, 2])
    print(f"  -> Fixed {len(bottom_set)} base nodes")

    # Apply distributed load at top
    load_nodes = list(top_set.intersection(center_x_set))
    n_load_nodes = len(load_nodes)

    if l_conf.get('Fx', 0) != 0:
        Fx_per_node = l_conf['Fx'] / n_load_nodes
        St.load_node(node_ids=load_nodes, dofs=[0], force=Fx_per_node)

    if l_conf.get('Fy', 0) != 0:
        Fy_per_node = l_conf['Fy'] / n_load_nodes
        St.load_node(node_ids=load_nodes, dofs=[1], force=Fy_per_node)

    print(f"  -> Applied distributed load to {n_load_nodes} top nodes")
    print(f"     Fy_total = {l_conf.get('Fy', 0) / 1e3:.1f} kN")

    return St


# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

def analyze_results(St, config):
    """
    Analyze results and compute error metrics.

    Returns:
        u: Displacement magnitude at control point
        error_F: Force equilibrium error as percentage
    """
    l_conf = config['loads']

    # Expand displacement if constraint coupling was used
    if hasattr(St, 'coupling_T') and St.coupling_T is not None:
        St.U = St.coupling_T @ St.U

    # Compute reaction forces
    if not hasattr(St, 'P_r') or St.P_r is None:
        if hasattr(St, 'coupling_T') and St.coupling_T is not None:
            St.P_r = np.zeros(len(St.U), dtype=float)
            if hasattr(St, '_get_P_r_block'): St._get_P_r_block()
            if hasattr(St, '_get_P_r_fem'): St._get_P_r_fem()
            if hasattr(St, '_get_P_r_hybrid'): St._get_P_r_hybrid()
        else:
            St.get_P_r()

    # Sum reaction forces at base
    base_nodes = find_nodes_at_height(St, 0.0, tolerance=1e-3)

    Rx_total = 0.0
    Ry_total = 0.0

    for node_id in base_nodes:
        dofs = St.get_dofs_from_node(node_id)
        Rx_total += St.P_r[dofs[0]]
        Ry_total += St.P_r[dofs[1]]

    # Calculate equilibrium error
    applied_Fx = l_conf.get('Fx', 0)
    applied_Fy = l_conf.get('Fy', 0)

    residual_x = applied_Fx + Rx_total
    residual_y = applied_Fy + Ry_total
    residual_mag = np.sqrt(residual_x ** 2 + residual_y ** 2)

    load_mag = np.sqrt(applied_Fx ** 2 + applied_Fy ** 2)
    error_F = 100 * residual_mag / load_mag if load_mag > 0 else 0.0

    return St.U, error_F


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_config(config):
    """
    Run a single configuration and return results.
    This function is self-contained for multiprocessing.
    """
    name = config['io']['filename']
    alpha_val = config['coupling']['penalty_stiffness']
    print(f"Running: {name} (alpha = {alpha_val:.2e})")

    # Build model
    St = create_hybrid_column_slices(config)

    # Apply boundary conditions
    St = apply_conditions(St, config)

    # Solve
    St = run_solver(St, config)

    # Analyze
    u, error_F = analyze_results(St, config)

    return alpha_val, u, error_F


def run_configs_parallel(configs):
    """
    Run configurations in parallel using all available CPU cores.
    Returns sorted lists of alpha, displacement, and equilibrium error.
    """
    print(f"\n{'=' * 60}")
    print(f"Starting parallel execution of {len(configs)} configurations")
    print(f"{'=' * 60}")

    with multiprocessing.Pool() as pool:
        results = pool.map(run_config, configs)

    alpha, u, error_F = zip(*results)

    print(f"\nCompleted {len(configs)} analyses.")
    return list(alpha), list(u), list(error_F)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(alpha, u_error, f_error, save_folder=None, filename="penalty_sensitivity.png"):
    """
    Plot sensitivity results with logarithmic axes.

    Args:
        alpha: List of penalty stiffness values
        u_error: List of displacement errors ||u_alpha - u_ref||
        f_error: List of force equilibrium errors (%)
    """
    # Try to use LaTeX rendering
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.unicode_minus": False,
        })
    except:
        plt.rcParams.update({"text.usetex": False})
        plt.rc('font', family="DejaVu Sans")

    fig, axs = plt.subplots(2, 1, figsize=(6, 5))

    plot_style = {'marker': 'x', 'markersize': 4, 'linestyle': '-', 'linewidth': 0.9}

    # Top: Displacement error
    axs[0].plot(alpha, 100 * u_error, color='black', **plot_style)
    axs[0].set_ylabel(r'$\frac{||u_\alpha - u^*||}{||u^*||}$ (\%)')
    axs[0].set_title(r'Global Relative Displacement Error')
    axs[0].grid(True, which="both", ls="-", alpha=0.5)

    # Bottom: Force equilibrium error
    axs[1].plot(alpha, f_error, color='black', **plot_style)
    axs[1].set_ylabel(r'$\frac{||R + F_{ext}||}{||F_{ext}||}$ (\%)')
    axs[1].set_title(r'Relative Force Equilibrium Error')
    axs[1].grid(True, which="both", ls="-", alpha=0.5)

    for ax in axs.flat:
        ax.set_xlabel(r'Penalty Stiffness $\alpha$')
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()

    if save_folder:
        if not os.path.exists(save_folder):
            print(f"Creating directory: {save_folder}")
            os.makedirs(save_folder)

        full_path = os.path.join(save_folder, filename)
        plt.savefig(full_path, dpi=300)
        print(f"Figure saved to: {full_path}")

    plt.show()


# =============================================================================
# MAIN SENSITIVITY ANALYSIS
# =============================================================================

def sensitivity(alphas, u_ref, filename='penalty_sensitivity.png'):
    """
    Run sensitivity analysis across a range of penalty stiffness values.

    Args:
        alphas: Array of penalty stiffness values to test
        u_ref: Reference displacement magnitude for error calculation
        filename: Output filename for the plot
    """
    configs = []
    for alpha in alphas:
        cfg = create_config(
            BASE_CONFIG,
            f"k_{alpha:.0e}",
            coupling={'penalty_stiffness': alpha}
        )
        configs.append(cfg)

    alpha_res, u, error_f = run_configs_parallel(configs)

    # 1. Calculate the reference norm once
    norm_ref = np.linalg.norm(u_ref)

    # 2. Handle the edge case (division by zero)
    if norm_ref == 0:
        rel_errors = np.full(len(u), np.inf)
    else:
        # 3. Vectorized Subtraction (Broadcasting)
        diffs = u - u_ref

        # 4. Vectorized Norm Calculation
        diff_norms = np.linalg.norm(diffs, axis=1).squeeze()

        # 5. Vectorized Division
        rel_errors = diff_norms / norm_ref

    plot_results(alpha_res, rel_errors, error_f, save_folder=RESULTS_ROOT, filename=filename)

    return alpha_res, rel_errors, error_f


def get_reference_displacement(method='constraint'):
    """
    Run a reference solution using constraint or mortar coupling.

    Note: For alternating slice geometries with proper interface alignment,
    constraint coupling now works correctly. Use mortar if you need
    non-matching mesh support.

    Returns the displacement magnitude at the control point.
    """
    print(f"\n{'=' * 60}")
    print(f"Computing reference solution ({method.upper()} coupling)")
    print(f"{'=' * 60}")

    ref_config = create_config(BASE_CONFIG, f'reference_{method}', coupling={'method': method})

    St = create_hybrid_column_slices(ref_config)
    St = apply_conditions(St, ref_config)
    St = run_solver(St, ref_config)

    # Expand if constraint coupling was used
    if hasattr(St, 'coupling_T') and St.coupling_T is not None:
        St.U = St.coupling_T @ St.U

    visualize(St, BASE_CONFIG)

    return St.U


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Define penalty stiffness range (logarithmically spaced)
    # Typical range: from material stiffness to 100x material stiffness
    start, stop = 1e0, 1e16
    K_PEN = np.unique(np.geomspace(start, stop, num=40, dtype=float))

    # Compute reference solution
    print("\n" + "=" * 60)
    print("   SQUARE GEOMETRY PENALTY SENSITIVITY ANALYSIS")
    print("   Boundary Condition: Compression")
    print("=" * 60)

    # Get reference from constraint coupling (now works with fixed interface alignment)
    u_ref = get_reference_displacement(method='constraint')

    # Run sensitivity analysis
    sensitivity(K_PEN, u_ref, f'square_penalty_sensitivity_[{start:.0e}-{stop:.0e}].png')
