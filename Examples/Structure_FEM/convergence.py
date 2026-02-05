"""
Convergence Study for FEM Elements
===================================

This script performs a mesh convergence analysis by:
1. Computing a reference solution with a very fine mesh
2. Running simulations at various element sizes
3. Plotting relative error vs element size (log-log scale)

The convergence rate is estimated from the slope of the log-log plot.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Library Imports ---
from Core.Structures.Structure_FEM import Structure_FEM
from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Solvers.Static import StaticLinear
from Examples.utils.mesh_generation import generate_node_grid, create_triangle_elements, create_quad_elements

# --- Plot Configuration ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# =============================================================================
# Configuration
# =============================================================================

# Beam geometry
L = 2.5  # Length [m]
H = 0.2  # Height [m]
t = 0.1  # Thickness [m]

# Material (Steel)
E = 200e9  # Young's modulus [Pa]
nu = 0.3  # Poisson's ratio

# Loading
Fy = -50e3  # Tip load [N]

# Element types to study
ELEMENT_TYPES = {
    'T3': {'type': 'triangle', 'order': 1, 'color': '#1f77b4', 'marker': 'o'},
    'T6': {'type': 'triangle', 'order': 2, 'color': '#ff7f0e', 'marker': 's'},
    'Q4': {'type': 'quad', 'order': 1, 'color': '#2ca02c', 'marker': '^'},
    'Q8': {'type': 'quad', 'order': 2, 'color': '#d62728', 'marker': 'D'},
}

# Output directory
OUTPUT_DIR = project_root / "Examples" / "Results" / "FEM" / "convergence"


# =============================================================================
# Core Functions
# =============================================================================

def create_cantilever(nx: int, ny: int, elem_type: str, order: int) -> Structure_FEM:
    """Create a cantilever beam FEM structure."""
    St = Structure_FEM(fixed_dofs_per_node=False)
    mat = PlaneStress(E=E, nu=nu, rho=0)
    geom = Geometry2D(t=t)

    # Generate mesh
    nodes, nnx, nny = generate_node_grid(nx, ny, L, H, order)

    if elem_type == 'triangle':
        elements = create_triangle_elements(nodes, nnx, nx, ny, order, mat, geom)
    else:
        elements = create_quad_elements(nodes, nnx, nx, ny, order, mat, geom)

    for elem in elements:
        St.list_fes.append(elem)

    St.make_nodes()
    return St


def apply_boundary_conditions(St: Structure_FEM) -> int:
    """Apply BCs: fixed left edge, point load at tip. Returns tip node index."""
    coords = np.array(St.list_nodes)
    tol = 1e-9

    # Fix left edge (x = 0)
    left_nodes = np.where(np.abs(coords[:, 0]) < tol)[0]
    St.fix_node(left_nodes.tolist(), [0, 1])

    # Load at tip (x = L, y = H)
    tip_mask = (np.abs(coords[:, 0] - L) < tol) & (np.abs(coords[:, 1] - H) < tol)
    tip_node = np.where(tip_mask)[0][0]
    St.load_node(int(tip_node), 1, Fy)

    return int(tip_node)


def get_tip_displacement(nx: int, ny: int, elem_type: str, order: int) -> float:
    """Build, solve, and return tip vertical displacement."""
    St = create_cantilever(nx, ny, elem_type, order)
    tip_node = apply_boundary_conditions(St)
    St = StaticLinear.solve(St)

    tip_dofs = St.get_dofs_from_node(tip_node)
    return St.U[tip_dofs[1]]  # uy


def compute_reference_solution(elem_config: dict, refinement_factor: int = 200) -> float:
    """Compute reference solution with very fine mesh."""
    # Use fine mesh: at least refinement_factor elements along length
    ny_ref = max(4, refinement_factor // 10)
    nx_ref = refinement_factor

    print(f"  Computing reference: {nx_ref}x{ny_ref} mesh...", end=" ", flush=True)
    u_ref = get_tip_displacement(nx_ref, ny_ref, elem_config['type'], elem_config['order'])
    print(f"u_ref = {u_ref:.8e} m")

    return u_ref


# =============================================================================
# Main Convergence Study
# =============================================================================

def run_convergence_study(element_sizes: np.ndarray):
    """
    Run convergence study for all element types.

    Parameters
    ----------
    element_sizes : np.ndarray
        Target element sizes [m] (largest to smallest)

    Returns
    -------
    dict
        Results containing element sizes, errors, and reference solutions
    """
    print("=" * 70)
    print("  MESH CONVERGENCE STUDY")
    print("=" * 70)
    print(f"\n  Geometry: L={L}m × H={H}m × t={t}m")
    print(f"  Material: E={E / 1e9:.0f} GPa, ν={nu}")
    print(f"  Loading:  Fy={Fy / 1e3:.1f} kN at tip")
    print(f"  Element sizes: {len(element_sizes)} values ({element_sizes.max():.4f}m → {element_sizes.min():.4f}m)")

    results = {
        'element_sizes': element_sizes,
        'errors': {},
        'displacements': {},
        'reference': {},
        'mesh_info': {}
    }

    for name, config in ELEMENT_TYPES.items():
        print(f"\n--- {name} ({config['type'].capitalize()}, Order {config['order']}) ---")

        # Step 1: Compute reference solution
        u_ref = compute_reference_solution(config)
        results['reference'][name] = u_ref

        # Step 2: Compute solutions at various mesh sizes
        displacements = []
        errors = []
        mesh_info = []

        for h in element_sizes:
            # Compute mesh dimensions from element size
            ny = max(1, int(np.ceil(H / h)))
            nx = max(1, int(np.ceil(L / h)))
            h_actual = max(L / nx, H / ny)  # Actual element size

            # Solve
            u_h = get_tip_displacement(nx, ny, config['type'], config['order'])
            displacements.append(u_h)

            # Compute relative error
            rel_error = abs(u_h - u_ref) / abs(u_ref)
            errors.append(rel_error)
            mesh_info.append({'nx': nx, 'ny': ny, 'h': h_actual})

            print(f"  h={h:.4f}m ({nx:3d}×{ny:2d}): u={u_h:.6e}m, error={rel_error:.4e}")

        results['displacements'][name] = np.array(displacements)
        results['errors'][name] = np.array(errors)
        results['mesh_info'][name] = mesh_info

    return results


def estimate_convergence_rate(h: np.ndarray, errors: np.ndarray) -> float:
    """Estimate convergence rate from log-log slope."""
    # Use linear regression on log-log data (skip any zero errors)
    mask = errors > 0
    if np.sum(mask) < 2:
        return np.nan

    log_h = np.log(h[mask])
    log_e = np.log(errors[mask])

    # Linear fit: log(e) = rate * log(h) + const
    coeffs = np.polyfit(log_h, log_e, 1)
    return coeffs[0]


def plot_convergence(results: dict, output_dir: Path):
    """Generate convergence plots."""
    os.makedirs(output_dir, exist_ok=True)

    h = results['element_sizes']

    # =========================================================================
    # Plot 1: Relative Error vs Element Size (Log-Log)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, config in ELEMENT_TYPES.items():
        errors = results['errors'][name]

        # Estimate convergence rate
        rate = estimate_convergence_rate(h, errors)

        ax.loglog(h, errors,
                  marker=config['marker'], markersize=6,
                  linestyle='-', linewidth=1.5,
                  color=config['color'],
                  label=f"{name} (rate $\\approx$ {rate:.2f})")

    # Reference slopes
    h_ref = np.array([h.min(), h.max()])
    for rate, style, label in [(1, '--', '$O(h)$'), (2, '-.', '$O(h^2)$')]:
        scale = 0.5 * results['errors']['T3'][-1] / (h.min() ** rate)
        ax.loglog(h_ref, scale * h_ref ** rate, style, color='gray', alpha=0.5, label=label)

    ax.set_xlabel(r"Element size $h$ [m]", fontsize=12)
    ax.set_ylabel(r"Relative error $|u_h - u_{ref}| / |u_{ref}|$", fontsize=12)
    ax.set_title("Mesh Convergence Study: Cantilever Beam", fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()  # Finer meshes on the right

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_error.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # =========================================================================
    # Plot 2: Displacement vs Element Size
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 7))

    for name, config in ELEMENT_TYPES.items():
        displacements = results['displacements'][name]
        u_ref = results['reference'][name]

        ax.semilogx(h, displacements * 1e3,  # Convert to mm
                    marker=config['marker'], markersize=6,
                    linestyle='-', linewidth=1.5,
                    color=config['color'],
                    label=f"{name}")

        # Reference line
        ax.axhline(u_ref * 1e3, color=config['color'], linestyle=':', alpha=0.5)

    ax.set_xlabel(r"Element size $h$ [m]", fontsize=12)
    ax.set_ylabel(r"Tip displacement $u_y$ [mm]", fontsize=12)
    ax.set_title("Tip Displacement vs Mesh Size", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    fig.savefig(output_dir / "convergence_displacement.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[OK] Plots saved to: {output_dir}")


def print_summary(results: dict):
    """Print summary table."""
    print("\n" + "=" * 70)
    print("  SUMMARY: Reference Solutions and Convergence Rates")
    print("=" * 70)

    h = results['element_sizes']

    print(f"\n{'Element':<8} {'Reference u_y [mm]':>18} {'Conv. Rate':>12}")
    print("-" * 42)

    for name in ELEMENT_TYPES:
        u_ref = results['reference'][name] * 1e3  # mm
        errors = results['errors'][name]
        rate = estimate_convergence_rate(h, errors)

        print(f"{name:<8} {u_ref:>18.6f} {rate:>12.2f}")

    print("-" * 42)

    # Theoretical: Euler-Bernoulli beam deflection
    I = t * H ** 3 / 12
    delta_EB = Fy * L ** 3 / (3 * E * I)
    print(f"\nEuler-Bernoulli (analytical): {delta_EB * 1e3:.6f} mm")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Element sizes: from H (coarse) to H/40 (fine)
    ELEMENT_SIZES = np.geomspace(H, H / 40, num=15)

    # Run study
    results = run_convergence_study(ELEMENT_SIZES)

    # Plot results
    plot_convergence(results, OUTPUT_DIR)

    # Print summary
    print_summary(results)
