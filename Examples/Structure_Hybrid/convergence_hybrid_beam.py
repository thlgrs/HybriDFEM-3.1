"""
Hybrid Beam Convergence Study
=============================
Convergence analysis for hybrid beam with alternating Block and FEM slices.

Refinement parameters:
- ny: Vertical refinement (elements/blocks high)
- nx_block_slice: Horizontal refinement per block slice
- nx_fem_slice: Horizontal refinement per FEM slice

Fixed parameters:
- n_slices: 4 (alternating block-fem-block-fem)
- block_slice_width: 0.625 m
- fem_slice_width: 0.625 m
- Total length: 2.5 m
"""

import gc
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Library Imports ---
from Examples.utils.model_builders import create_hybrid_beam_slices, find_nodes_at_length, find_nodes_at_height
from Examples.utils.solvers import run_solver
from Examples.utils.helpers import create_config

# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_ROOT = str(project_root / "Examples" / 'Results' / 'Hybrid' / 'convergence_hybrid_beam')

BASE_CONFIG = {
    'geometry': {
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
    },
    'elements': {
        'type': 'quad',
        'order': 1,
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
        'tolerance': 1e-5,
        'integration_order': 3,
        'penalty_stiffness': 1e12,
        'interface_orientation': 'vertical',
    },
    'loads': {
        'Type': 'Point',
        'Fy': -100e3,
    },
    'bc': {
        'type': 'cantilever',
    },
    'solver': {
        'name': 'linear',
        'optimized': True
    },
    'io': {
        'filename': 'convergence_hybrid_beam',
        'dir': RESULTS_ROOT,
        'show_nodes': False,
        'scale': 20,
    }
}

# Element type definitions
ELEMENT_TYPES = {
    't3': {'type': 'triangle', 'order': 1},
    't6': {'type': 'triangle', 'order': 2},
    'q4': {'type': 'quad', 'order': 1},
    'q8': {'type': 'quad', 'order': 2},
}

# Coupling method definitions
COUPLING_METHODS = ['constraint', 'penalty', 'lagrange', 'mortar']


# =============================================================================
# MODEL HELPERS
# =============================================================================

def create_model(config):
    return create_hybrid_beam_slices(config)


def apply_conditions(St, config):
    """Apply boundary conditions and loads. Returns (St, control_node)."""
    g = config['geometry']
    l_conf = config['loads']

    n_slices = g['n_slices']
    start_block = (g['start_with'] == 'block')

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

    tol = 1e-3
    left_set = set(find_nodes_at_length(St, 0.0, tolerance=tol))
    right_set = set(find_nodes_at_length(St, total_length, tolerance=tol))
    top_set = set(find_nodes_at_height(St, total_height, tolerance=tol))

    # Cantilever: Fix left edge
    for node in left_set:
        St.fix_node(node_ids=[node], dofs=[0, 1, 2])

    # Load right-top corner
    nodes_to_load = list(right_set.intersection(top_set))
    control_node = nodes_to_load[0] if nodes_to_load else list(right_set)[0]

    if l_conf.get('Fy', False):
        Fy = l_conf['Fy'] / len(nodes_to_load)
        St.load_node(node_ids=nodes_to_load, dofs=[1], force=Fy)

    return St, control_node


def get_tip_displacement(config):
    """Run simulation and return tip displacement (uy) in meters."""
    try:
        St = create_model(config)
        St, control_node = apply_conditions(St, config)
        St = run_solver(St, config)

        # Expand displacement if constraint coupling
        if hasattr(St, 'coupling_T') and St.coupling_T is not None:
            St.U = St.coupling_T @ St.U

        dofs = St.get_dofs_from_node(control_node)
        return St.U[dofs[1]]
    except Exception as e:
        print(f"  Error: {e}")
        return np.nan


def get_dof_count(config):
    """Estimate DOF count for a hybrid config."""
    g = config['geometry']
    elem_type = config['elements']['type']
    elem_order = config['elements']['order']

    ny = g['ny']
    nx_block = g['nx_block_slice']
    nx_fem = g['nx_fem_slice']
    n_slices = g['n_slices']
    start_block = (g['start_with'] == 'block')

    if start_block:
        n_block_slices = (n_slices + 1) // 2
        n_fem_slices = n_slices // 2
    else:
        n_fem_slices = (n_slices + 1) // 2
        n_block_slices = n_slices // 2

    # Block nodes: 3 DOF per node (ux, uy, theta)
    # Each block slice has (nx_block + 1) * (ny + 1) nodes
    n_block_nodes = n_block_slices * (nx_block + 1) * (ny + 1)
    block_dofs = n_block_nodes * 3

    # FEM nodes: 2 DOF per node (ux, uy)
    if elem_type == 'quad':
        if elem_order == 1:
            n_fem_nodes = n_fem_slices * (nx_fem + 1) * (ny + 1)
        else:  # order 2
            n_fem_nodes = n_fem_slices * (2 * nx_fem + 1) * (2 * ny + 1)
    else:  # triangle
        if elem_order == 1:
            n_fem_nodes = n_fem_slices * (nx_fem + 1) * (ny + 1)
        else:  # order 2
            n_fem_nodes = n_fem_slices * (2 * nx_fem + 1) * (2 * ny + 1)

    fem_dofs = n_fem_nodes * 2

    return block_dofs + fem_dofs


def estimate_task_memory_mb(config):
    """Estimate memory usage in MB for a single hybrid simulation based on DOF count."""
    dofs = get_dof_count(config)
    # Rough estimate: sparse matrix storage + vectors + overhead
    # Hybrid structures have more overhead due to coupling matrices
    # Sparse K matrix: ~50 bytes per DOF (assuming ~25 non-zeros per row)
    # Coupling matrices (T, C): additional ~30 bytes per DOF
    # Vectors (U, P, etc.): ~100 bytes per DOF
    # Python overhead: ~80 MB base (higher for hybrid due to more objects)
    return 80 + (dofs * 180) / (1024 * 1024)


# =============================================================================
# CONFIG GENERATION
# =============================================================================

def generate_configs(ny_range, nx_range, element_types=None, coupling_method='constraint'):
    """
    Generate configuration grid for convergence study.

    Args:
        ny_range: Array of vertical refinement values
        nx_range: Array of horizontal refinement values (used for both block and FEM slices)
        element_types: Dict of element types. If None, uses all.
        coupling_method: Coupling method to use.

    Returns:
        dict with 'ny', 'nx' grids and 'configs' dict per element type
    """
    if element_types is None:
        element_types = ELEMENT_TYPES

    # Create 2D meshgrid (ny, nx)
    ny_grid, nx_grid = np.meshgrid(ny_range, nx_range, indexing='ij')

    def _make_config(ny, nx, elem_key, elem_params):
        name = f'{elem_key}_{coupling_method}_ny{ny}_nx{nx}'
        return create_config(
            BASE_CONFIG,
            name,
            contact={'nb_cps': int(80 // int(ny))},
            geometry={'ny': int(ny), 'nx_block_slice': int(nx), 'nx_fem_slice': int(nx)},
            elements=elem_params,
            coupling={'method': coupling_method}
        )

    vectorized_creator = np.vectorize(_make_config, otypes=[object])

    result = {
        'ny': ny_grid,
        'nx': nx_grid,
        'configs': {}
    }

    for key, params in element_types.items():
        config_matrix = vectorized_creator(ny_grid, nx_grid, key, params)
        result['configs'][key] = config_matrix

    return result


# =============================================================================
# PARALLEL EXECUTION
# =============================================================================

def _solve_single_config(args):
    """Worker function for parallel execution."""
    elem_key, idx, config = args
    disp = get_tip_displacement(config)
    dofs = get_dof_count(config)
    return (elem_key, idx, disp, dofs)


def convergence_grid(ny_range, nx_range,
                     coupling_method='constraint',
                     element_types=None,
                     output_dir=RESULTS_ROOT,
                     n_workers=None,
                     parallel=True,
                     max_memory_gb=None,
                     batch_size=None):
    """
    Run convergence study over a 2D grid of refinements (ny × nx).

    Args:
        ny_range: Array of vertical refinement values
        nx_range: Array of horizontal refinement values (same for block and FEM slices)
        coupling_method: Coupling method ('constraint', 'penalty', 'lagrange', 'mortar')
        element_types: Dict of element types. If None, uses all.
        output_dir: Directory to save results
        n_workers: Number of parallel workers. If None, auto-calculated based on memory.
        parallel: If True, run in parallel
        max_memory_gb: Maximum memory to use (GB). If None, uses 75% of available RAM.
        batch_size: Process tasks in batches of this size. If None, auto-calculated.

    Returns:
        pd.DataFrame: Results with columns [ny, nx, dofs_*, disp_*]
    """
    # Try to import psutil for memory detection
    try:
        import psutil
        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False
        print("  [Warning] psutil not installed. Using conservative memory defaults.")
        print("            Install with: pip install psutil")

    if element_types is None:
        element_types = ELEMENT_TYPES

    print("=" * 60)
    print(f"HYBRID BEAM CONVERGENCE STUDY")
    print(f"Coupling Method: {coupling_method.upper()}")
    print("=" * 60)

    # ========================================================================
    # Phase 1: Generate Configs
    # ========================================================================
    print("\n--- Phase 1: Generating Configurations ---")
    data_grid = generate_configs(ny_range, nx_range, element_types, coupling_method)

    ny_grid = data_grid['ny']
    nx_grid = data_grid['nx']

    n_elements = len(element_types)
    total_sims = ny_grid.size * n_elements

    print(f"  Grid shape: {ny_grid.shape} (ny × nx)")
    print(f"  ny range: {ny_range.min()} → {ny_range.max()} ({len(ny_range)} values)")
    print(f"  nx range: {nx_range.min()} → {nx_range.max()} ({len(nx_range)} values)")
    print(f"  Total simulations: {total_sims} ({n_elements} element types)")

    # ========================================================================
    # Phase 2: Run Simulations
    # ========================================================================
    # Build flat task list
    tasks = []
    for elem_key, config_matrix in data_grid['configs'].items():
        flat_configs = config_matrix.flatten()
        for idx, config in enumerate(flat_configs):
            tasks.append((elem_key, idx, config))

    # Initialize results storage
    results = {key: {'disp': np.zeros(ny_grid.size), 'dofs': np.zeros(ny_grid.size, dtype=int)}
               for key in element_types.keys()}

    if parallel:
        # --- Memory-aware parallel execution ---
        # Get available memory
        if max_memory_gb is None:
            if HAS_PSUTIL:
                available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
                max_memory_gb = available_memory_gb * 0.75  # Use 75% of available RAM
            else:
                # Conservative fallback: assume 8 GB available
                max_memory_gb = 8.0

        # Estimate peak memory per task (use largest mesh as conservative estimate)
        largest_config = tasks[-1][2]  # Last task likely has largest mesh
        estimated_memory_per_task_mb = estimate_task_memory_mb(largest_config)
        estimated_memory_per_task_gb = estimated_memory_per_task_mb / 1024

        # Calculate safe number of workers
        if n_workers is None:
            cpu_count = multiprocessing.cpu_count()
            memory_limited_workers = max(1, int(max_memory_gb / estimated_memory_per_task_gb))
            n_workers = min(cpu_count - 1, memory_limited_workers)
            n_workers = max(1, n_workers)  # At least 1 worker

        # Calculate batch size if not specified
        if batch_size is None:
            # Process in batches to allow garbage collection between batches
            batch_size = max(n_workers * 4, 50)  # At least 4 tasks per worker per batch

        print(f"\n--- Phase 2: Running Simulations (Memory-Aware) ---")
        print(f"  Available memory: {max_memory_gb:.1f} GB")
        print(f"  Est. memory per task: {estimated_memory_per_task_mb:.1f} MB")
        print(f"  Workers: {n_workers} (CPU: {multiprocessing.cpu_count()})")
        print(f"  Batch size: {batch_size}")

        completed = 0
        n_batches = (len(tasks) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_solve_single_config, task) for task in batch_tasks]

                for future in as_completed(futures):
                    elem_key, idx, disp, dofs = future.result()
                    results[elem_key]['disp'][idx] = disp
                    results[elem_key]['dofs'][idx] = dofs

                    completed += 1
                    if completed % 20 == 0 or completed == total_sims:
                        pct = completed / total_sims * 100
                        print(f"  Progress: {completed}/{total_sims} ({pct:.1f}%)", flush=True)

            # Explicit garbage collection between batches
            gc.collect()
    else:
        print(f"\n--- Phase 2: Running Simulations (serial mode) ---")

        for idx, task in enumerate(tasks):
            elem_key, task_idx, disp, dofs = _solve_single_config(task)
            results[elem_key]['disp'][task_idx] = disp
            results[elem_key]['dofs'][task_idx] = dofs

            if (idx + 1) % 20 == 0 or (idx + 1) == total_sims:
                pct = (idx + 1) / total_sims * 100
                print(f"  Progress: {idx + 1}/{total_sims} ({pct:.1f}%)", flush=True)

    print(f"  All simulations complete!")

    # ========================================================================
    # Phase 3: Save Results to CSV
    # ========================================================================
    print("\n--- Phase 3: Saving Results to CSV ---")
    os.makedirs(f"{output_dir}/{coupling_method}", exist_ok=True)

    # Flatten grids and build DataFrame
    rows = []
    flat_ny = ny_grid.flatten()
    flat_nx = nx_grid.flatten()

    for idx in range(ny_grid.size):
        row = {
            'ny': flat_ny[idx],
            'nx': flat_nx[idx],
        }
        for elem_key in element_types.keys():
            row[f'dofs_{elem_key}'] = results[elem_key]['dofs'][idx]
            row[f'disp_{elem_key}'] = results[elem_key]['disp'][idx]
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/{coupling_method}/convergence_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> Results saved to: {csv_path}")

    # ========================================================================
    # Phase 4: Generate Plots
    # ========================================================================
    print("\n--- Phase 4: Generating Plots ---")
    plot_convergence(df, output_dir, coupling_method, element_types)

    print("\n" + "=" * 60)
    print("[OK] Convergence study complete!")
    print("=" * 60)

    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_convergence(df, output_dir=RESULTS_ROOT, coupling_method='constraint', element_types=None):
    """Generate convergence plots from DataFrame."""
    if element_types is None:
        disp_cols = [col for col in df.columns if col.startswith('disp_')]
        element_types = {col.replace('disp_', ''): {} for col in disp_cols}

    os.makedirs(f"{output_dir}/{coupling_method}", exist_ok=True)

    elem_colors = {
        't3': '#1f77b4', 't6': '#ff7f0e',
        'q4': '#2ca02c', 'q8': '#d62728', 'q9': '#9467bd'
    }
    elem_labels = {
        't3': 'T3 (3-node tri)', 't6': 'T6 (6-node tri)',
        'q4': 'Q4 (4-node quad)', 'q8': 'Q8 (8-node quad)', 'q9': 'Q9 (9-node quad)'
    }

    # ========================================================================
    # Plot 1: Displacement vs DOFs (scatter)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for elem_key in element_types.keys():
        c = elem_colors.get(elem_key, 'black')
        ax.scatter(df[f'dofs_{elem_key}'], df[f'disp_{elem_key}'],
                   color=c, alpha=0.6, s=25, label=elem_labels.get(elem_key, elem_key))

    ax.set_xlabel(r"Number of DOFs", fontsize=12)
    ax.set_ylabel(r"Tip Displacement $u_y$ [m]", fontsize=12)
    # ax.set_title(f"Convergence: {coupling_method.capitalize()} Coupling")
    ax.set_xscale('log')
    ax.legend(title="Element Type", fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(f"{output_dir}/{coupling_method}/disp_vs_dofs.png", dpi=300)
    plt.close(fig)

    # ========================================================================
    # Plot 2: Displacement vs ny (for each nx, with lines)
    # ========================================================================
    nx_values = np.sort(df['nx'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(nx_values)))

    for elem_key in element_types.keys():
        c = elem_colors.get(elem_key, 'black')
        for i, nx in enumerate(nx_values):
            df_nx = df[df['nx'] == nx].sort_values('ny')
            label = elem_labels.get(elem_key, elem_key) if i == 0 else None
            ax.plot(df_nx['ny'], df_nx[f'disp_{elem_key}'],
                    marker='o', markersize=4, linestyle='-', linewidth=1,
                    color=c, alpha=0.15 + 0.85 * i / len(nx_values), label=label)

    ax.set_xlabel(r"Vertical refinement $n_y$", fontsize=14)
    ax.set_ylabel(r"Tip Displacement $u_y$ [m]", fontsize=14)
    # ax.set_title(f"Convergence vs $n_y$ ({coupling_method.capitalize()} Coupling)")
    ax.legend(title="Element Type", fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/{coupling_method}/disp_vs_ny.png", dpi=300)
    plt.close(fig)

    # ========================================================================
    # Plot 3: Displacement vs nx (for each ny, with lines)
    # ========================================================================
    ny_values = np.sort(df['ny'].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(ny_values)))

    for elem_key in element_types.keys():
        c = elem_colors.get(elem_key, 'black')
        for i, ny in enumerate(ny_values):
            df_ny = df[df['ny'] == ny].sort_values('nx')
            label = elem_labels.get(elem_key, elem_key) if i == 0 else None
            ax.plot(df_ny['nx'], df_ny[f'disp_{elem_key}'],
                    marker='o', markersize=4, linestyle='-', linewidth=1,
                    color=c, alpha=0.15 + 0.85 * i / len(ny_values), label=label)

    ax.set_xlabel(r"Horizontal refinement $n_x$", fontsize=14)
    ax.set_ylabel(r"Tip Displacement $u_y$ [m]", fontsize=14)
    # ax.set_title(f"Convergence vs $n_x$ ({coupling_method.capitalize()} Coupling)")
    ax.legend(title="Element Type", fontsize=16)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/{coupling_method}/disp_vs_nx.png", dpi=300)
    plt.close(fig)

    print(f"  -> Plots saved to: {output_dir}/{coupling_method}/")


def plot_convergence_from_csv(csv_path=None, coupling_method='constraint', output_dir=RESULTS_ROOT):
    """Load and plot from saved CSV."""
    if csv_path is None:
        csv_path = f"{RESULTS_ROOT}/{coupling_method}/convergence_results.csv"

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    plot_convergence(df, output_dir, coupling_method)

    print("\nPlotting complete!")
    return df


# =============================================================================
# MAIN
# =============================================================================

# Refinement ranges

NX_RANGE = np.flip(np.unique(np.geomspace(2, 40, num=20, dtype=int)))
NY_RANGE = np.flip(np.unique(np.geomspace(2, 20, num=10, dtype=int)))

if __name__ == "__main__":
    plot_convergence_from_csv()
    """# Run convergence study for constraint coupling
    df = convergence_grid(
        NY_RANGE, NX_RANGE,
        coupling_method='constraint'
    )

    # To run other coupling methods:
    df_penalty = convergence_grid(NY_RANGE, NX_RANGE, coupling_method='penalty')
    df_lagrange = convergence_grid(NY_RANGE, NX_RANGE, coupling_method='lagrange')
    df_mortar = convergence_grid(NY_RANGE, NX_RANGE, coupling_method='mortar')

    # Or reload and replot:
    # df = plot_convergence_from_csv(coupling_method='constraint')"""
