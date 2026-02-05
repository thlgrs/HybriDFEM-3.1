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
from Examples.utils.helpers import create_config
from Examples.utils.model_builders import create_fem_column as create_model
from Examples.utils.solvers import run_solver

RESULTS_ROOT = str(project_root / "Examples" / 'convergence' / 'Results' / 'FEM' / 'cantilever')
from Examples.Structure_FEM.cantilever import BASE_CONFIG, apply_conditions


def refinement_configs_vectorized(refinements):
    refinements_x = refinements[0]
    refinements_y = refinements[1]

    ny_grid, nx_grid = np.meshgrid(refinements_y, refinements_x, indexing='ij')

    # Define the element parameters
    element_types = {
        't3': {'type': 'triangle', 'order': 1},
        't6': {'type': 'triangle', 'order': 2},
        'q4': {'type': 'quad', 'order': 1},
        'q8': {'type': 'quad', 'order': 2},
        'q9': {'type': 'quad', 'order': 2, 'prefer_quad9': True},
    }

    def _make_single_config(nx, ny, elem_key, elem_params):
        name = f'{elem_key}_{nx}x{ny}'
        return create_config(
            BASE_CONFIG,
            name,
            geometry={'nx': nx, 'ny': ny},
            elements=elem_params
        )

    vectorized_creator = np.vectorize(_make_single_config, otypes=[object])
    result = {
        'nx': nx_grid,
        'ny': ny_grid,
        'configs': {}
    }
    for key, params in element_types.items():
        config_matrix = vectorized_creator(nx_grid, ny_grid, key, params)
        result['configs'][key] = config_matrix
    return result


def tip_disps_rel_error(config):
    St = create_model(config)
    St, tip_nodes = apply_conditions(St, config)
    St = run_solver(St, config)
    if tip_nodes:
        dof_y = St.get_dofs_from_node(tip_nodes[0])[1]
        return ((St.U[dof_y] + 0.0195989) / -0.0195989) * 100
    return np.nan


def tip_displacement(config):
    """Run simulation and return tip displacement (uy) in meters."""
    St = create_model(config)
    St, tip_nodes = apply_conditions(St, config)
    St = run_solver(St, config)
    if tip_nodes:
        dof_y = St.get_dofs_from_node(tip_nodes[0])[1]
        return St.U[dof_y]
    return np.nan


def _solve_single_config(args):
    """Worker function for parallel execution. Returns (key, i, j, disp, dofs)."""
    elem_key, i, j, config = args
    disp = tip_displacement(config)
    dofs = get_dof_count(config)
    return (elem_key, i, j, disp, dofs)


def convergence_vectorized(refinements_x, refinements_y, output_dir=RESULTS_ROOT):
    """
    Vectorized convergence study.
    Args:
        refinements_x: array of nx values
        refinements_y: array of ny values
        output_dir: string path for saving plots
    """
    print("--- Phase 1: Generating Config Matrices ---")
    data_grid = refinement_configs_vectorized((refinements_x, refinements_y))

    nx_grid = data_grid['nx']  # Shape: (n_ny, n_nx)
    ny_grid = data_grid['ny']  # Shape: (n_ny, n_nx)

    vectorized_solver = np.vectorize(tip_disps_rel_error, otypes=[float])

    # Dictionary to store the result matrices (deflections_error)
    # Structure: {'t3': Matrix(floats), 'q4': Matrix(floats), ...}
    results_data = {}

    print("--- Phase 2: Running Simulations (Vectorized) ---")
    for key, config_matrix in data_grid['configs'].items():
        print(f"Processing matrix for: {key} (Shape: {config_matrix.shape})")

        # MAGIC LINE: Runs the simulation for every coordinate in the grid at once
        results_data[key] = vectorized_solver(config_matrix)

    # --- Save results to CSV for later analysis with pandas ---
    print("\n--- Saving Results to CSV ---")
    os.makedirs(f"{output_dir}/convergence", exist_ok=True)

    # Flatten the grid data and create a DataFrame
    rows = []
    for i in range(nx_grid.shape[0]):
        for j in range(nx_grid.shape[1]):
            row = {
                'nx': nx_grid[i, j],
                'ny': ny_grid[i, j],
            }
            # Add error for each element type
            for elem_key, error_matrix in results_data.items():
                row[f'error_{elem_key}'] = error_matrix[i, j]
            rows.append(row)

    df_results = pd.DataFrame(rows)
    csv_path = f"{output_dir}/convergence/convergence_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  -> Results saved to: {csv_path}")

    print("\n--- Phase 3: Generating Plots ---")

    elem_colors = {
        't3': '#1f77b4', 't6': '#ff7f0e',
        'q4': '#2ca02c', 'q8': '#d62728', 'q9': '#9467bd'
    }

    # -- Global Summary Setup --
    fig_sum, ax_sum = plt.subplots(figsize=(10, 6))

    for key, error_matrix in results_data.items():
        # Get the color
        c = elem_colors.get(key, 'black')

        # -- Individual Plot --
        fig, ax = plt.subplots(figsize=(10, 6))

        # PLOTTING TRICK:
        # Matplotlib plots columns as series. Our matrices are (ny, nx).
        # Rows are fixed 'ny' (lines we want). Columns are 'nx' (x-axis).
        # Transpose (.T) so that:
        #   X-axis input is (n_nx, n_ny) -> columns are distinct x-series
        #   Y-axis input is (n_nx, n_ny) -> columns are distinct y-series
        # This one command plots ALL lines for this element type.

        # We use the colors map to color lines by vertical refinement
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(refinements_y)))

        # Plotting the whole matrix at once
        # Note: We iterate manually here only to apply the gradient colors per line easily
        # but we use the pre-computed rows directly.
        for i in range(error_matrix.shape[0]):
            ax.plot(nx_grid[i, :], error_matrix[i, :],
                    marker='o', markersize=4, linestyle='-', linewidth=1.5,
                    color=colors[i], label=f'$n_y$ = {ny_grid[i, 0]}')

        ax.set_xlabel(r"Horizontal refinement $n_x$")
        ax.set_ylabel(r"Relative error $\frac{u_y-u_{ref}}{u_{ref}}$ [\%]")
        ax.legend(title="Vertical Refinement", fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.savefig(f"{output_dir}/convergence/convergence{key}.png", dpi=300)
        plt.close(fig)

        # -- Add to Summary Plot --
        # For the summary, we plot the matrix transpose directly to get all lines
        # We set label only for the first line
        ax_sum.plot(nx_grid.T, error_matrix.T,
                    color=c, alpha=0.6, linewidth=1.0, marker='.', markersize=2)

        # Add a dummy line for the legend
        ax_sum.plot([], [], color=c, label=key)

    # Finalize Summary
    ax_sum.set_title("Global Convergence Comparison (All Element Types)")
    ax_sum.set_xlabel(r"Horizontal refinement $n_x$")
    ax_sum.set_ylabel(r"Relative error $\frac{u_y - u_{ref}}{u_{ref}}$ [\%]")
    ax_sum.legend(title="Element Type")
    ax_sum.grid(True, alpha=0.3)
    ax_sum.invert_yaxis()
    ax_sum.figure.savefig(f"{output_dir}/convergence/comparison.png", dpi=300)
    plt.close(fig_sum)


def plot_convergence_from_csv(csv_path=None, output_dir=RESULTS_ROOT, element_types=None):
    """
    Load convergence results from CSV and generate plots.

    Args:
        csv_path: Path to the CSV file. If None, uses default location.
        output_dir: Directory to save the generated plots.
        element_types: List of element types to plot (e.g., ['t3', 't6', 'q4']).
                      If None, plots all available element types.

    Returns:
        pd.DataFrame: The loaded convergence data.

    Example:
        >>> df = plot_convergence_from_csv()
        >>> df = plot_convergence_from_csv(element_types=['t3', 'q4'])
    """
    # Default path
    if csv_path is None:
        csv_path = f"{RESULTS_ROOT}/convergence/convergence_results.csv"

    # Load data
    print(f"Loading convergence data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Extract available element types from column names
    error_cols = [col for col in df.columns if col.startswith('error_')]
    available_types = [col.replace('error_', '') for col in error_cols]

    # Filter element types if specified
    if element_types is None:
        element_types = available_types
    else:
        # Validate requested types
        invalid = set(element_types) - set(available_types)
        if invalid:
            print(f"Warning: Element types {invalid} not found in data. Available: {available_types}")
        element_types = [t for t in element_types if t in available_types]

    print(f"Plotting element types: {element_types}")

    # Get unique refinement values
    nx_values = np.sort(df['nx'].unique())
    ny_values = np.sort(df['ny'].unique())

    # Setup output directory
    os.makedirs(f"{output_dir}/convergence", exist_ok=True)

    # Color schemes
    elem_colors = {
        't3': '#1f77b4', 't6': '#ff7f0e',
        'q4': '#2ca02c', 'q8': '#d62728', 'q9': '#9467bd'
    }

    # --- Individual plots per element type ---
    for elem_key in element_types:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(ny_values)))

        for i, ny in enumerate(ny_values):
            # Filter data for this ny value
            df_ny = df[df['ny'] == ny].sort_values('nx')
            ax.plot(df_ny['nx'], df_ny[f'error_{elem_key}'],
                    marker='o', markersize=4, linestyle='-', linewidth=1.5,
                    color=colors[i], label=f'$n_y$ = {ny}')

        ax.set_xlabel(r"Horizontal refinement $n_x$")
        ax.set_ylabel(r"Relative error $\frac{u_y - u_{ref}}{u_{ref}}$ [\%]")
        ax.set_title(f"Convergence Study: {elem_key.upper()}")
        ax.legend(title="Vertical Refinement", fontsize=9)
        ax.grid(True, alpha=0.3)

        plot_path = f"{output_dir}/convergence/convergence_{elem_key}.png"
        fig.savefig(plot_path, dpi=300)
        print(f"  -> Saved: {plot_path}")
        plt.close(fig)

    # --- Summary comparison plot ---
    fig_sum, ax_sum = plt.subplots(figsize=(12, 8))

    for elem_key in element_types:
        c = elem_colors.get(elem_key, 'black')

        # Plot all lines for this element type
        for ny in ny_values:
            df_ny = df[df['ny'] == ny].sort_values('nx')
            ax_sum.plot(df_ny['nx'], df_ny[f'error_{elem_key}'],
                        color=c, alpha=0.6, linewidth=1.0, marker='.', markersize=2)

        # Legend entry (dummy line)
        ax_sum.plot([], [], color=c, label=elem_key.upper(), linewidth=2)

    ax_sum.set_title("Global Convergence Comparison (All Element Types)")
    ax_sum.set_xlabel(r"Horizontal refinement $n_x$")
    ax_sum.set_ylabel(r"Relative error $\frac{u_y - u_{ref}}{u_{ref}}$ [\%]")
    ax_sum.legend(title="Element Type")
    ax_sum.grid(True, alpha=0.3)
    ax_sum.invert_yaxis()

    summary_path = f"{output_dir}/convergence/comparison.png"
    fig_sum.savefig(summary_path, dpi=300)
    print(f"  -> Saved: {summary_path}")
    plt.close(fig_sum)

    print("\nPlotting complete!")
    return df


def get_dof_count(config):
    nx = config['geometry']['nx']
    ny = config['geometry']['ny']
    elem_type = config['elements']['type']
    elem_order = config['elements']['order']

    # Check for Quad9 special case
    prefer_quad9 = config['elements'].get('prefer_quad9', False)

    if elem_type == 'quad':
        if elem_order == 1:  # Q4: Bilinear
            n_nodes = (nx + 1) * (ny + 1)
        elif elem_order == 2:  # Q8 or Q9
            if prefer_quad9:  # Q9: Full quadratic with center nodes
                n_nodes = (2 * nx + 1) * (2 * ny + 1)
            else:  # Q8: Serendipity (no center nodes)
                # Total grid nodes minus center nodes
                n_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
        else:
            # Fallback for unexpected orders
            n_nodes = (nx + 1) * (ny + 1)
    else:  # Triangle elements
        if elem_order == 1:  # T3: Linear
            # Two triangles per quad cell, sharing nodes
            n_nodes = (nx + 1) * (ny + 1)
        elif elem_order == 2:  # T6: Quadratic
            # Quadratic mesh needs mid-side nodes
            n_nodes = (2 * nx + 1) * (2 * ny + 1)
        else:
            # Fallback
            n_nodes = (nx + 1) * (ny + 1)

    return n_nodes * 2  # 2 DOF per node (ux, uy)


def refinement_configs_by_element_size(element_sizes, beam_config=None):
    import warnings

    # Extract geometry from config
    if beam_config is None:
        x_dim = BASE_CONFIG['geometry']['x_dim']
        y_dim = BASE_CONFIG['geometry']['y_dim']
    else:
        x_dim = beam_config['x_dim']
        y_dim = beam_config['y_dim']

    # Convert to numpy array if needed
    element_sizes = np.asarray(element_sizes)

    # Calculate nx, ny for each target element size
    ny_values = np.ceil(y_dim / element_sizes).astype(int)
    nx_values = np.ceil(x_dim / element_sizes).astype(int)

    # Ensure minimum of 1 element in each direction
    ny_values = np.maximum(ny_values, 1)
    nx_values = np.maximum(nx_values, 1)

    # Calculate actual element sizes achieved (may differ from target due to ceiling)
    h_actual_y = y_dim / ny_values
    h_actual_x = x_dim / nx_values

    # Define element types (same as in refinement_configs_vectorized)
    element_types = {
        't3': {'type': 'triangle', 'order': 1},
        't6': {'type': 'triangle', 'order': 2},
        'q4': {'type': 'quad', 'order': 1},
        'q8': {'type': 'quad', 'order': 2},
        'q9': {'type': 'quad', 'order': 2, 'prefer_quad9': True},
    }

    # Generate configs for each element type
    configs = {}
    max_dof_warning_threshold = 100000

    for key, params in element_types.items():
        config_list = []
        for i, (nx, ny, h) in enumerate(zip(nx_values, ny_values, element_sizes)):
            # Create config name with element size
            name = f'{key}_h{h:.4f}'

            # Create the configuration
            config = create_config(
                BASE_CONFIG,
                name,
                geometry={'nx': int(nx), 'ny': int(ny)},
                elements=params
            )

            # Check DOF count and warn if excessive
            estimated_dofs = get_dof_count(config)
            if estimated_dofs > max_dof_warning_threshold:
                warnings.warn(
                    f"Large mesh detected for {name}: nx={nx}, ny={ny}, "
                    f"estimated DOFs={estimated_dofs:,}. This may be slow.",
                    UserWarning
                )

            config_list.append(config)

        configs[key] = np.array(config_list, dtype=object)

    return {
        'element_sizes': element_sizes,
        'nx': nx_values,
        'ny': ny_values,
        'h_actual_x': h_actual_x,
        'h_actual_y': h_actual_y,
        'configs': configs
    }


def plot_convergence_h(element_sizes, results, dof_counts, output_dir):
    # Create output directory
    os.makedirs(f"{output_dir}/convergence_h", exist_ok=True)

    # Define consistent colors for element types
    elem_colors = {
        't3': '#1f77b4',  # Blue
        't6': '#ff7f0e',  # Orange
        'q4': '#2ca02c',  # Green
        'q8': '#d62728',  # Red
        'q9': '#9467bd'  # Purple
    }

    elem_labels = {
        't3': 'T3 (3-node triangle)',
        't6': 'T6 (6-node triangle)',
        'q4': 'Q4 (4-node quad)',
        'q8': 'Q8 (8-node quad)',
        'q9': 'Q9 (9-node quad)'
    }

    # ========================================================================
    # Plot 1: Individual Element Deflection Plots
    # ========================================================================
    for key, deflections_error in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot FEM results
        ax.plot(element_sizes, deflections_error,
                marker='o', markersize=6, linestyle='-', linewidth=2,
                color=elem_colors[key], label='FEM')

        ax.set_title(f"Convergence: {elem_labels[key]}", fontsize=14, fontweight='bold')
        ax.set_xlabel(r"Element Size $h$ [m]", fontsize=12)
        ax.set_ylabel(r"Relative error $\frac{u_y - u_{ref}}{u_{ref}}$ [\%]", fontsize=12)
        ax.set_xscale('log')
        ax.invert_xaxis()
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        fig.savefig(f"{output_dir}/convergence_h/{key}_deflection.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ========================================================================
    # Plot 2: Global Comparison (All Elements)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, deflections_error in results.items():
        ax.plot(element_sizes, deflections_error,
                marker='o', markersize=5, linestyle='-', linewidth=2,
                color=elem_colors[key], label=elem_labels[key])

    # ax.set_title("Convergence Comparison: All Element Types", fontsize=14, fontweight='bold')
    ax.set_xlabel(r"Element Size $h$ [m]", fontsize=12)
    ax.set_ylabel(r"Relative error $\frac{u_y - u_{ref}}{u_{ref}}$ [\%]", fontsize=12)
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend(title="Element Type", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(f"{output_dir}/convergence_h/comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ========================================================================
    # Plot 3: Global Comparison by DOF Count
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, deflections_error in results.items():
        dofs = dof_counts[key]
        ax.plot(dofs, deflections_error,
                marker='o', markersize=5, linestyle='-', linewidth=2,
                color=elem_colors[key], label=elem_labels[key])

    ax.set_xlabel(r"Number of DOFs", fontsize=12)
    ax.set_ylabel(r"Relative error $\frac{u_y - u_{ref}}{u_{ref}}$ [\%]", fontsize=12)
    ax.set_xscale('log')
    ax.legend(title="Element Type", fontsize=10, title_fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(f"{output_dir}/convergence_h/comparison_dofs.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  -> Plots saved to: {output_dir}/convergence_h/")


def estimate_task_memory_mb(config):
    """Estimate memory usage in MB for a single FEM simulation based on DOF count."""
    dofs = get_dof_count(config)
    # Rough estimate: sparse matrix storage + vectors + overhead
    # Sparse K matrix: ~50 bytes per DOF (assuming ~25 non-zeros per row, 8 bytes each)
    # Vectors (U, P, etc.): ~100 bytes per DOF
    # Python overhead: ~50 MB base
    return 50 + (dofs * 150) / (1024 * 1024)


def convergence_grid(nx_range, ny_range, output_dir=RESULTS_ROOT, n_workers=None, parallel=True,
                     max_memory_gb=None, batch_size=None):
    """
    Run convergence study over a 2D grid of mesh refinements (nx × ny).

    Stores raw displacement and DOF counts for each element type,
    enabling flexible post-processing with pandas.

    Args:
        nx_range: Array of horizontal refinement values
        ny_range: Array of vertical refinement values
        output_dir: Directory to save results
        n_workers: Number of parallel workers. If None, auto-calculated based on memory.
        parallel: If True, run simulations in parallel. Set to False for debugging.
        max_memory_gb: Maximum memory to use (GB). If None, uses 75% of available RAM.
        batch_size: Process tasks in batches of this size. If None, auto-calculated.

    Returns:
        pd.DataFrame: Results with columns [nx, ny, dofs_*, disp_*] for each element type
    """
    try:
        import psutil
        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False
        print("  [Warning] psutil not installed. Using conservative memory defaults.")
        print("            Install with: pip install psutil")

    print("=" * 60)
    print("CONVERGENCE STUDY: Grid-Based (nx × ny)")
    print("=" * 60)

    # ========================================================================
    # Phase 1: Generate Config Matrices
    # ========================================================================
    print("\n--- Phase 1: Generating Config Matrices ---")
    data_grid = refinement_configs_vectorized((nx_range, ny_range))

    nx_grid = data_grid['nx']  # Shape: (n_ny, n_nx)
    ny_grid = data_grid['ny']  # Shape: (n_ny, n_nx)

    n_elements = len(data_grid['configs'])
    total_sims = nx_grid.size * n_elements

    print(f"  Grid shape: {nx_grid.shape} (ny × nx)")
    print(f"  nx range: {nx_range.min()} → {nx_range.max()} ({len(nx_range)} values)")
    print(f"  ny range: {ny_range.min()} → {ny_range.max()} ({len(ny_range)} values)")
    print(f"  Total configs per element: {nx_grid.size}")
    print(f"  Total simulations: {total_sims} ({n_elements} element types)")

    # ========================================================================
    # Phase 2: Run Simulations
    # ========================================================================
    # Build flat list of all tasks: (elem_key, i, j, config)
    tasks = []
    for elem_key, config_matrix in data_grid['configs'].items():
        for i in range(config_matrix.shape[0]):
            for j in range(config_matrix.shape[1]):
                tasks.append((elem_key, i, j, config_matrix[i, j]))

    # Initialize result matrices
    results_disp = {key: np.zeros(nx_grid.shape) for key in data_grid['configs'].keys()}
    results_dofs = {key: np.zeros(nx_grid.shape, dtype=int) for key in data_grid['configs'].keys()}

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
        largest_config = tasks[-1][3]  # Last task likely has largest mesh
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
                    elem_key, i, j, disp, dofs = future.result()
                    results_disp[elem_key][i, j] = disp
                    results_dofs[elem_key][i, j] = dofs

                    completed += 1
                    if completed % 50 == 0 or completed == total_sims:
                        pct = completed / total_sims * 100
                        print(f"  Progress: {completed}/{total_sims} ({pct:.1f}%)", flush=True)

            # Explicit garbage collection between batches
            gc.collect()
    else:
        # --- Serial execution (for debugging) ---
        print(f"\n--- Phase 2: Running Simulations (serial mode) ---")

        for idx, task in enumerate(tasks):
            elem_key, i, j, disp, dofs = _solve_single_config(task)
            results_disp[elem_key][i, j] = disp
            results_dofs[elem_key][i, j] = dofs

            if (idx + 1) % 50 == 0 or (idx + 1) == total_sims:
                pct = (idx + 1) / total_sims * 100
                print(f"  Progress: {idx + 1}/{total_sims} ({pct:.1f}%)", flush=True)

    print(f"  All simulations complete!")

    # ========================================================================
    # Phase 3: Save Results to CSV
    # ========================================================================
    print("\n--- Phase 3: Saving Results to CSV ---")
    os.makedirs(f"{output_dir}/convergence_grid", exist_ok=True)

    # Flatten 2D grids into rows
    rows = []
    for i in range(nx_grid.shape[0]):
        for j in range(nx_grid.shape[1]):
            row = {
                'nx': nx_grid[i, j],
                'ny': ny_grid[i, j],
            }
            # Add DOF count and displacement for each element type
            for elem_key in results_disp.keys():
                row[f'dofs_{elem_key}'] = results_dofs[elem_key][i, j]
                row[f'disp_{elem_key}'] = results_disp[elem_key][i, j]
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = f"{output_dir}/convergence_grid/convergence_grid_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> Results saved to: {csv_path}")

    # ========================================================================
    # Phase 4: Generate Default Plots
    # ========================================================================
    print("\n--- Phase 4: Generating Plots ---")
    plot_convergence_grid(df, output_dir)

    print("\n" + "=" * 60)
    print("[OK] Grid convergence study complete!")
    print("=" * 60)

    return df


def plot_convergence_grid(df, output_dir=RESULTS_ROOT, element_types=None):
    """
    Generate convergence plots from grid results DataFrame.

    Args:
        df: DataFrame with columns [nx, ny, dofs_*, disp_*]
        output_dir: Directory to save plots
        element_types: List of element types to plot. If None, plots all.
    """
    os.makedirs(f"{output_dir}/convergence_grid", exist_ok=True)

    # Extract available element types
    disp_cols = [col for col in df.columns if col.startswith('disp_')]
    available_types = [col.replace('disp_', '') for col in disp_cols]

    if element_types is None:
        element_types = available_types
    else:
        element_types = [t for t in element_types if t in available_types]

    # Color schemes
    elem_colors = {
        't3': '#1f77b4', 't6': '#ff7f0e',
        'q4': '#2ca02c', 'q8': '#d62728', 'q9': '#9467bd'
    }
    elem_labels = {
        't3': 'T3 (3-node triangle)',
        't6': 'T6 (6-node triangle)',
        'q4': 'Q4 (4-node quad)',
        'q8': 'Q8 (8-node quad)',
        'q9': 'Q9 (9-node quad)'
    }

    ny_values = np.sort(df['ny'].unique())

    # ========================================================================
    # Plot 1: Displacement vs DOFs (Global comparison)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for elem_key in element_types:
        c = elem_colors.get(elem_key, 'black')

        # Plot all data points for this element
        ax.scatter(df[f'dofs_{elem_key}'], df[f'disp_{elem_key}'],
                   color=c, alpha=0.5, s=20)

        # Add legend entry
        ax.scatter([], [], color=c, label=elem_labels.get(elem_key, elem_key), s=40)

    ax.set_xlabel(r"Number of DOFs", fontsize=12)
    ax.set_ylabel(r"Tip Displacement $u_y$ [m]", fontsize=12)
    ax.set_xscale('log')
    ax.legend(title="Element Type", fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    fig.savefig(f"{output_dir}/convergence_grid/disp_vs_dofs.png", dpi=300)
    plt.close(fig)

    # ========================================================================
    # Plot 2: Displacement vs nx (lines for each ny)
    # ========================================================================
    for elem_key in element_types:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(ny_values)))

        for i, ny in enumerate(ny_values):
            df_ny = df[df['ny'] == ny].sort_values('nx')
            ax.plot(df_ny['nx'], df_ny[f'disp_{elem_key}'],
                    marker='o', markersize=4, linestyle='-', linewidth=1.5,
                    color=colors[i], label=f'$n_y$ = {ny}')

        ax.set_xlabel(r"Horizontal refinement $n_x$", fontsize=12)
        ax.set_ylabel(r"Tip Displacement $u_y$ [m]", fontsize=12)
        ax.set_title(f"Convergence: {elem_labels.get(elem_key, elem_key)}")
        ax.legend(title="Vertical Refinement", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"{output_dir}/convergence_grid/{elem_key}_disp_vs_nx.png", dpi=300)
        plt.close(fig)

    # ========================================================================
    # Plot 3: Global comparison - all elements
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    for elem_key in element_types:
        c = elem_colors.get(elem_key, 'black')

        for ny in ny_values:
            df_ny = df[df['ny'] == ny].sort_values('nx')
            ax.plot(df_ny['nx'], df_ny[f'disp_{elem_key}'],
                    color=c, alpha=0.5, linewidth=1.0, marker='.', markersize=2)

        ax.plot([], [], color=c, label=elem_labels.get(elem_key, elem_key), linewidth=2)

    ax.set_xlabel(r"Horizontal refinement $n_x$", fontsize=12)
    ax.set_ylabel(r"Tip Displacement $u_y$ [m]", fontsize=12)
    ax.legend(title="Element Type", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(f"{output_dir}/convergence_grid/comparison_disp_vs_nx.png", dpi=300)
    plt.close(fig)

    print(f"  -> Plots saved to: {output_dir}/convergence_grid/")


def plot_convergence_grid_from_csv(csv_path=None, output_dir=RESULTS_ROOT, element_types=None):
    """
    Load grid convergence results from CSV and generate plots.

    Args:
        csv_path: Path to CSV file. If None, uses default location.
        output_dir: Directory to save plots.
        element_types: List of element types to plot. If None, plots all.

    Returns:
        pd.DataFrame: The loaded data for further analysis.

    Example:
        >>> df = plot_convergence_grid_from_csv()
        >>> df = plot_convergence_grid_from_csv(element_types=['t6', 'q8'])
    """
    if csv_path is None:
        csv_path = f"{RESULTS_ROOT}/convergence_grid/convergence_grid_results.csv"

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    plot_convergence_grid(df, output_dir, element_types)

    print("\nPlotting complete!")
    return df


def plot_convergence_h_from_csv(csv_path=None, output_dir=RESULTS_ROOT, element_types=None):
    """
    Load convergence results from CSV and generate plots.

    Args:
        csv_path: Path to the CSV file. If None, uses default location.
        output_dir: Directory to save the generated plots.
        element_types: List of element types to plot (e.g., ['t3', 't6', 'q4']).
                      If None, plots all available element types.

    Returns:
        pd.DataFrame: The loaded convergence data.

    Example:
        >>> df = plot_convergence_h_from_csv()
        >>> df = plot_convergence_h_from_csv(element_types=['t3', 'q4'])
    """
    # Default path
    if csv_path is None:
        csv_path = f"{RESULTS_ROOT}/convergence_h/convergence_h_results.csv"

    # Load data
    print(f"Loading convergence data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Extract available element types from column names
    error_cols = [col for col in df.columns if col.startswith('error_')]
    available_types = [col.replace('error_', '') for col in error_cols]

    # Filter element types if specified
    if element_types is None:
        element_types = available_types
    else:
        invalid = set(element_types) - set(available_types)
        if invalid:
            print(f"Warning: Element types {invalid} not found. Available: {available_types}")
        element_types = [t for t in element_types if t in available_types]

    print(f"Plotting element types: {element_types}")

    # Reconstruct data structures for plot_convergence_h
    element_sizes = df['element_size'].values
    results = {key: df[f'error_{key}'].values for key in element_types}
    dof_counts = {key: df[f'dofs_{key}'].values for key in element_types}

    # Generate plots
    plot_convergence_h(element_sizes, results, dof_counts, output_dir)

    print("\nPlotting complete!")
    return df


def convergence_by_element_size(element_sizes, output_dir=RESULTS_ROOT):
    print("=" * 60)
    print("CONVERGENCE STUDY: Element-Size Based")
    print("=" * 60)

    # ========================================================================
    # Phase 2: Generate Configurations
    # ========================================================================
    print("\n--- Phase 2: Generating Configurations ---")

    data = refinement_configs_by_element_size(element_sizes)

    print(f"  Element sizes: {len(element_sizes)} values")
    print(f"    Range: {element_sizes.min():.4f} m → {element_sizes.max():.4f} m")
    print(f"  Mesh refinements:")
    print(f"    nx range: {data['nx'].min()} → {data['nx'].max()}")
    print(f"    ny range: {data['ny'].min()} → {data['ny'].max()}")
    print(f"  Total simulations: {len(element_sizes)} sizes × 5 elements = {len(element_sizes) * 5}")

    # ========================================================================
    # Phase 3: Run Simulations
    # ========================================================================
    print("\n--- Phase 3: Running Simulations ---")

    vectorized_solver = np.vectorize(tip_disps_rel_error, otypes=[float])

    results_data = {}
    dof_counts = {}

    for key, config_array in data['configs'].items():
        print(f"  Processing: {key.upper():3s} ({len(config_array)} meshes) ... ", end='', flush=True)

        # Run all simulations for this element type
        deflections_error = vectorized_solver(config_array)
        results_data[key] = deflections_error

        # Extract DOF counts for each mesh
        dofs = np.array([get_dof_count(cfg) for cfg in config_array])
        dof_counts[key] = dofs

        # Report final mesh stats
        final_nx = data['nx'][-1]
        final_ny = data['ny'][-1]
        final_dofs = dofs[-1]
        print(f"Done! (finest: {final_nx}×{final_ny}, {final_dofs:,} DOFs)")

    # ========================================================================
    # Phase 4: Save Results to CSV
    # ========================================================================
    print("\n--- Phase 4: Saving Results to CSV ---")
    os.makedirs(f"{output_dir}/convergence_h", exist_ok=True)

    # Build DataFrame with element sizes, DOF counts, and errors
    rows = []
    for i, h in enumerate(element_sizes):
        row = {
            'element_size': h,
            'nx': data['nx'][i],
            'ny': data['ny'][i],
        }
        # Add DOF count and error for each element type
        for elem_key in results_data.keys():
            row[f'dofs_{elem_key}'] = dof_counts[elem_key][i]
            row[f'error_{elem_key}'] = results_data[elem_key][i]
        rows.append(row)

    df_results = pd.DataFrame(rows)
    csv_path = f"{output_dir}/convergence_h/convergence_h_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  -> Results saved to: {csv_path}")

    # ========================================================================
    # Phase 5: Generate Plots
    # ========================================================================
    print("\n--- Phase 5: Generating Plots ---")

    plot_convergence_h(
        element_sizes,
        results_data,
        dof_counts,
        output_dir
    )

    print("\n" + "=" * 60)
    print("[OK] Convergence study complete!")
    print("=" * 60)

    return {
        'element_sizes': element_sizes,
        'results': results_data,
        'mesh_info': {
            'nx': data['nx'],
            'ny': data['ny'],
            'dofs': dof_counts
        }
    }


# Standard element configurations
T3 = create_config(BASE_CONFIG, 'Fem_Beam_T3', io={'dir': RESULTS_ROOT}, elements={'type': 'triangle', 'order': 1})
T6 = create_config(BASE_CONFIG, 'Fem_Beam_T6', io={'dir': RESULTS_ROOT}, elements={'type': 'triangle', 'order': 2})
Q4 = create_config(BASE_CONFIG, 'Fem_Beam_Q4', io={'dir': RESULTS_ROOT}, elements={'type': 'quad', 'order': 1})
Q8 = create_config(BASE_CONFIG, 'Fem_Beam_Q8', io={'dir': RESULTS_ROOT}, elements={'type': 'quad', 'order': 2})
Q9 = create_config(BASE_CONFIG, 'Fem_Beam_Q9', io={'dir': RESULTS_ROOT}, elements={'type': 'quad', 'order': 3})

ALL_CONFIGS = [T3, T6, Q4, Q8, Q9]


def compute_nx(L, H, NY):
    size = H / NY
    return int(L // size)


NX_RANGE = np.unique(np.geomspace(1, 200, num=30, dtype=int))
NY_RANGE = np.unique(np.geomspace(1, 25, num=15, dtype=int))
ELEMENT_SIZES = np.unique(np.geomspace(1, 16, num=15, dtype=float))

if __name__ == "__main__":
    # Grid-based convergence study (recommended)
    df = convergence_grid(np.flip(NX_RANGE), np.flip(NY_RANGE))

    # Or use element-size-based study:
    # convergence_by_element_size(ELEMENT_SIZES)

    # Or reload and replot from saved CSV:
    # df = plot_convergence_grid_from_csv()
