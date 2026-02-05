"""
Analysis Comparison Utilities
=============================

Utilities for comparing results from different analysis runs (HDF5 files).
"""

import os
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
from Core.Solvers.Visualizer import PlotStyle


def compare_analyses(file_pattern, results_dir='Results', output_dir='Results/Comparisons',
                     title=None, show_force=True, style=None):
    """
    Compare multiple analysis results by plotting Load-Displacement curves.
    
    Automatically aligns control node displacements even if DOF indices differ 
    (e.g. Constraint vs Lagrange), by reading the 'control_dof' metadata 
    saved by the solver.

    Args:
        file_pattern: Glob pattern for HDF5 files (e.g., 'hybrid_dispctrl_*.h5')
        results_dir: Directory containing HDF5 files
        output_dir: Directory to save comparison plots
        title: Plot title
        show_force: If True, plot Force (N). If False, plot Load Factor (lambda).
        style: PlotStyle object (optional)
    """
    # Setup paths
    search_path = os.path.join(results_dir, file_pattern)
    files = glob.glob(search_path)

    if not files:
        print(f"[WARN] No files found matching: {search_path}")
        return

    files.sort()
    os.makedirs(output_dir, exist_ok=True)

    # Setup plot
    if style is None: style = PlotStyle()
    style.apply_global_config()
    fig, ax = plt.subplots(figsize=style.figsize)
    style.apply_to_figure(fig)
    style.apply_to_axes(ax)

    print(f"\nComparing {len(files)} analyses:")

    # Color cycle
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for c in prop_cycle]

    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        label = filename.replace('.h5', '').replace('hybrid_dispctrl_', '').replace('hybrid_linear_', '')

        try:
            with h5py.File(filepath, 'r') as f:
                # 1. Check for necessary data
                if 'U_conv' not in f or 'LoadFactor_conv' not in f:
                    print(f"  [SKIP] {filename}: Missing convergence data")
                    continue

                # 2. Get Control Info
                # Try to get control_dof from metadata (saved by solver)
                if 'control_dof' in f.attrs:
                    ctrl_dof = int(f.attrs['control_dof'])
                else:
                    # Fallback: Try to detect from max displacement
                    # (Heuristic: Control node usually moves the most in disp control)
                    U_final = f['U_conv'][:, -1]
                    ctrl_dof = np.argmax(np.abs(U_final))
                    print(f"  [INFO] {filename}: Auto-detected control DOF {ctrl_dof}")

                # 3. Extract Data
                lambdas = f['LoadFactor_conv'][:]
                U_history = f['U_conv'][:]

                if ctrl_dof >= U_history.shape[0]:
                    print(f"  [ERR]  {filename}: Control DOF {ctrl_dof} out of bounds")
                    continue

                disp = U_history[ctrl_dof, :]

                # 4. Compute Force
                y_values = lambdas
                y_label_str = "Load Factor"

                if show_force:
                    if 'P_ref' in f.attrs:
                        p_ref = float(f.attrs['P_ref'])
                    elif 'P_ref' in f:  # Sometimes stored as dataset
                        p_ref = np.linalg.norm(f['P_ref'][:])
                    else:
                        p_ref = 1.0  # Default

                    y_values = lambdas * p_ref
                    y_label_str = "Force [N]"

                # 5. Plot
                ax.plot(disp, y_values,
                        label=label,
                        color=colors[i % len(colors)],
                        linewidth=1.5,
                        marker='o' if len(disp) < 20 else None,
                        markevery=max(1, len(disp) // 10))

                print(f"  [PLOT] {filename} (Steps: {len(disp)})")

        except Exception as e:
            print(f"  [ERR]  {filename}: {e}")

    # Formatting
    ax.set_xlabel("Displacement [m]", fontsize=style.label_fontsize)
    ax.set_ylabel(y_label_str, fontsize=style.label_fontsize)
    ax.set_title(title if title else "Displacement Control Comparison", fontsize=style.title_fontsize)
    ax.legend()

    # Save
    out_name = file_pattern.replace('*', 'ALL').replace('.h5', '') + "_compare.png"
    save_path = os.path.join(output_dir, out_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nComparison saved to: {save_path}")
