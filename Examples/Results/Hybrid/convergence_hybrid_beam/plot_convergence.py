import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

MARKERS = Line2D.filled_markers[1:]


def relative_error(subset, target_value: float):
    error = (subset - target_value) / target_value
    return 100 * error


def timo(E, P, nu, L, h, b):
    I = (b * h ** 3) / 12
    A = b * h
    G = E / (2 * (1 + nu))
    return (P * L ** 3) / (3 * E * I) + (5 * P * L) / (6 * G * A)


def plot_sensitivity(folder, filename, element, vary_param, metric='displacement', target=None):
    """
    Plots sensitivity of results (displacement or error) with respect to a varying parameter (nx or ny).

    Args:
        folder (str): Path to the output folder (also used for reading CSV if filename is relative).
        filename (str): Path to the CSV results file (relative to folder or absolute).
        element (str): Element type (e.g., 't3', 'q4').
        vary_param (str): The parameter to vary ('nx' or 'ny').
        metric (str): 'displacement' or 'error'.
        target (float, optional): Target value for error calculation. Required if metric is 'error'.
    """
    filepath = f'{folder}/{filename}' if not os.path.isabs(filename) else filename
    df = pd.read_csv(filepath)
    df['nx'] *= 3

    unique_vals = sorted(df[vary_param].unique())

    # Special sampling for 'nx' to match original logic (limit number of lines)
    if vary_param == 'nx':
        n_select = len(df['ny'].unique())
        if len(unique_vals) > n_select:
            indices = np.linspace(0, len(unique_vals) - 1, n_select, dtype=int)
            unique_vals = [unique_vals[i] for i in indices]

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(unique_vals)))
    fig, ax = plt.subplots(figsize=(6, 4))

    for j, val in enumerate(unique_vals):
        subset = df[df[vary_param] == val].sort_values(f'dofs_{element}')

        x_data = subset[f'dofs_{element}']
        y_data = subset[f'disp_{element}']

        if metric == 'error':
            if target is None:
                raise ValueError("Target value is required for error plots.")
            y_data = relative_error(y_data, target)

        label_param = f"{vary_param[0]}_{vary_param[1]}"  # e.g., n_x or n_y
        ax.plot(
            x_data,
            y_data,
            color=colors[j],
            linewidth=1,
            label=f'${label_param}$ = {val}',
            marker=MARKERS[j % len(MARKERS)],
            markersize=4
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'DOFs')

    if metric == 'displacement':
        ax.set_ylabel(r'Tip displacement $u_y$')
        legend_loc = 'best'
        bbox = (1, 1)
    else:
        ax.set_ylabel(r'Relative error [\%]')
        legend_loc = 'best'
        bbox = (1, 0)

    ax.legend()#bbox_to_anchor=bbox, loc=legend_loc)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name = f'{folder}/{metric}_convergence_{vary_param}_{element}.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_element_comparison(folder, filename, fixed_param, metric='displacement', target=None):
    """
    Plots comparison between elements for a fixed grid parameter.

    Args:
        folder (str): Path to the output folder.
        filename (str): Path to the CSV results file (relative to folder or absolute).
        fixed_param (str): The parameter to hold fixed ('nx' or 'ny').
                           The maximum value of this parameter is used.
        metric (str): 'displacement' or 'error'.
        target (float, optional): Target value for error calculation.
    """
    filepath = f'{folder}/{filename}' if not os.path.isabs(filename) else filename
    df = pd.read_csv(filepath)
    df['nx'] *= 3
    elements = ['t3', 't6', 'q4', 'q8']

    # Select the last (max) unique value for the fixed parameter
    fixed_val = [sorted(df[fixed_param].unique())[0], sorted(df[fixed_param].unique())[-1]]

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, element in enumerate(elements):
        for j, fixed in enumerate(fixed_val):
            subset = df[df[fixed_param] == fixed].sort_values(f'dofs_{element}')

            x_data = subset[f'dofs_{element}']
            y_data = subset[f'disp_{element}']

            if metric == 'error':
                if target is None:
                    raise ValueError("Target value is required for error plots.")
                y_data = relative_error(y_data, target)

            label_param = f"{fixed_param[0]}_{fixed_param[1]}"
            ax.plot(
                x_data,
                y_data,
                color=colors[i],
                linewidth=1,
                label=f"{element.capitalize()}: ${label_param}$ = {fixed}",
                marker=MARKERS[j],
                markersize=4
            )

    ax.set_xscale('log')
    ax.set_xlabel(r'DOFs')

    if metric == 'displacement':
        ax.set_ylabel(r'Tip displacement $u_y$')
        legend_loc = 'upper right'
        bbox = (1, 1)
    else:
        ax.set_ylabel(r'Relative error [\%]')
        legend_loc = 'lower right'
        bbox = (1, 0)

    ax.legend(title=r'Elements')#, bbox_to_anchor=bbox, loc=legend_loc)
    ax.grid(True, alpha=0.3)

    out_name = f'{folder}/{metric}_convergence_{fixed_param}_global.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_displacement_comparison(folder, filename, reference_file, element, vary_param='ny', figsize=(8, 5)):
    """
    Plots displacement comparison between hybrid and reference (FEM) results.

    Args:
        folder (str): Path to the output folder (and location of hybrid CSV).
        filename (str): Path to the hybrid CSV results file (relative to folder).
        reference_file (str): Path to the reference CSV file (relative to script dir or absolute).
        element (str): Element type (e.g., 't3', 'q4').
        vary_param (str): The parameter to vary for grouping ('nx' or 'ny'). Default 'ny'.
        figsize (tuple): Figure size (width, height).
    """
    # Build paths
    filepath = f'{folder}/{filename}' if not os.path.isabs(filename) else filename
    ref_filepath = reference_file if os.path.isabs(reference_file) else reference_file

    df = pd.read_csv(filepath)
    df['nx'] *= 3
    df_ref = pd.read_csv(ref_filepath)

    # Find common values of the varying parameter
    unique_vals = sorted(df[vary_param].unique())
    unique_vals_ref = sorted(df_ref[vary_param].unique())
    common_vals = sorted(set(unique_vals).intersection(set(unique_vals_ref)))

    if not common_vals:
        print(f"Warning: No common {vary_param} values found between hybrid and reference data.")
        return

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(common_vals)))
    fig, ax = plt.subplots(figsize=figsize)

    for i, val in enumerate(common_vals):
        # Hybrid data (solid line)
        subset = df[df[vary_param] == val].sort_values(f'dofs_{element}')
        label_param = f"{vary_param[0]}_{vary_param[1]}"
        ax.plot(
            subset[f'dofs_{element}'],
            subset[f'disp_{element}'],
            color=colors[i],
            linewidth=1,
            label=f'${label_param}$ = {val} (hybrid)',
            marker=MARKERS[i % len(MARKERS)],
            markersize=3
        )

        # Reference data (dashed line)
        subset_ref = df_ref[df_ref[vary_param] == val].sort_values(f'dofs_{element}')
        ax.plot(
            subset_ref[f'dofs_{element}'],
            subset_ref[f'disp_{element}'],
            color=colors[i],
            linewidth=1,
            linestyle='--',
            label=f'${label_param}$ = {val} (fem)',
            marker=MARKERS[i % len(MARKERS)],
            markersize=3
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'DOFs')
    ax.set_ylabel(r'Tip displacement $u_y$')
    ax.legend(title=f'Discretization ${vary_param[0]}_{vary_param[1]}$', loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name = f'{folder}/displacement_comparison_{vary_param}_{element}.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_comparison(folder, filename, reference_file, element, target, vary_param='ny', figsize=(8, 5)):
    """
    Plots error comparison between hybrid and reference (FEM) results.

    Args:
        folder (str): Path to the output folder (and location of hybrid CSV).
        filename (str): Path to the hybrid CSV results file (relative to folder).
        reference_file (str): Path to the reference CSV file (relative to script dir or absolute).
        element (str): Element type (e.g., 't3', 'q4').
        target (float): Target (analytical) value for error calculation.
        vary_param (str): The parameter to vary for grouping ('nx' or 'ny'). Default 'ny'.
        figsize (tuple): Figure size (width, height).
    """
    # Build paths
    filepath = f'{folder}/{filename}' if not os.path.isabs(filename) else filename
    ref_filepath = reference_file if os.path.isabs(reference_file) else reference_file

    df = pd.read_csv(filepath)
    df['nx'] *= 3
    df_ref = pd.read_csv(ref_filepath)

    # Find common values of the varying parameter
    unique_vals = sorted(df[vary_param].unique())
    unique_vals_ref = sorted(df_ref[vary_param].unique())
    common_vals = sorted(set(unique_vals).intersection(set(unique_vals_ref)))

    if not common_vals:
        print(f"Warning: No common {vary_param} values found between hybrid and reference data.")
        return

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(common_vals)))
    fig, ax = plt.subplots(figsize=figsize)

    for i, val in enumerate(common_vals):
        # Hybrid data (solid line)
        subset = df[df[vary_param] == val].sort_values(f'dofs_{element}')
        error = relative_error(subset[f'disp_{element}'], target)
        label_param = f"{vary_param[0]}_{vary_param[1]}"
        ax.plot(
            subset[f'dofs_{element}'],
            error,
            color=colors[i],
            linewidth=1,
            label=f'${label_param}$ = {val} (hybrid)',
            marker=MARKERS[i % len(MARKERS)],
            markersize=3
        )

        # Reference data (dashed line)
        subset_ref = df_ref[df_ref[vary_param] == val].sort_values(f'dofs_{element}')
        error_ref = relative_error(subset_ref[f'disp_{element}'], target)
        ax.plot(
            subset_ref[f'dofs_{element}'],
            error_ref,
            color=colors[i],
            linewidth=1,
            linestyle='--',
            label=f'${label_param}$ = {val} (fem)',
            marker=MARKERS[i % len(MARKERS)],
            markersize=3
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'DOFs')
    ax.set_ylabel(r'Relative error [\%]')
    ax.legend(title=f'Discretization ${vary_param[0]}_{vary_param[1]}$', loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name = f'{folder}/error_comparison_{vary_param}_{element}.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    #New version (now working)
    FOLDERS = ['constraint', 'penalty', 'lagrange', 'mortar']
    FILENAME = 'convergence_results.csv'
    REFERENCE_FILE = 'convergence_grid_results_reference.csv'
    ELEMENTS = ['t3', 't6', 'q4', 'q8']

    timoshenko = timo(E=30e9, P=-100e3, nu=0.0, L=3, h=0.5, b=0.2)
    print(f"Target Timoshenko value: {timoshenko}")

    for folder in FOLDERS:
        for element in ELEMENTS:
            # Plot Displacement Sensitivity
            plot_sensitivity(folder, FILENAME, element, 'nx', metric='displacement')
            plot_sensitivity(folder, FILENAME, element, 'ny', metric='displacement')

            # Plot Error Sensitivity
            plot_sensitivity(folder, FILENAME, element, 'nx', metric='error', target=timoshenko)
            plot_sensitivity(folder, FILENAME, element, 'ny', metric='error', target=timoshenko)

            # Plot Comparison with Reference (FEM)
            plot_displacement_comparison(folder, FILENAME, REFERENCE_FILE, element, vary_param='nx')
            plot_displacement_comparison(folder, FILENAME, REFERENCE_FILE, element, vary_param='ny')
            plot_error_comparison(folder, FILENAME, REFERENCE_FILE, element, timoshenko, vary_param='nx')
            plot_error_comparison(folder, FILENAME, REFERENCE_FILE, element, timoshenko, vary_param='ny')

        # Plot Element Comparisons
        plot_element_comparison(folder, FILENAME, 'nx', metric='displacement')
        plot_element_comparison(folder, FILENAME, 'ny', metric='displacement')

        plot_element_comparison(folder, FILENAME, 'nx', metric='error', target=timoshenko)
        plot_element_comparison(folder, FILENAME, 'ny', metric='error', target=timoshenko)