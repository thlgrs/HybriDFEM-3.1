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


def plot_sensitivity(filename, element, vary_param, metric='displacement', target=None):
    """
    Plots sensitivity of results (displacement or error) with respect to a varying parameter (nx or ny).

    Args:
        filename (str): Path to the CSV results file.
        element (str): Element type (e.g., 't3', 'q4').
        vary_param (str): The parameter to vary ('nx' or 'ny').
        metric (str): 'displacement' or 'error'.
        target (float, optional): Target value for error calculation. Required if metric is 'error'.
    """
    df = pd.read_csv(filename)

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
        legend_loc = 'upper right'
        bbox = (1, 1)
    else:
        ax.set_ylabel(r'Relative error [\%]')
        legend_loc = 'lower right'
        bbox = (1, 0)

    ax.legend(bbox_to_anchor=bbox, loc=legend_loc)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    # Corrected typo: 'dispacement' -> 'displacement'
    out_name = f'{metric}_convergence_{vary_param}_{element}.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_element_comparison(filename, fixed_param, metric='displacement', target=None):
    """
    Plots comparison between elements for a fixed grid parameter.

    Args:
        filename (str): Path to the CSV results file.
        fixed_param (str): The parameter to hold fixed ('nx' or 'ny').
                           The maximum value of this parameter is used.
        metric (str): 'displacement' or 'error'.
        target (float, optional): Target value for error calculation.
    """
    df = pd.read_csv(filename)
    elements = ['t3', 't6', 'q4', 'q8', 'q9']

    # Select the last (max) unique value for the fixed parameter
    fixed_val = sorted(df[fixed_param].unique())[-1]

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, element in enumerate(elements):
        subset = df[df[fixed_param] == fixed_val].sort_values(f'dofs_{element}')

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
            label=f"{element.capitalize()}: ${label_param}$ = {fixed_val}",
            marker='^',
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

    ax.legend(title=r'Elements', bbox_to_anchor=bbox, loc=legend_loc)
    ax.grid(True, alpha=0.3)

    out_name = f'{metric}_convergence_{fixed_param}_global.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    FILENAME = 'convergence_grid_results.csv'
    ELEMENTS = ['t3', 't6', 'q4', 'q8', 'q9']


    for element in ELEMENTS:
        # Plot Displacement Sensitivity
        plot_sensitivity(FILENAME, element, 'nx', metric='displacement')
        plot_sensitivity(FILENAME, element, 'ny', metric='displacement')

        # Plot Error Sensitivity
        plot_sensitivity(FILENAME, element, 'nx', metric='error', target=-3.54e-3)
        plot_sensitivity(FILENAME, element, 'ny', metric='error', target=-3.54e-3)

    # Plot Element Comparisons
    plot_element_comparison(FILENAME, 'nx', metric='displacement')
    plot_element_comparison(FILENAME, 'ny', metric='displacement')

    plot_element_comparison(FILENAME, 'nx', metric='error', target=-3.54e-3)
    plot_element_comparison(FILENAME, 'ny', metric='error', target=-3.54e-3)