import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


def plot_convergence_heatmap(sheet_name, value_col, vmin=-1e-6, vmax=1e-6, label='Error Intensity', xlabel=None,
                             ylabel=None, save_path=None, multiplier=1.0, max_ny=None, precision=2):
    """
    Generates a heatmap from the Convergence.xlsx data.
    """
    # Use path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Convergence.xlsx')

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    if max_ny is not None:
        df = df[df['ny'] <= max_ny]

    grid_data = df.pivot_table(index='ny', columns='nx', values=value_col) * multiplier
    grid_data.sort_index(ascending=False, inplace=True)

    plt.figure(figsize=(8, 5))
    sns.heatmap(grid_data, cmap='coolwarm', vmin=vmin, vmax=vmax,
                linewidths=0.2, linecolor='black',
                cbar_kws={'label': label, 'format': f'%.{precision}f', 'orientation': 'horizontal', 'pad': 0.2})

    plt.xlabel(xlabel if xlabel else r'Discretization $n_x$')
    plt.ylabel(ylabel if ylabel else r'Discretization $n_y$')

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    if save_path:
        output_dir = os.path.join(script_dir, 'heatmaps')
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, save_path)
        plt.savefig(full_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {full_path}")
        plt.close()
    else:
        plt.show()


def compare_hybrid():
    columns = ['disp_t3', 'disp_t6', 'disp_q4', 'disp_q8', ]
    names = ['mortar_diff_', 'lagrange_diff_', 'penalty_diff_']
    sheets = ['Constraint-Mortar', 'Constraint-Lagrange', 'Constraint-Penalty']

    for sheet, name in zip(sheets, names):
        for col in columns:
            plot_convergence_heatmap(
                save_path=name + col,
                sheet_name=sheet,
                value_col=col,
                label='Displacement difference [mm]',
                multiplier=1e3,
                vmax=1e-3,
                vmin=0
            )


def displacement():
    columns = ['disp_t3',
               'disp_t6',
               'disp_q4',
               'disp_q8', ]
    sheets = ['Constraint', 'Penalty', 'Lagrange', 'Mortar', 'FEM']
    names = ['Constraint', 'Penalty', 'Lagrange', 'Mortar', 'FEM']
    for sheet, name in zip(sheets, names):
        for col in columns:
            plot_convergence_heatmap(
                save_path=col + '_' + name,
                sheet_name=sheet,
                value_col=col,
                label='Tip displacement [mm]',
                multiplier=1e3,
                vmin=-16,
                vmax=-12
            )

    plot_convergence_heatmap(
        save_path='timo_err_' + 'Block',
        sheet_name='Block',
        value_col='timo_err_t3',
        label='Tip displacement [mm]',
        multiplier=1e3,
        vmin=-16,
        vmax=-12
    )


def timoshenko():
    columns = ['timo_err_t3',
               'timo_err_t6',
               'timo_err_q4',
               'timo_err_q8', ]
    sheets = ['Constraint', 'Penalty', 'Lagrange', 'Mortar']
    names = ['Constraint', 'Penalty', 'Lagrange', 'Mortar']
    for sheet, name in zip(sheets, names):
        for col in columns:
            plot_convergence_heatmap(
                save_path=col + '_' + name,
                sheet_name=sheet,
                value_col=col,
                label='Relative error to Timoshenko [\%]',
                multiplier=1e2,
                vmin=-5,
                vmax=5,
            )


def timo_block():
    plot_convergence_heatmap(
        save_path='timo_err_' + 'Block',
        sheet_name='Block',
        value_col='timo_err_t3',
        label='Relative error to Timoshenko [\%]',
        multiplier=1e2,
        vmin=-5,
        vmax=5,
    )


def timo_hybrid():
    columns = ['timo_err_t3',
               'timo_err_t6',
               'timo_err_q4',
               'timo_err_q8', ]
    sheets = ['Constraint', 'Penalty', 'Lagrange', 'Mortar']
    names = ['Constraint', 'Penalty', 'Lagrange', 'Mortar']
    for sheet, name in zip(sheets, names):
        for col in columns:
            plot_convergence_heatmap(
                save_path=col + '_' + name,
                sheet_name=sheet,
                value_col=col,
                label='Relative error to Timoshenko [\%]',
                multiplier=1e2,
                vmin=-5,
                vmax=5
            )


def timo_fem():
    columns = ['timo_err_t3',
               'timo_err_t6',
               'timo_err_q4',
               'timo_err_q8',
               'timo_err_q9']
    sheets = ['Midspan', 'FEM']
    names = ['Midspan', 'FEM']
    for sheet, name in zip(sheets, names):
        for col in columns:
            plot_convergence_heatmap(
                save_path=col + '_' + name,
                sheet_name=sheet,
                value_col=col,
                label='Relative error to Timoshenko [\%]',
                multiplier=1e2,
                vmin=-10,
                vmax=5,
            )


if __name__ == "__main__":
    timo_hybrid()
