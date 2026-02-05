import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

markers = Line2D.filled_markers[1:]


def plot_displacement_nx(filename='convergence_results.csv', element='t3'):
    df = pd.read_csv(filename)
    nx_unique = sorted(df['nx'].unique())
    n = len(nx_unique)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n)) # Niveaux de gris

    fig, ax = plt.subplots(figsize=(8, 5))

    for j, nx_val in enumerate(nx_unique):
        subset = df[df['nx'] == nx_val].sort_values(f'dofs_{element}')
        ax.plot(
            subset[f'dofs_{element}'],
            subset[f'disp_{element}'],
            color=colors[j],
            linewidth=1,
            label=f'$n_x$ = {nx_val}',
            marker = markers[j % len(markers)],
            markersize=4
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'DOFs')
    ax.set_ylabel(r'Tip displacement $u_y$')
    ax.legend(title=r'Vertical discretization $n_y$', bbox_to_anchor=(1, 1), loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'dispacement_convergence_block_nx.png', dpi=300, bbox_inches='tight')


def plot_error_nx(target, filename='convergence_results.csv', element='t3'):
    df = pd.read_csv(filename)

    nx_unique = sorted(df['nx'].unique())
    n = len(nx_unique)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n)) # Niveaux de gris

    fig, ax = plt.subplots(figsize=(8, 5))

    for j, nx_val in enumerate(nx_unique):
        subset = df[df['nx'] == nx_val].sort_values(f'dofs_{element}')
        error = relative_error(subset[f'disp_{element}'], target)
        ax.plot(
            subset[f'dofs_{element}'],
            error,
            color=colors[j],
            linewidth=1,
            label=f'$n_y$ = {nx_val}',
            marker = markers[j % len(markers)],
            markersize=4
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'DOFs')
    ax.set_ylabel(r'Relative error [\%]')
    ax.legend(title=r'Vertical discretization $n_y$', bbox_to_anchor=(1, 0), loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'error_convergence_block_nx.png', dpi=300, bbox_inches='tight')


def plot_displacement_ny(filename='convergence_results.csv', element='t3'):
    df = pd.read_csv(filename)

    ny_unique = sorted(df['ny'].unique())
    n = len(ny_unique)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n)) # Niveaux de gris

    fig, ax = plt.subplots(figsize=(8, 5))

    for j, ny_val in enumerate(ny_unique):
        subset = df[df['ny'] == ny_val].sort_values(f'dofs_{element}')
        ax.plot(
            subset[f'dofs_{element}'],
            subset[f'disp_{element}'],
            color=colors[j],
            linewidth=1,
            label=f'$n_y$ = {ny_val}',
            marker = markers[j % len(markers)],
            markersize=4
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'DOFs')
    ax.set_ylabel(r'Tip displacement $u_y$')
    ax.legend(title=r'Vertical discretization $n_y$', bbox_to_anchor=(1, 1), loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'dispacement_convergence_block_ny.png', dpi=300, bbox_inches='tight')


def plot_error_ny(target, filename='convergence_results.csv', element='t3'):
    df = pd.read_csv(filename)

    ny_unique = sorted(df['ny'].unique())
    n = len(ny_unique)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n)) # Niveaux de gris

    fig, ax = plt.subplots(figsize=(8, 5))

    for j, ny_val in enumerate(ny_unique):
        subset = df[df['ny'] == ny_val].sort_values(f'dofs_{element}')
        error = relative_error(subset[f'disp_{element}'], target)
        ax.plot(
            subset[f'dofs_{element}'],
            error,
            color=colors[j],
            linewidth=1,
            label=f'$n_y$ = {ny_val}',
            marker = markers[j % len(markers)],
            markersize=4
        )

    ax.set_xscale('log')
    ax.set_xlabel(r'DOFs')
    ax.set_ylabel(r'Relative error [\%]')
    ax.legend(title=r'Vertical discretization $n_y$', bbox_to_anchor=(1, 0), loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'error_convergence_block_ny.png', dpi=300, bbox_inches='tight')

def relative_error(subset,target_value:float):
    error = (subset-target_value)/target_value
    return 100*error

def timo(E,P,nu,L,h,b):
    I = (b * h ** 3) / 12
    A = b * h
    G = E / (2 * (1 + nu))
    return (P * L ** 3) / (3 * E * I) + (5 * P * L) / (6 * G * A)


if __name__ == "__main__":
    timoshenko = timo(E=30e9, P=-100e3, nu=0.0, L=3, h=0.5, b=0.2)
    plot_displacement_nx('convergence_results.csv')
    plot_displacement_ny('convergence_results.csv')
    plot_error_nx(timoshenko, 'convergence_results.csv')
    plot_error_ny(timoshenko, 'convergence_results.csv')
