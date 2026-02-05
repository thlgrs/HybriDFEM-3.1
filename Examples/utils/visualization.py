"""
Shared Visualization Utilities
===============================

Centralized visualization logic for example scripts.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from Core.Solvers.Visualizer import Visualizer


def visualize(St, config):
    figsize = config['io'].get('figsize', None)
    try:
        plot_initial(St, config)
    except (ValueError, RuntimeError):
        pass
    try:
        plot_deformed(St, config)
    except (ValueError, RuntimeError):
        pass
    try:
        plot_stress(St, config)
    except (ValueError, RuntimeError):
        pass
    try:
        plot_displacement(St, config)
    except (ValueError, RuntimeError):
        pass


def plot_initial(St, config):
    io_conf = config['io']
    out_path = os.path.join(io_conf['dir'], io_conf['filename'])
    os.makedirs(io_conf['dir'], exist_ok=True)
    show_nodes = io_conf.get('show_nodes', False)
    figsize = io_conf.get('figsize', None)
    fig = Visualizer.plot_initial(
        St,
        figsize=figsize,
        save_path=out_path + '_initial.png',
        show_nodes=show_nodes
    )
    plt.close(fig)


def plot_deformed(St, config):
    io_conf = config['io']
    out_path = os.path.join(io_conf['dir'], io_conf['filename'])
    os.makedirs(io_conf['dir'], exist_ok=True)
    figsize = io_conf.get('figsize', None)
    scale = io_conf.get('scale', 1.0)
    show_nodes = io_conf.get('show_nodes', False)
    fig = Visualizer.plot_deformed(
        St,
        scale=scale,
        figsize=figsize,
        save_path=out_path + '_deformed.png',
        show_nodes=show_nodes
    )
    plt.close(fig)


def plot_stress(St, config):
    """
    Generate standard plots for structure.

    Args:
        St: Solved structure object
        config: Configuration dictionary (must contain 'io' section)
    """
    io_conf = config['io']
    out_path = os.path.join(io_conf['dir'], io_conf['filename'])
    os.makedirs(io_conf['dir'], exist_ok=True)

    show_nodes = io_conf.get('show_nodes', False)
    sigma_x_range = io_conf.get('sigma_x_range', [None, None])
    sigma_y_range = io_conf.get('sigma_y_range', [None, None])
    tau_yx_range = io_conf.get('tau_yx_range', [None, None])
    tau_xy_range = io_conf.get('tau_xy_range', [None, None])

    # Create 2x2 subplot for comprehensive stress analysis
    figsize = io_conf.get('figsize', None)

    # Horizontal stress
    s1 = Visualizer.plot_stress(
        St, scale=0,
        component='normal', angle=np.pi / 2,
        show_nodes=show_nodes,
        vmin=sigma_x_range[0],
        vmax=sigma_x_range[1],
        figsize=figsize,
        save_path=out_path + '_stress_xx.png',
    )
    # Vertical stress
    s2 = Visualizer.plot_stress(
        St, scale=0,
        component='normal', angle=0,
        show_nodes=show_nodes,
        vmin=sigma_y_range[0],
        vmax=sigma_y_range[1],
        figsize=figsize,
        save_path=out_path + '_stress_yy.png',
    )
    # Shear
    s3 = Visualizer.plot_stress(
        St, scale=0,
        component='shear', angle=0,
        show_nodes=show_nodes,
        vmin=tau_yx_range[0],
        vmax=tau_yx_range[1],
        figsize=figsize,
        save_path=out_path + '_stress_xy.png',
    )

    plt.close(s1)
    plt.close(s2)
    plt.close(s3)


def plot_displacement(St, config):
    io_conf = config['io']
    out_path = os.path.join(io_conf['dir'], io_conf['filename'])
    os.makedirs(io_conf['dir'], exist_ok=True)

    show_nodes = io_conf.get('show_nodes', False)
    figsize = io_conf.get('figsize', None)
    vmax_disp = io_conf.get('vmax_disp', None)

    fig = Visualizer.plot_displacement(
        St,
        scale=0,
        figsize=figsize,
        save_path=out_path + '_disp.png',
        show_nodes=show_nodes,
        vmax=vmax_disp,
    )
    plt.close(fig)
