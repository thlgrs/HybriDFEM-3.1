# -*- coding: utf-8 -*-
"""
Modal Analysis - Plastic Frame with Modal Degradation
======================================================

Refactored from: Legacy/Examples/Modal Analysis/Plastic_Frame/Plastic_Frame.py

This example demonstrates modal analysis of a portal frame with plastic
hinges, tracking the change in natural frequencies as the structure
undergoes plastic deformation (modal degradation).

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Modal + Nonlinear static (displacement control)

Configuration:
- Portal frame: 3m columns, 3m beam
- Material: E = 30 GPa, fy = 20 MPa (elasto-plastic)
- Incremental pushover with modal analysis at each step
"""

import os
import sys
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt
import h5py

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Structure_Block, Modal, StaticNonLinear
from Core.Structures import BeamBlock
from Core.Objects.ConstitutiveLaw.Material import Plastic_Mat


def main():
    # ==========================================================================
    # Output Directory Setup
    # ==========================================================================
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ==========================================================================
    # Parameters
    # ==========================================================================
    # Frame geometry
    N1 = np.array([0, 0], dtype=float)    # Left column base
    N2 = np.array([0, 3], dtype=float)    # Left corner
    N3 = np.array([3, 3], dtype=float)    # Right corner
    N4 = np.array([3, 0], dtype=float)    # Right column base

    # Cross-section dimensions
    B_b = 0.2     # Beam width [m]
    H_b = 0.2     # Beam height [m]
    H_c = 0.2 * 2 ** (1/3)  # Column height (scaled for stiffness)

    CPS = 20      # Contact points per interface
    BLOCKS = 30   # Blocks per member

    # Material properties
    E = 30e9      # Young's modulus [Pa]
    NU = 0.0      # Poisson's ratio
    FY = 20e6     # Yield stress [Pa]
    RHO = 2000.0  # Density [kg/m³]

    # Analysis parameters
    nb_modes = 3
    n_incr = 50     # Number of load increments
    INCR = 2e-3     # Displacement increment [m]
    STEPS = 3       # Steps per increment

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    # Create plastic material
    MAT = Plastic_Mat(E, NU, FY)

    St = Structure_Block()

    # Left column (N1 to N2)
    Col1_temp = BeamBlock(N1, N2, BLOCKS, H_c, RHO, b=B_b, material=MAT)
    for block in Col1_temp.list_blocks:
        St.list_blocks.append(block)

    # Beam (N2 to N3)
    Beam_temp = BeamBlock(N2, N3, BLOCKS, H_b, RHO, b=B_b, material=MAT)
    for block in Beam_temp.list_blocks:
        St.list_blocks.append(block)

    # Right column (N3 to N4)
    Col2_temp = BeamBlock(N3, N4, BLOCKS, H_c, RHO, b=B_b, material=MAT)
    for block in Col2_temp.list_blocks:
        St.list_blocks.append(block)

    St.make_nodes()
    St.make_cfs(lin_geom=True, nb_cps=CPS)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Find nodes at supports
    N1_idx = St.get_node_at(N1)
    N4_idx = St.get_node_at(N4)

    # Pin supports (fix translations, free rotation)
    St.fix_node(N1_idx, [0, 1])
    St.fix_node(N4_idx, [0, 1])

    # Apply horizontal load at top of left column
    control_node = BLOCKS - 1  # Top of first column
    St.load_node(control_node, [0], 1e4)

    # ==========================================================================
    # Initial Modal Analysis
    # ==========================================================================
    print("Running initial modal analysis...")

    # Save structure for reloading
    struct_file = os.path.join(save_path, 'Plastic_Frame.pkl')
    St.save_structure(struct_file)

    St = Modal.solve(St, nb_modes, filename=os.path.join(save_path, 'Step_0_Modal'))

    # Store frequencies at each step
    w = np.zeros((nb_modes, n_incr + 1))
    w[:, 0] = St.eig_vals[:nb_modes]

    print(f"Initial frequencies: {np.around(St.eig_vals[:nb_modes], 3)} rad/s")
    Modal.plot_modes(St, nb_modes, scale=10, title="Plastic Frame - Initial Modes")

    # ==========================================================================
    # Incremental Pushover with Modal Tracking
    # ==========================================================================
    print(f"\nRunning incremental pushover ({n_incr} increments)...")

    LIST = [0]
    P = np.array([])
    U = np.array([])

    for i in range(n_incr):
        # Reload structure from saved state
        with open(struct_file, 'rb') as file:
            St = pickle.load(file)

        # Create displacement list for this increment
        LIST_D = np.linspace(LIST[-1], INCR * (i + 1), STEPS).tolist()

        # Displacement control analysis
        disp_file = f'Step_{i+1}_DispControl'
        St = StaticNonLinear.solve_dispcontrol(
            St, LIST_D, 0, control_node, 0,
            tol=1,
            filename=disp_file,
            dir_name=save_path
        )

        # Save structure state
        St.save_structure(struct_file)

        # Modal analysis at current deformed state
        St = Modal.solve(St, nb_modes, filename=os.path.join(save_path, f'Step_{i+1}_Modal'))
        w[:, i + 1] = St.eig_vals[:nb_modes]
        print(f"Step {i+1}: frequencies = {np.around(St.eig_vals[:nb_modes], 3)} rad/s")

        # Update displacement list for next increment
        LIST = LIST_D

    # ==========================================================================
    # Collect Force-Displacement Results
    # ==========================================================================
    for i in range(1, n_incr + 1):
        results = os.path.join(save_path, f'Step_{i}_DispControl.h5')
        if os.path.exists(results):
            with h5py.File(results, 'r') as hf:
                P = np.append(P, hf['P_r_conv'][3 * control_node] / 1000)
                U = np.append(U, hf['U_conv'][3 * control_node] * 1000)

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    # Force-displacement curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    ax1.plot(U, P, '-o', markerfacecolor='white', color='black', markersize=3)
    ax1.set_xlabel('Top horizontal displacement [mm]')
    ax1.set_ylabel('Applied Force [kN]')
    ax1.set_title('Plastic Frame - Pushover Curve')
    ax1.grid(True)

    # Modal degradation
    steps = np.arange(n_incr + 1)
    for mode in range(nb_modes):
        ax2.plot(steps, w[mode, :] / w[mode, 0] * 100, '-o',
                label=f'Mode {mode+1}', markersize=3)
    ax2.set_xlabel('Load step')
    ax2.set_ylabel('Relative frequency [%]')
    ax2.set_title('Modal Degradation')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'Plastic_Frame_Results.png'), dpi=150)

    # Save results
    with h5py.File(os.path.join(save_path, 'Results_Modal_Deg.h5'), 'w') as hf:
        hf.create_dataset('U', data=U)
        hf.create_dataset('P', data=P)
        hf.create_dataset('w', data=w)

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Plastic Frame - Modal Degradation Analysis Complete")
    print("="*60)
    print(f"Frame dimensions: 3m x 3m")
    print(f"Blocks per member: {BLOCKS}")
    print(f"Material: E = {E/1e9} GPa, fy = {FY/1e6} MPa")
    print(f"Number of increments: {n_incr}")
    print(f"Max displacement: {INCR * n_incr * 1000:.1f} mm")
    print(f"Final frequency ratios: {np.around(w[:, -1] / w[:, 0] * 100, 1)}%")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
