# -*- coding: utf-8 -*-
"""
Linear Crane Free Vibration
===========================

Refactored from: Legacy/Examples/Linear_Dynamic/Linear_Crane/Linear_Crane_FreeVib.py

This example demonstrates modal and linear dynamic analysis of an L-shaped
crane structure with harmonic excitation.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Modal + Linear dynamic (forced vibration)

Configuration:
- L-shaped crane: vertical column + horizontal arm
- Each member: 10 blocks, 7m length
- Cross-section: 0.124m x 0.124m (HEB 120 equivalent)
- Steel material: E = 210 GPa
- 5% Rayleigh damping
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Structure_Block, Modal, Dynamic
from Core.Structures import BeamBlock
from Core.Objects.ConstitutiveLaw.Material import Material


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
    # Time integration method
    Meth = ['HHT', 0.3]  # Hilber-Hughes-Taylor with alpha=0.3

    # Geometry - L-shaped crane
    N1 = np.array([0, 0], dtype=float)   # Base
    N2 = np.array([0, 7], dtype=float)   # Corner
    N3 = np.array([7, 7], dtype=float)   # Tip

    H = 0.124    # Cross-section height [m] (HEB 120)
    B = 0.124    # Cross-section width [m]

    BLOCKS = 10   # Blocks per member
    CPS = 100     # Contact points per interface

    # Material - Steel
    E = 210e9    # Young's modulus [Pa]
    NU = 0.0     # Poisson's ratio
    RHO = 5574.0  # Equivalent density [kg/m³]

    # Excitation parameters
    F_ref = 500   # Reference force [N]
    w_s = 10      # Excitation frequency [rad/s]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    material = Material(E, NU, rho=RHO, shear_def=True)

    # Create L-shaped structure from two beam segments
    St = Structure_Block()

    # Vertical column (N1 to N2)
    Column_temp = BeamBlock(N1, N2, BLOCKS, H, RHO, b=B, material=material)
    for block in Column_temp.list_blocks:
        St.list_blocks.append(block)

    # Horizontal arm (N2 to N3)
    Arm_temp = BeamBlock(N2, N3, BLOCKS, H, RHO, b=B, material=material)
    for block in Arm_temp.list_blocks:
        St.list_blocks.append(block)

    St.make_nodes()
    St.make_cfs(lin_geom=True, nb_cps=CPS)

    # ==========================================================================
    # Boundary Conditions and Loading
    # ==========================================================================
    # Find nodes
    N1_idx = St.get_node_at(N1)
    N3_idx = St.get_node_at(N3)

    # Apply harmonic loading at tip
    St.load_node(N3_idx, [1], F_ref)  # Vertical force at tip
    St.fix_node(N1_idx, [0, 1, 2])    # Fixed base

    # ==========================================================================
    # Modal Analysis
    # ==========================================================================
    print("Running modal analysis...")
    nb_modes = 4
    St = Modal.solve(St, nb_modes, filename=os.path.join(save_path, 'Crane_Modal'))

    if St.eig_vals is not None and len(St.eig_vals) >= 2:
        freqs = np.sqrt(np.abs(St.eig_vals[:nb_modes])) / (2 * np.pi)
        print(f"Natural frequencies: {freqs} Hz")
        Modal.plot_modes(St, 2, scale=10, title="Crane Mode Shapes")

    # ==========================================================================
    # Save Structure
    # ==========================================================================
    St.save_structure(filename=os.path.join(save_path, 'Crane'))

    # ==========================================================================
    # Dynamic Analysis Setup (Optional - uncomment to run)
    # ==========================================================================
    # Define excitation function
    def excitation(t):
        return -np.sin(w_s * t)

    # Set Rayleigh damping (5%)
    St.set_damping_properties(xsi=0.05, damp_type='RAYLEIGH')

    # Uncomment below to run dynamic analysis:
    # print("\nRunning linear dynamic analysis (forced vibration)...")
    # print(f"  Excitation frequency: {w_s} rad/s")
    # print(f"  Duration: 20 s")
    # print(f"  Time step: 0.01 s")
    #
    # St = Dynamic.solve_dyn_linear(
    #     St, 20, 1e-2,
    #     lmbda=excitation,
    #     Meth=Meth,
    #     filename='Crane_Dynamic',
    #     dir_name=save_path
    # )

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Linear Crane Analysis Complete")
    print("="*60)
    print(f"Structure: L-shaped crane")
    print(f"  Column: {N1} to {N2}")
    print(f"  Arm: {N2} to {N3}")
    print(f"Number of blocks: {2 * BLOCKS}")
    print(f"Cross-section: {H}m x {B}m")
    print(f"Material: Steel (E = {E/1e9} GPa)")
    print(f"Damping: 5% Rayleigh")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
