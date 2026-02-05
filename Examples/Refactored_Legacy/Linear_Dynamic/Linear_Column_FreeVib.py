# -*- coding: utf-8 -*-
"""
Linear Column Free Vibration
=============================

Refactored from: Legacy/Examples/Linear_Dynamic/Linear_Column_Vibrating/Linear_Column_FreeVib.py

This example demonstrates linear dynamic analysis of a cantilever column
under initial displacement, testing the free vibration response using
different time integration methods.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Linear dynamic (free vibration from initial displacement)

Configuration:
- Cantilever beam: 6 blocks, 4m length
- Cross-section: 0.1m x 0.1m
- Material: E = 4*pi^2 (for convenient frequency)
- Initial displacement from static load
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Structure_Block, Static, Dynamic
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
    # Time integration method options:
    # 'CDM' - Central Difference Method
    # 'LA' - Linear Acceleration (Newmark)
    # ['HHT', alpha] - Hilber-Hughes-Taylor
    # ['WIL', theta] - Wilson-theta
    # ['GEN', alpha] - Generalized alpha
    Meth = ['GEN', 0.5]  # Generalized alpha method

    # Geometry
    N1 = np.array([0, 0], dtype=float)
    N2 = np.array([4, 0], dtype=float)

    H = 0.1    # Cross-section height [m]
    B = 0.1    # Cross-section width [m]

    BLOCKS = 6
    CPS = 25   # Contact points per interface

    # Material (E chosen for convenient frequency)
    E = 4 * np.pi ** 2
    NU = 0.0
    RHO = 2.0

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    material = Material(E, NU, rho=RHO, shear_def=True)

    # Create beam structure
    Beam_temp = BeamBlock(N1, N2, BLOCKS, H, RHO, b=B, material=material)

    St = Structure_Block()
    for block in Beam_temp.list_blocks:
        St.list_blocks.append(block)

    St.make_nodes()
    St.make_cfs(lin_geom=True, nb_cps=CPS)

    # ==========================================================================
    # Boundary Conditions and Loading
    # ==========================================================================
    # Apply tip load for initial displacement
    F = -E * 1e-3  # Small load for linear response

    # Find nodes at N1 (fixed) and N2 (loaded)
    N1_idx = St.get_node_at(N1)
    N2_idx = St.get_node_at(N2)

    St.load_node(N2_idx, [1], F, fixed=True)  # Tip load
    St.fix_node(N1_idx, [0, 1, 2])  # Fixed support

    # ==========================================================================
    # Static Analysis (for initial displacement)
    # ==========================================================================
    print("Running static analysis for initial displacement...")
    St = Static.solve_linear(St)

    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=False,
            title="Linear Column - Initial Displacement")

    # Store initial displacement
    U0 = St.U.copy()

    # ==========================================================================
    # Dynamic Analysis (Free Vibration)
    # ==========================================================================
    print("\nRunning linear dynamic analysis (free vibration)...")
    print(f"  Time integration method: {Meth}")
    print(f"  Duration: 5 s")
    print(f"  Time step: 0.001 s")

    # Set damping (zero for undamped free vibration)
    St.set_damping_properties(xsi=0.0, damp_type='STIFF')

    St = Dynamic.solve_dyn_linear(
        St, 5, 1e-3,
        U0=U0,
        Meth=Meth,
        filename='Column_FreeVib',
        dir_name=save_path
    )

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    result_file = os.path.join(save_path, 'Column_FreeVib.h5')
    if os.path.exists(result_file):
        with h5py.File(result_file, 'r') as hf:
            time = hf['Time'][:]
            U = hf['U_conv'][3 * N2_idx + 1, :]  # Vertical displacement at tip

        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(time, U * 1000, 'b-', linewidth=0.5, label='Tip displacement')
        plt.xlabel('Time [s]')
        plt.ylabel('Vertical displacement [mm]')
        plt.title('Linear Column - Free Vibration Response')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, 'column_freevib.png'), dpi=150)

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Linear Column Free Vibration Analysis Complete")
    print("="*60)
    print(f"Beam length: {np.linalg.norm(N2 - N1)} m")
    print(f"Number of blocks: {BLOCKS}")
    print(f"Cross-section: {H}m x {B}m")
    print(f"Young's modulus: E = 4*pi^2 = {E:.4f}")
    print(f"Time integration: {Meth}")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
