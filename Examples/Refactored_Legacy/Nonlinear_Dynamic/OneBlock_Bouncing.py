# -*- coding: utf-8 -*-
"""
One Block Bouncing
==================

Refactored from: Legacy/Examples/Nonlinear_Dynamic/OneBlock/OneBlock_Bouncing.py

This example demonstrates nonlinear dynamic analysis of a single block
bouncing on a fixed base with no-tension contact, starting from an
initial horizontal displacement.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear dynamic (Newmark/CDM time integration)

Configuration:
- Single rectangular block (0.4m x 0.4m) on fixed base
- No-tension contact (kn=ks=1e8 N/m)
- Initial horizontal displacement: 0.1m
- Undamped free vibration
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
from Core import Structure_Block, Dynamic
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.ConstitutiveLaw.Contact import NoTension_CD


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
    Meth = 'NWK'  # Newmark method (can also use 'CDM')

    # Contact stiffness
    kn = 1e8      # Normal stiffness [N/m]
    ks = kn       # Shear stiffness [N/m]

    # Block dimensions
    H = 0.4       # Block height [m]
    L = 0.4       # Block length [m]
    B = 1.0       # Out-of-plane thickness [m]

    rho = 1000    # Density [kg/m³]

    # Base dimensions
    L_base = 1.0  # Base length [m]
    H_base = 0.2  # Base height [m]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=rho)

    # Reference points
    N1 = np.array([0, -H_base / 2], dtype=float)  # Base center
    N2 = np.array([0, H / 2], dtype=float)        # Block center (elevated)

    x = np.array([0.5, 0])
    y = np.array([0, 0.5])

    # Base block vertices
    vertices_base = np.array([
        N1 + L_base * x - H_base * y,
        N1 + L_base * x + H_base * y,
        N1 - L_base * x + H_base * y,
        N1 - L_base * x - H_base * y
    ])
    St.add_block_from_vertices(vertices_base, b=B, material=material)

    # Upper block vertices
    vertices_block = np.array([
        N2 + L * x - H * y,
        N2 + L * x + H * y,
        N2 - L * x + H * y,
        N2 - L * x - H * y
    ])
    St.add_block_from_vertices(vertices_block, b=B, material=material)

    # Initialize structure
    St.make_nodes()

    # Create contact faces with no-tension contact
    St.make_cfs(lin_geom=False, nb_cps=2, offset=0.0, contact=NoTension_CD(kn, ks))

    # ==========================================================================
    # Boundary Conditions and Loading
    # ==========================================================================
    # Get block properties
    M = St.list_blocks[1].m
    W = 9.81 * M
    I = St.list_blocks[1].I

    print(f"Upper block weight: {W:.2f} N")
    print(f"Rotational inertia: {I:.6f} kg.m²")

    # Apply self-weight to upper block
    St.load_node(1, [1], -W, fixed=True)

    # Fix base block
    St.fix_node(0, [0, 1, 2])

    # ==========================================================================
    # Initial Conditions
    # ==========================================================================
    # Initial horizontal displacement
    U0 = np.zeros(6)
    U0[4] = 0.1  # Horizontal displacement of upper block

    # ==========================================================================
    # Dynamic Analysis
    # ==========================================================================
    print(f"\nRunning nonlinear dynamic analysis...")
    print(f"  Method: {Meth}")
    print(f"  Duration: 10 s")
    print(f"  Time step: 0.001 s")
    print(f"  Initial displacement: {U0[4]} m")

    # Zero damping (undamped)
    St.set_damping_properties(xsi=0.00, damp_type='STIFF')

    St = Dynamic.solve_dyn_nonlinear(
        St, 10, 1e-3,
        Meth=Meth,
        U0=U0,
        filename='OneBlock_Bouncing',
        dir_name=save_path
    )

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=True,
            title="One Block Bouncing - Final Position")

    # Plot time history
    result_file = os.path.join(save_path, 'OneBlock_Bouncing.h5')
    if os.path.exists(result_file):
        with h5py.File(result_file, 'r') as hf:
            time = hf['Time'][:]
            U_x = hf['U_conv'][3, :]  # Horizontal displacement (block DOF 0)
            U_y = hf['U_conv'][4, :]  # Vertical displacement (block DOF 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)

        ax1.plot(time, U_x * 1000, 'b-', linewidth=0.5)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Horizontal displacement [mm]')
        ax1.set_title('One Block Bouncing - Horizontal Motion')
        ax1.grid(True)

        ax2.plot(time, U_y * 1000, 'r-', linewidth=0.5)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Vertical displacement [mm]')
        ax2.set_title('One Block Bouncing - Vertical Motion')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'oneblock_bouncing.png'), dpi=150)

    # Save structure
    St.save_structure(os.path.join(save_path, 'Rocking_block'))

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("One Block Bouncing Analysis Complete")
    print("="*60)
    print(f"Block dimensions: {L}m x {H}m x {B}m")
    print(f"Block mass: {M:.2f} kg")
    print(f"Contact stiffness: kn = ks = {kn:.2e} N/m")
    print(f"Initial horizontal displacement: {U0[4]*1000:.0f} mm")
    print(f"Time integration: {Meth}")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
