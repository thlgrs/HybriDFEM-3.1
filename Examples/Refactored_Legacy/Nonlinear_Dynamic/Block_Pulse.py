# -*- coding: utf-8 -*-
"""
Block Pulse Response
====================

Refactored from: Legacy/Examples/Nonlinear_Dynamic/OneBlock_Pulse/Block_Pulse.py

This example demonstrates nonlinear dynamic analysis of a block subjected
to a sinusoidal pulse base excitation, simulating earthquake-like loading.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear dynamic (Newmark time integration)

Configuration:
- Single rectangular block (0.5m x 1.0m) on fixed base
- No-tension contact (kn=ks=1e9 N/m)
- Sinusoidal pulse excitation: 0.66g amplitude, 0.25s half-period
- Rayleigh damping
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
from Core import Structure_Block, StaticNonLinear, Dynamic
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
    # Contact stiffness
    kn = 1e9      # Normal stiffness [N/m]
    ks = 1e9      # Shear stiffness [N/m]

    # Block dimensions
    B = 1.0       # Out-of-plane thickness [m]
    H = 1.0       # Block height [m]
    L = 0.5       # Block length [m]

    rho = 2700    # Density [kg/m³]

    # Base dimensions
    L_base = 1.0  # Base length [m]
    H_base = 0.5  # Base height [m]

    # Excitation parameters
    t_p = 0.25    # Half-period [s]
    w_s = np.pi / t_p  # Angular frequency
    a = 0.66      # Amplitude [g]
    lag = 0.0     # Time lag [s]

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
    # Using nb_cps as list requires surface= parameter
    cps_list = np.linspace(-1, 1, 20).tolist()
    St.make_cfs(lin_geom=False, nb_cps=cps_list, offset=-1, surface=NoTension_CD(kn, ks))

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix base block
    St.fix_node(0, [0, 1, 2])

    # Get block mass
    M = St.list_blocks[1].m
    W = M * 9.81

    # Apply self-weight
    St.load_node(1, [1], -W)

    # ==========================================================================
    # Static Initialization (settle under gravity)
    # ==========================================================================
    print("Running static initialization (gravity settling)...")
    St = StaticNonLinear.solve_forcecontrol(St, 2, tol=0.1)

    # Reset loading for dynamic analysis
    St.reset_loading()

    # Reapply loads for dynamic analysis
    St.load_node(1, [1], -W, fixed=True)  # Self-weight (constant)
    St.load_node(1, [0], -W)              # Horizontal load (scaled by lambda)

    St.plot(show_deformed=False, show_contact_faces=True,
            title="Block Pulse - Initial (after settling)")

    # ==========================================================================
    # Define Excitation Function
    # ==========================================================================
    def lmbda(t):
        """Sinusoidal pulse excitation."""
        if t < lag:
            return 0
        if t < 2 * t_p + lag:
            return a * np.sin(w_s * (t - lag))
        return 0

    print(f"\nExcitation parameters:")
    print(f"  Half-period: {t_p} s")
    print(f"  Amplitude: {a}g")
    print(f"  Duration: {2*t_p} s")

    # ==========================================================================
    # Dynamic Analysis
    # ==========================================================================
    # Set Rayleigh damping (zero damping for this example)
    St.set_damping_properties(xsi=[0.0, 0.0], damp_type='RAYLEIGH')

    print(f"\nRunning nonlinear dynamic analysis...")
    print(f"  Method: NWK (Newmark)")
    print(f"  Duration: 10 s")
    print(f"  Time step: 0.0005 s")

    St = Dynamic.solve_dyn_nonlinear(
        St, 10, 5e-4,
        Meth='NWK',
        lmbda=lmbda,
        filename=f'TwoBlocks_{a}g',
        dir_name=save_path
    )

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=True,
            title="Block Pulse - Final Position")

    # Plot time history
    result_file = os.path.join(save_path, f'TwoBlocks_{a}g.h5')
    if os.path.exists(result_file):
        with h5py.File(result_file, 'r') as hf:
            time = hf['Time'][:]
            U_x = hf['U_conv'][3, :]  # Horizontal displacement
            U_y = hf['U_conv'][4, :]  # Vertical displacement
            R = hf['U_conv'][5, :]    # Rotation

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=150)

        axes[0].plot(time, U_x * 1000, 'b-', linewidth=0.5)
        axes[0].set_ylabel('Horizontal disp. [mm]')
        axes[0].set_title(f'Block Pulse Response ({a}g amplitude)')
        axes[0].grid(True)

        axes[1].plot(time, U_y * 1000, 'r-', linewidth=0.5)
        axes[1].set_ylabel('Vertical disp. [mm]')
        axes[1].grid(True)

        axes[2].plot(time, np.degrees(R), 'g-', linewidth=0.5)
        axes[2].set_xlabel('Time [s]')
        axes[2].set_ylabel('Rotation [deg]')
        axes[2].grid(True)

        # Add excitation overlay
        excitation = np.array([lmbda(t) for t in time])
        ax_exc = axes[0].twinx()
        ax_exc.plot(time, excitation, 'r--', alpha=0.5, linewidth=1, label='Excitation')
        ax_exc.set_ylabel('Excitation [g]', color='r')
        ax_exc.tick_params(axis='y', labelcolor='r')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'block_pulse_{a}g.png'), dpi=150)

    # Save structure
    St.save_structure(os.path.join(save_path, 'TwoBlocks_Pulse'))

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Block Pulse Response Analysis Complete")
    print("="*60)
    print(f"Block dimensions: {L}m x {H}m x {B}m")
    print(f"Block mass: {M:.2f} kg")
    print(f"Contact stiffness: kn = ks = {kn:.2e} N/m")
    print(f"Excitation: {a}g sinusoidal pulse, T = {2*t_p} s")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
