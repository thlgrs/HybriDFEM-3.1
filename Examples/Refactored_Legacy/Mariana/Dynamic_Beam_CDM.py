# -*- coding: utf-8 -*-
"""
Dynamic Beam with Mixed Hardening Material (CDM)
================================================

Refactored from: Legacy/Examples/Mariana/Dynamic_Beam_CDM.py

This example demonstrates nonlinear dynamic analysis of a cantilever beam
with mixed hardening material behavior using the Central Difference Method (CDM).

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear dynamic (CDM)

Configuration:
- Cantilever beam: h=0.015m, L=8h=0.12m
- 10 blocks
- Mixed hardening material: E=193.3 GPa, fy=580 MPa, H=15.4 GPa
- Sinusoidal loading at tip
- No damping
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Structure_Block
from Core.Structures import BeamBlock
from Core.Solvers.Dynamic import Dynamic
from Core.Objects.ConstitutiveLaw.Material import Mixed_Hardening_Mat


def main():
    # ==========================================================================
    # Output Directory Setup
    # ==========================================================================
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ==========================================================================
    # Geometry Parameters
    # ==========================================================================
    h = 0.015         # Section depth [m]
    a = 8 * h         # Beam length [m] = 0.12m
    t = 0.0135        # Section width [m]
    n_blocks = 10     # Number of blocks

    N1 = np.array([0.0, 0.0])  # Base (fixed end)
    N2 = np.array([0.0, a])    # Tip (loaded end)

    # ==========================================================================
    # Material Parameters (Mixed Hardening)
    # ==========================================================================
    E = 193333.33e6   # Young's modulus [Pa]
    FY = 580e6        # Yield stress [Pa]
    H = 15425.53e6    # Hardening modulus [Pa]
    r = 0.025         # Kinematic/isotropic hardening ratio
    NU = 0.3          # Poisson's ratio

    material = Mixed_Hardening_Mat(E, NU, FY, H, r)

    # ==========================================================================
    # Loading Parameters
    # ==========================================================================
    # Maximum force based on section modulus and yield moment
    Wb = t * h ** 2 / 6       # Section modulus [m³]
    My = Wb * FY               # Yield moment [N·m]
    Pmax = 2 * My / a          # Maximum force [N]

    print(f"\nBeam Parameters:")
    print(f"  Length: {a * 1000:.1f} mm")
    print(f"  Section: {h * 1000:.1f} x {t * 1000:.1f} mm")
    print(f"  Yield moment: {My:.2f} N·m")
    print(f"  Maximum force: {Pmax:.2f} N")

    # Sinusoidal load parameters
    T = 1.0              # Total time [s]
    n_steps = 1000       # Number of time steps
    omega = 10 * np.pi   # Angular frequency [rad/s]
    dt = T / n_steps     # Time step [s]

    time_vec = np.linspace(0, T, n_steps + 1)
    lambda_list = list(np.sin(omega * time_vec))

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = BeamBlock(N1, N2, n_blocks, h, rho=7800, b=t, material=material)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces
    St.make_cfs(lin_geom=True, nb_cps=10)

    # ==========================================================================
    # Boundary Conditions and Loading
    # ==========================================================================
    # Fixed base
    St.fix_node(N1, [0, 1, 2])

    # Apply tip force (will be scaled by lambda_list)
    St.reset_loading()
    St.load_node(N2, [0], Pmax)

    # Plot initial structure
    St.plot(show_contact_faces=True, title="Dynamic Beam - Initial")

    # ==========================================================================
    # Dynamic Analysis (CDM)
    # ==========================================================================
    filename = "Dynamic_Beam_CDM"

    print(f"\nRunning nonlinear dynamic analysis...")
    print(f"  Method: CDM (Central Difference Method)")
    print(f"  Duration: {T}s")
    print(f"  Time step: {dt * 1000:.3f}ms")
    print(f"  Frequency: {omega / (2 * np.pi):.2f} Hz")

    # Set damping (no damping)
    St.set_damping_properties(xsi=0.0)

    start_time = time.time()

    Dyn = Dynamic(St)
    St = Dyn.nonlinear(
        t_end=T,
        dt=dt,
        Meth='CDM',
        lmbda=lambda_list,
        filename=filename,
        dir_name=save_path
    )

    elapsed = time.time() - start_time
    print(f"  Simulation time: {elapsed:.2f} seconds")

    # Save structure
    St.save_structure(os.path.join(save_path, filename))

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    # Read results and extract tip displacement
    results_file = os.path.join(save_path, f"{filename}_CDM.h5")
    if os.path.exists(results_file):
        with h5py.File(results_file, 'r') as hf:
            U_all = hf['U_conv'][-3, :]  # X displacement at top node

        # Plot force-displacement curve (hysteresis loop)
        plt.figure(figsize=(8, 6))
        plt.plot(U_all * 1e3, np.array(lambda_list) * Pmax / 1e3, '-',
                 color='black', linewidth=1.2)
        plt.xlabel('Displacement [mm]')
        plt.ylabel('Force [kN]')
        plt.title('Force-Displacement Hysteresis - Dynamic Beam')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'hysteresis_curve.png'), dpi=150)

    # Plot final deformed structure
    St.plot(show_deformed=True, deformation_scale=10, show_contact_faces=True,
            title="Dynamic Beam - Deformed")

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Dynamic Analysis Results - Mixed Hardening Beam")
    print("="*60)
    print(f"Beam length: {a * 1000:.1f} mm")
    print(f"Number of blocks: {n_blocks}")
    print(f"Material: Mixed hardening (E={E/1e9:.1f}GPa, fy={FY/1e6:.0f}MPa)")
    print(f"Load type: Sinusoidal (f={omega/(2*np.pi):.1f}Hz)")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
