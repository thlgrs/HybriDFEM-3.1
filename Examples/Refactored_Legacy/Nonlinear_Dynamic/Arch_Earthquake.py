# -*- coding: utf-8 -*-
"""
Arch Earthquake - Nonlinear Dynamic Analysis
============================================

Refactored from: Legacy/Examples/Nonlinear_Dynamic/Lemos_Arch_Pulse/Arch_Earthquake.py

This example demonstrates nonlinear dynamic analysis of a masonry arch
subject to earthquake ground motion, based on Lemos' arch studies.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear dynamic (Newmark method)

Configuration:
- Semi-circular arch: R=3.75m mean radius, 17 voussoirs
- Block thickness: 0.5m radial, 1.0m out-of-plane
- Fixed base support
- No-tension contact with contact deletion (NoTension_CD)
- Earthquake excitation with 5% stiffness proportional damping
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
from Core import Structure_Block, StaticNonLinear
from Core.Structures import ArchBlock
from Core.Solvers.Dynamic import Dynamic
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
    kn = 20e9       # Normal stiffness [N/m]
    ks = 0.4 * kn   # Shear stiffness [N/m]
    mu = 0.88       # Friction coefficient (not used with NoTension_CD)

    # Arch geometry
    B = 1.0         # Out-of-plane thickness [m]
    H = 0.5         # Radial thickness [m]
    R = 7.5 / 2 + H / 2  # Mean radius [m]

    nb_blocks = 17  # Number of voussoirs

    # Material properties
    rho = 2700  # Density [kg/m³]

    # Base block dimensions
    H_base = 0.5             # Base height [m]
    L_base = 1.1 * 2 * R     # Base width [m]

    # Reference point
    N1 = np.array([0, 0], dtype=float)
    x = np.array([0.5, 0])
    y = np.array([0, 1])

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=rho)

    # Add base block (fixed support)
    vertices = np.array([N1, N1, N1, N1])
    vertices[0] += L_base * x - H_base * y
    vertices[1] += L_base * x
    vertices[2] += -L_base * x
    vertices[3] += -L_base * x - H_base * y
    St.add_block_from_vertices(vertices, b=B, material=material)

    # Add arch using ArchBlock generator
    # Note: We'll build the arch manually to match the Legacy approach
    d_a = np.pi / nb_blocks  # Angle increment
    angle = 0

    R_int = R - H / 2
    R_out = R + H / 2

    for i in range(nb_blocks):
        c = N1  # Center at origin
        unit_dir_1 = np.array([np.cos(angle), np.sin(angle)])
        unit_dir_2 = np.array([np.cos(angle + d_a), np.sin(angle + d_a)])

        vertices = np.array([
            c + R_int * unit_dir_1,
            c + R_out * unit_dir_1,
            c + R_out * unit_dir_2,
            c + R_int * unit_dir_2
        ])
        St.add_block_from_vertices(vertices, b=B, material=material)
        angle += d_a

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with no-tension contact (with deletion)
    # Note: Use 'surface=' parameter when specifying list-based contact point positions
    St.make_cfs(
        lin_geom=False,
        nb_cps=[-1, 0, 1],
        surface=NoTension_CD(kn, ks),
        offset=-1
    )

    # Change first two contact faces (base-arch interface) to simpler contact
    St.list_cfs[0].change_cps(nb_cp=[-1, 1], offset=-1, surface=NoTension_CD(kn, ks))
    St.list_cfs[1].change_cps(nb_cp=[-1, 1], offset=-1, surface=NoTension_CD(kn, ks))

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix base block
    St.fix_node(0, [0, 1, 2])

    # Apply gravity to all arch blocks
    for i in range(1, nb_blocks + 1):
        M = St.list_blocks[i].m
        W = M * 9.81
        St.load_node(i, [1], -W)

    # ==========================================================================
    # Static Gravity Analysis
    # ==========================================================================
    print("\nApplying gravity loads (static analysis)...")
    St = StaticNonLinear.solve_forcecontrol(
        St, steps=5,
        filename='Arch_Gravity',
        dir_name=save_path,
        tol=1e-3,
        max_iter=100
    )

    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=True,
            title="Arch - After Gravity")

    # Reset loading and apply fixed loads for dynamic
    St.reset_loading()

    for i in range(nb_blocks):
        M = St.list_blocks[i].m
        W = M * 9.81
        St.load_node(i, [1], -W, fixed=True)
        St.load_node(i, [0], -W)  # Horizontal inertia forces

    # ==========================================================================
    # Generate Synthetic Earthquake Record
    # ==========================================================================
    print("\nGenerating synthetic earthquake record...")

    # Create a simple synthetic earthquake record
    # (Instead of loading from file, generate a representative motion)
    nb_points = 3005
    dt_eq = 5e-3
    t_end_eq = dt_eq * nb_points
    time = np.arange(nb_points) * dt_eq

    # Generate synthetic velocity using filtered noise
    np.random.seed(42)  # For reproducibility

    # Create modulated random motion
    f0 = 2.0  # Dominant frequency [Hz]
    omega = 2 * np.pi * f0
    envelope = np.exp(-((time - 5)**2) / 10)  # Gaussian envelope
    vlocy = envelope * np.sin(omega * time + np.random.randn(len(time)) * 0.5)

    # Compute acceleration as gradient of velocity
    acc = np.gradient(vlocy, dt_eq)

    # Scale to desired peak acceleration
    peak = 0.2  # Peak ground acceleration [g]
    lmbda_arr = acc * peak / np.max(np.abs(acc))

    # Interpolate to desired time step
    dt_new = 1e-4
    new_time = np.arange(time[0], time[-1], dt_new)

    from scipy.interpolate import interp1d
    interpolator = interp1d(time, lmbda_arr, kind="linear", fill_value="extrapolate")
    new_lmbda = interpolator(new_time)

    # Plot earthquake record
    plt.figure(figsize=(10, 3))
    plt.plot(new_time, new_lmbda, 'b-', linewidth=0.5)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [g]')
    plt.title(f'Synthetic Earthquake Record (PGA = {peak}g)')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'earthquake_record.png'), dpi=150)

    print(f"  Peak ground acceleration: {peak}g")
    print(f"  Duration: {t_end_eq}s")

    # ==========================================================================
    # Set Damping Properties
    # ==========================================================================
    St.set_damping_properties(xsi=0.05, damp_type='STIFF')

    # ==========================================================================
    # Nonlinear Dynamic Analysis
    # ==========================================================================
    print(f"\nRunning nonlinear dynamic analysis...")
    print(f"  Method: Newmark (NWK)")
    print(f"  Duration: 10s")
    print(f"  Time step: {dt_new}s")

    Dyn = Dynamic(St)
    St = Dyn.nonlinear(
        t_end=10,  # Reduced from 16s for faster execution
        dt=dt_new,
        Meth='NWK',
        lmbda=new_lmbda.tolist(),
        filename=f'Arch_EQ_{peak}g',
        dir_name=save_path
    )

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=True,
            title="Arch - After Earthquake")

    # Save structure
    St.save_structure(os.path.join(save_path, 'Arch_Earthquake'))

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Nonlinear Dynamic Results - Arch Earthquake")
    print("="*60)
    print(f"Arch radius: {R}m")
    print(f"Number of voussoirs: {nb_blocks}")
    print(f"Contact stiffness: kn={kn/1e9}GPa, ks={ks/1e9}GPa")
    print(f"Peak ground acceleration: {peak}g")
    print(f"Damping: 5% stiffness proportional")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
