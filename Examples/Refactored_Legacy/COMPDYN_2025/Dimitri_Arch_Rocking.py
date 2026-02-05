# -*- coding: utf-8 -*-
"""
Dimitri Arch Rocking - COMPDYN 2025
===================================

Refactored from: Legacy/Examples/COMPDYN_2025/Dimitri_Arch/Arch_Rocking.py

This example demonstrates nonlinear dynamic analysis of a masonry arch
with complex geometry (irregular voussoirs and abutments) subjected to
horizontal base excitation.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear dynamic (HHT method)

Configuration:
- Arch with 12 voussoirs plus 2 irregular abutment blocks
- Fixed base support
- No-tension contact at interfaces
- Half-sine pulse horizontal excitation
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
from Core.Solvers.Dynamic import Dynamic
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.ConstitutiveLaw.Contact import NoTension_EP


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
    # Integration method
    Meth = ['HHT', 0.3]

    # Contact stiffness
    kn = 1e8       # Normal stiffness [N/m]
    ks = kn        # Shear stiffness [N/m]

    # Material properties
    rho = 2620.0   # Density [kg/m³]

    # Base block dimensions
    L_base = 40    # Base width [m]
    H_base = 2     # Base height [m]

    # Direction vectors
    x = np.array([0.5, 0])
    y = np.array([0, 0.5])

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=rho)

    # --- Base block (fixed support) ---
    N1 = np.array([0, -H_base / 2], dtype=float)
    vertices = np.array([N1, N1, N1, N1])
    vertices[0] += L_base * x - H_base * y
    vertices[1] += L_base * x + H_base * y
    vertices[2] += -L_base * x + H_base * y
    vertices[3] += -L_base * x - H_base * y
    St.add_block_from_vertices(vertices, b=1.0, material=material)

    # --- Left abutment (irregular polygon) ---
    vertices = np.array([
        [-10, 0],
        [-10, 16.4],
        [-10 + 10 * (1 - np.cos(np.pi / 12)), 16.4 + 10 * np.sin(np.pi / 12)],
        [-10 + 10 * (1 - np.cos(np.pi / 6)), 16.4 + 10 * np.sin(np.pi / 6)],
        [-10, 16.4 + 10 * np.tan(np.pi / 6)],
        [-10, 30],
        [-15, 30],
        [-15, 0]
    ], dtype=float)
    St.add_block_from_vertices(vertices, b=1.0, material=material)

    # --- Right abutment (mirrored) ---
    vertices_right = vertices.copy()
    vertices_right[:, 0] *= -1
    vertices_right = np.flip(vertices_right, axis=0)
    St.add_block_from_vertices(vertices_right, b=1.0, material=material)

    # --- Arch voussoirs (12 blocks) ---
    C = np.array([0, 16.4], dtype=float)  # Arch center
    a1 = np.pi / 6                         # Start angle
    a2 = 5 * np.pi / 6                     # End angle
    R = 10 + 0.75                          # Mean radius
    h = 1.5                                # Radial thickness
    n_blocks = 12                          # Number of voussoirs

    d_a = (a2 - a1) / n_blocks
    R_int = R - h / 2
    R_out = R + h / 2
    angle = a1

    for i in range(n_blocks):
        unit_dir_1 = np.array([np.cos(angle), np.sin(angle)])
        unit_dir_2 = np.array([np.cos(angle + d_a), np.sin(angle + d_a)])

        vertices = np.array([
            C + R_int * unit_dir_1,
            C + R_out * unit_dir_1,
            C + R_out * unit_dir_2,
            C + R_int * unit_dir_2
        ])
        St.add_block_from_vertices(vertices, b=1.0, material=material)
        angle += d_a

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with no-tension contact
    # nb_cps=2 is integer, so use contact= parameter
    St.make_cfs(
        lin_geom=False,
        nb_cps=2,
        contact=NoTension_EP(kn, ks),
        offset=0.0
    )

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix base block
    St.fix_node(0, [0, 1, 2])

    # Apply horizontal inertia and gravity loads to all arch blocks
    for i in range(1, len(St.list_blocks)):
        M = St.list_blocks[i].m
        W = 9.81 * M
        St.load_node(i, [0], W)              # Horizontal inertia (for dynamic)
        St.load_node(i, [1], -W, fixed=True) # Self-weight (fixed)

    # Plot initial structure
    St.plot(show_deformed=True, deformation_scale=10, title="Dimitri Arch - Initial")

    # ==========================================================================
    # Excitation Function and Damping
    # ==========================================================================
    t_p = 0.2    # Pulse period [s]
    a = 0.4      # Amplitude [g]
    lag = 0.01   # Time lag [s]

    def lmbda(x):
        """Half-sine pulse excitation function."""
        if x < lag:
            return 0
        if x < t_p + lag:
            return a
        elif x < 3 * t_p + lag:
            return -a / 2
        else:
            return 0

    # Set damping (Rayleigh - no damping for rocking)
    St.set_damping_properties(xsi=0.00, damp_type='RAYLEIGH')

    # ==========================================================================
    # Nonlinear Dynamic Analysis
    # ==========================================================================
    print(f"\nRunning nonlinear dynamic analysis (Dimitri Arch)...")
    print(f"  Method: HHT (alpha=0.3)")
    print(f"  Duration: 10s")
    print(f"  Time step: 1e-3s")
    print(f"  Pulse: period={t_p}s, amplitude={a}g")

    Dyn = Dynamic(St)
    St = Dyn.nonlinear(
        t_end=10,
        dt=1e-3,
        Meth=Meth,
        lmbda=lmbda,
        filename=f't_p={t_p}_a={a}',
        dir_name=save_path
    )

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=False,
            title="Dimitri Arch - After Rocking")

    # Save structure
    St.save_structure(os.path.join(save_path, 'Dimitri_Arch'))

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Nonlinear Dynamic Results - Dimitri Arch")
    print("="*60)
    print(f"Number of blocks: {len(St.list_blocks)}")
    print(f"Pulse period: {t_p}s")
    print(f"Pulse amplitude: {a}g")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
