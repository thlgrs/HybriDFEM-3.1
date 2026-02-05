# -*- coding: utf-8 -*-
"""
Column Rocking - Nonlinear Dynamic Analysis
===========================================

Refactored from: Legacy/Examples/Nonlinear_Dynamic/Dimitri_Column/Dimitri_Column_EqInertia/Column_Rocking.py

This example demonstrates nonlinear dynamic analysis of a tapered column
subject to horizontal base excitation, with rocking behavior at block interfaces.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear dynamic (HHT method)

Configuration:
- Tapered column: ~6m height, tapering from 1.11m to 0.9m base width
- 7 main blocks + 1 top cap block
- Fixed base support
- No-tension contact with elastic-plastic behavior (NoTension_EP)
- Half-sine pulse excitation
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
    # Contact stiffness
    kn = 5e10  # Normal stiffness [N/m]
    ks = 5e10  # Shear stiffness [N/m]

    # Column geometry (tapered)
    B1 = 1.11      # Bottom width [m]
    B2 = 0.9       # Top width [m]
    H = 5.95       # Total height [m]
    H_small = 0.21 # Top cap height [m]

    nb_blocks = 7  # Number of main blocks

    # Derived parameters
    dH = (H - H_small) / nb_blocks
    dB = (B1 - B2) / H
    B2_current = B1 - dB * dH

    # Material properties
    rho = 2620  # Density [kg/m³]

    # Base block dimensions
    L_base = 2.5   # Base width [m]
    H_base = 0.75  # Base height [m]

    # Reference points
    N1 = np.array([0, -H_base / 2], dtype=float)  # Base center
    N2 = np.array([0, dH / 2], dtype=float)        # First block
    N3 = np.array([0, H - H_small / 2], dtype=float)  # Top cap
    x = np.array([0.5, 0])  # Half-width direction
    y = np.array([0, 0.5])  # Half-height direction

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=rho)

    # --- Base block (fixed support) ---
    vertices = np.array([N1, N1, N1, N1])
    vertices[0] += L_base * x - H_base * y
    vertices[1] += L_base * x + H_base * y
    vertices[2] += -L_base * x + H_base * y
    vertices[3] += -L_base * x - H_base * y
    St.add_block_from_vertices(vertices, b=1.0, material=material)

    # --- Tapered column blocks ---
    B1_current = B1
    B2_loop = B2_current
    for i in range(nb_blocks):
        vertices = np.array([N2, N2, N2, N2])
        vertices[0] += B1_current * x - dH * y
        vertices[1] += B2_loop * x + dH * y
        vertices[2] += -B2_loop * x + dH * y
        vertices[3] += -B1_current * x - dH * y
        B1_current = B2_loop
        B2_loop -= dB * dH
        N2 += 2 * dH * y
        St.add_block_from_vertices(vertices, b=1.0, material=material)

    # --- Top cap block ---
    B2_cap = B1_current - dB * H_small
    vertices = np.array([N3, N3, N3, N3])
    vertices[0] += B1_current * x - H_small * y
    vertices[1] += B2_cap * x + H_small * y
    vertices[2] += -B2_cap * x + H_small * y
    vertices[3] += -B1_current * x - H_small * y
    St.add_block_from_vertices(vertices, b=1.0, material=material)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with no-tension contact (elastic-plastic)
    # Note: Use 'surface=' parameter when specifying list-based contact point positions
    St.make_cfs(
        lin_geom=False,
        nb_cps=[-1, 1],
        surface=NoTension_EP(kn, ks),
        offset=-1
    )

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix base block
    St.fix_node(0, [0, 1, 2])

    # Apply gravity to all column blocks (fixed loads)
    for i in range(1, nb_blocks + 2):
        M = St.list_blocks[i].m
        W = M * 9.81
        St.load_node(i, [1], -W, fixed=True)

    # ==========================================================================
    # Static Gravity Analysis
    # ==========================================================================
    print("\nApplying gravity loads (static analysis)...")
    St = StaticNonLinear.solve_forcecontrol(
        St, steps=10,
        filename='Column_Gravity',
        dir_name=save_path,
        tol=1e-3
    )

    # Apply horizontal inertia forces (for dynamic)
    for i in range(1, nb_blocks + 2):
        M = St.list_blocks[i].m
        W = M * 9.81
        St.load_node(i, [0], -W)

    # Plot initial structure
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=True,
            title="Column - After Gravity")

    # ==========================================================================
    # Excitation Function and Damping
    # ==========================================================================
    t_p = 0.2   # Pulse period [s]
    a = 1.4     # Amplitude [g]
    lag = 0.0   # Time lag [s]

    print(f"\nPulse excitation: period={t_p}s, amplitude={a}g")

    def lmbda(x):
        """Half-sine pulse excitation function."""
        if x < lag:
            return 0
        if x < t_p + lag:
            return a
        if x < 3 * t_p + lag:
            return -a / 2
        return 0

    # Set damping (Rayleigh)
    St.set_damping_properties(xsi=0.0, damp_type='INIT')

    # ==========================================================================
    # Nonlinear Dynamic Analysis
    # ==========================================================================
    print(f"\nRunning nonlinear dynamic analysis...")
    print(f"  Method: HHT (alpha=0.3)")
    print(f"  Duration: 2s")
    print(f"  Time step: 5e-4s")

    Dyn = Dynamic(St)
    St = Dyn.nonlinear(
        t_end=2,
        dt=5e-4,
        Meth=['HHT', 0.3],
        lmbda=lmbda,
        filename=f'Column_tp={t_p}_a={a}',
        dir_name=save_path
    )

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=True,
            title="Column - After Rocking")

    # Save structure
    St.save_structure(os.path.join(save_path, 'Column_Rocking'))

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Nonlinear Dynamic Results - Column Rocking")
    print("="*60)
    print(f"Column height: {H}m")
    print(f"Number of blocks: {nb_blocks + 2}")
    print(f"Pulse period: {t_p}s")
    print(f"Pulse amplitude: {a}g")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
