# -*- coding: utf-8 -*-
"""
Dimitri Column Rocking - COMPDYN 2025
=====================================

Refactored from: Legacy/Examples/COMPDYN_2025/Dimitri_Column/Column_Rocking.py

This example demonstrates nonlinear dynamic analysis of a tapered masonry
column subjected to horizontal base excitation, exhibiting rocking behavior
at the block interfaces.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear dynamic (HHT method)

Configuration:
- Tapered column: ~6m height, 7 main blocks + 1 top cap
- Width tapers from 1.11m (base) to ~0.9m (top)
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
from Core import Structure_Block
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

    # Column geometry (tapered)
    B1 = 1.11      # Bottom width [m]
    B2_init = 0.9  # Top width [m] (approximate)
    H = 5.95       # Total height [m]
    H_small = 0.21 # Top cap height [m]
    nb_blocks = 7  # Number of main blocks

    # Derived parameters
    dH = (H - H_small) / nb_blocks
    dB = (B1 - B2_init) / H
    B2 = B1 - dB * dH  # Width after first block

    # Material properties
    rho = 2620     # Density [kg/m³]

    # Base block dimensions
    L_base = 2.5   # Base width [m]
    H_base = 0.75  # Base height [m]

    # Reference points
    N1 = np.array([0, -H_base / 2], dtype=float)  # Base center
    N2 = np.array([0, dH / 2], dtype=float)        # First column block
    N3 = np.array([0, H - H_small / 2], dtype=float)  # Top cap

    # Direction vectors
    x = np.array([0.5, 0])
    y = np.array([0, 0.5])

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

    # --- Tapered column blocks (7 main blocks) ---
    B1_current = B1
    B2_current = B2
    N2_current = N2.copy()

    for i in range(nb_blocks):
        vertices = np.array([N2_current, N2_current, N2_current, N2_current])
        vertices[0] += B1_current * x - dH * y
        vertices[1] += B2_current * x + dH * y
        vertices[2] += -B2_current * x + dH * y
        vertices[3] += -B1_current * x - dH * y

        # Update for next block
        B1_current = B2_current
        B2_current -= dB * dH
        N2_current += 2 * dH * y

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

    # Apply horizontal inertia and gravity loads to all column blocks
    for i in range(1, nb_blocks + 2):  # 7 main + 1 cap = 8 blocks
        M = St.list_blocks[i].m
        W = 9.81 * M
        St.load_node(i, [0], W)              # Horizontal inertia (for dynamic)
        St.load_node(i, [1], -W, fixed=True) # Self-weight (fixed)

    # Plot initial structure
    St.plot(show_deformed=True, deformation_scale=1, title="Dimitri Column - Initial")

    # ==========================================================================
    # Excitation Function and Damping
    # ==========================================================================
    t_p = 0.8    # Pulse period [s]
    a = 0.2      # Amplitude [g]
    lag = 0.1    # Time lag [s]

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
    print(f"\nRunning nonlinear dynamic analysis (Dimitri Column)...")
    print(f"  Method: HHT (alpha=0.3)")
    print(f"  Duration: {10 + lag}s")
    print(f"  Time step: 1e-3s")
    print(f"  Pulse: period={t_p}s, amplitude={a}g")

    Dyn = Dynamic(St)
    St = Dyn.nonlinear(
        t_end=10 + lag,
        dt=1e-3,
        Meth=Meth,
        lmbda=lmbda,
        filename=f't_p={t_p}_a={a}',
        dir_name=save_path
    )

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=True,
            title="Dimitri Column - After Rocking")

    # Save structure
    St.save_structure(os.path.join(save_path, 'Dimitri_Column'))

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Nonlinear Dynamic Results - Dimitri Column")
    print("="*60)
    print(f"Column height: {H}m")
    print(f"Number of blocks: {nb_blocks + 2} (base + {nb_blocks} main + cap)")
    print(f"Pulse period: {t_p}s")
    print(f"Pulse amplitude: {a}g")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
