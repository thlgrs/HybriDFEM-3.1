# -*- coding: utf-8 -*-
"""
Modal Analysis - Large Masonry Wall
====================================

Refactored from: Legacy/Examples/Modal Analysis/EigVals_LargeWall.py

This example computes natural frequencies and mode shapes for a large
masonry wall with running bond pattern.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Modal (eigenvalue analysis)

Configuration:
- Wall: 6.24m x 5.2m (L x H)
- Block pattern: Running bond (alternating rows)
- Material: E=0.2 GPa, G=0.01 GPa, rho=1500 kg/m3
- Contact: Surface with kn = E/t, ks = G/t
"""

import sys
from pathlib import Path

import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Modal
from Core.Structures import WallBlock
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.DFEM.Surface import Surface


def main():
    # ==========================================================================
    # Parameters
    # ==========================================================================
    N1 = np.array([0, 0])   # Wall origin

    L_wall = 6.24   # Wall length [m]
    H_wall = 5.2    # Wall height [m]

    Blocks_Bed = 6 * 4    # Blocks in bed joint direction (horizontal)
    Blocks_Head = 20 * 4  # Blocks in head joint direction (vertical)

    H = H_wall / Blocks_Head  # Block height [m]
    L = L_wall / Blocks_Bed   # Block length [m]
    B = 0.12                  # Wall thickness [m]

    CPS = 6  # Contact pairs per interface

    # Material properties
    E = 0.2e9   # Young's modulus [Pa]
    G = 0.01e9  # Shear modulus [Pa]
    t = 0.01 / 2  # Joint thickness [m]

    kn = E / t  # Normal stiffness
    ks = G / t  # Shear stiffness

    RHO = 1500.0  # Density [kg/m3]

    # ==========================================================================
    # Create Wall Pattern
    # ==========================================================================
    # Running bond: alternating full rows and rows with half blocks at ends
    Line1 = []  # Full blocks
    Line2 = []  # Half block at start and end

    for i in range(Blocks_Bed):
        Line1.append(1.0)
        if i == 0:
            Line2.append(0.5)
            Line2.append(1.0)
        elif i == Blocks_Bed - 1:
            Line2.append(0.5)
        else:
            Line2.append(1.0)

    PATTERN = []
    for i in range(Blocks_Head):
        if i % 2 == 0:
            PATTERN.append(Line1)
        else:
            PATTERN.append(Line2)

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    material = Material(E=E, nu=0.2, rho=RHO)

    St = WallBlock(N1, L, H, PATTERN, rho=RHO, b=B, material=material)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with surface contact law
    St.make_cfs(lin_geom=True, nb_cps=CPS, surface=Surface(kn, ks))

    # ==========================================================================
    # Modal Analysis
    # ==========================================================================
    # Note: Wall is free-standing (no boundary conditions)
    St = Modal.solve_modal(St, modes=11, save=False)

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Modal Analysis Results - Large Masonry Wall")
    print("="*60)
    print(f"Wall dimensions: {L_wall}m x {H_wall}m x {B}m")
    print(f"Block dimensions: {L:.4f}m x {H:.4f}m")
    print(f"Total blocks: {len(St.list_blocks)}")
    print("-"*60)
    print(f"Natural frequencies (rad/s):")
    print(np.around(St.eig_vals[:11], 3))
    print("-"*60)
    print(f"Natural frequencies (Hz):")
    print(np.around(St.eig_vals[:11] / (2 * np.pi), 3))
    print("="*60)

    import matplotlib.pyplot as plt
    plt.show()

    return St


if __name__ == "__main__":
    St = main()
