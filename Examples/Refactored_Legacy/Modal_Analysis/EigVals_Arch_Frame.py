# -*- coding: utf-8 -*-
"""
Modal Analysis - Arch-Frame Structure
=====================================

Refactored from: Legacy/Examples/Modal Analysis/EigVals_Arch-Frame.py

This example computes natural frequencies and mode shapes for an arch-frame
structure consisting of two vertical columns and a semicircular arch.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Modal (eigenvalue analysis)

Configuration:
- Two vertical columns: 4m height
- Semicircular arch: 4m radius, spanning 8m
- Cross-section: 0.5m x 1.0m (H x B)
- Material: E=30 GPa, nu=0.3, rho=2500 kg/m3
"""

import sys
from pathlib import Path

import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Structure_Block, Modal
from Core.Structures import BeamBlock, ArchBlock
from Core.Objects.ConstitutiveLaw.Material import Material


def main():
    # ==========================================================================
    # Parameters
    # ==========================================================================
    N1 = np.array([0, 0], dtype=float)   # Left column base
    N2 = np.array([0, 4], dtype=float)   # Left column top
    N3 = np.array([8, 4], dtype=float)   # Right column top
    N4 = np.array([8, 0], dtype=float)   # Right column base
    C = np.array([4, 4], dtype=float)    # Arch center

    H = 0.5   # Cross-section height [m]
    B = 1.0   # Cross-section width [m]

    BLOCKS = 30   # Blocks per column
    CPS = 50      # Contact pairs per interface

    E = 30e9      # Young's modulus [Pa]
    NU = 0.3      # Poisson's ratio
    RHO = 2500    # Density [kg/m3]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    material = Material(E, NU, rho=RHO)

    # Create combined structure manually
    St = Structure_Block()

    # Left column (N1 to N2)
    left_col = BeamBlock(N1, N2, BLOCKS, H, rho=RHO, b=B, material=material, end_2=False)
    for block in left_col.list_blocks:
        St.list_blocks.append(block)

    # Arch (semicircle from 0 to pi radians, centered at C)
    arch = ArchBlock(C, 0, np.pi, 4, 3 * BLOCKS, H, rho=RHO, b=B, material=material)
    for block in arch.list_blocks:
        St.list_blocks.append(block)

    # Right column (N3 to N4)
    right_col = BeamBlock(N3, N4, BLOCKS, H, rho=RHO, b=B, material=material, end_1=False)
    for block in right_col.list_blocks:
        St.list_blocks.append(block)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces (linear geometry)
    St.make_cfs(lin_geom=True, nb_cps=CPS)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix base of both columns
    node_N1 = St.get_node_id(N1)
    node_N4 = St.get_node_id(N4)

    if node_N1 is not None:
        St.fix_node(node_N1, [0, 1, 2])
    if node_N4 is not None:
        St.fix_node(node_N4, [0, 1, 2])

    # ==========================================================================
    # Plot Structure
    # ==========================================================================
    St.plot(show_contact_faces=False, title="Arch-Frame Structure")

    # ==========================================================================
    # Modal Analysis
    # ==========================================================================
    St = Modal.solve_modal(St, modes=5, save=False)

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Modal Analysis Results - Arch-Frame Structure")
    print("="*60)
    print(f"First 5 natural frequencies (rad/s): {St.eig_vals[:5]}")
    print(f"First 5 natural frequencies (Hz): {St.eig_vals[:5] / (2 * np.pi)}")
    print("="*60)

    import matplotlib.pyplot as plt
    plt.show()

    return St


if __name__ == "__main__":
    St = main()
