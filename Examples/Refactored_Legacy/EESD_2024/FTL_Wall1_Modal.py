# -*- coding: utf-8 -*-
"""
F&TL Wall 1 - Modal Analysis - EESD 2024
=========================================

Refactored from: Legacy/Examples/EESD_2024/F&TL_Wall1.py

This example demonstrates modal analysis of a masonry wall (F&TL Wall 1)
with computation of modal contribution factors for both displacement
and base shear response.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Modal analysis with contribution factors

Configuration:
- Running bond pattern: 5 blocks per course, 6 courses
- Block dimensions: 0.4m x 0.175m
- Linear elastic contact at interfaces
- Modal contribution factors for:
  - Corner displacement
  - Base shear
- Rayleigh damping ratio computation
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
from Core import Structure_Block, Modal
from Core.Structures import WallBlock
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.ConstitutiveLaw.Contact import Contact


def main():
    # ==========================================================================
    # Parameters
    # ==========================================================================
    N1 = np.array([0, 0])

    # Block dimensions
    H_b = 0.175   # Block height [m]
    L_b = 0.4     # Block length [m]
    B = 1.0       # Out-of-plane thickness [m]

    # Contact stiffness
    kn = 1e7      # Normal stiffness [N/m]
    ks = 1e7      # Shear stiffness [N/m]

    # Material properties (normalized for modal analysis)
    RHO = 10000 / 9.81  # Density [kg/m³] (gives unit weight of ~10 kN/m³)

    # Wall pattern parameters
    Blocks_Bed = 5   # Blocks per bed course
    Blocks_Head = 6  # Number of courses

    # ==========================================================================
    # Generate Wall Pattern
    # ==========================================================================
    # Create running bond pattern
    Line1 = []  # Full blocks course
    Line2 = []  # Offset course (starts with half block)

    for i in range(Blocks_Bed):
        Line1.append(1.0)
        if i == 0:
            Line2.append(0.5)
            Line2.append(1.0)
        elif i == Blocks_Bed - 1:
            Line2.append(0.5)
        else:
            Line2.append(1.0)

    # Build pattern: alternating courses
    PATTERN = []
    for i in range(Blocks_Head):
        if i % 2 == 0:
            PATTERN.append(Line2)
        else:
            PATTERN.append(Line1)

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=RHO)

    # Add base block (fixed foundation)
    vertices = np.array([
        [Blocks_Bed * L_b, -H_b],
        [Blocks_Bed * L_b, 0],
        [0, 0],
        [0, -H_b]
    ])
    St.add_block_from_vertices(vertices, b=B, material=material)

    # Create a temporary wall structure to get the blocks
    Wall_temp = WallBlock(N1, L_b, H_b, PATTERN, RHO, b=B, material=material)

    # Add wall blocks to main structure
    for block in Wall_temp.list_blocks:
        St.list_blocks.append(block)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with linear elastic contact
    # nb_cps=2 is integer, so use contact= parameter
    St.make_cfs(
        lin_geom=True,
        nb_cps=2,
        contact=Contact(kn, ks),
        offset=0.0
    )

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix base block
    St.fix_node(0, [0, 1, 2])

    # Apply horizontal loads to all wall blocks
    Total_mass = 0
    for i in range(1, len(St.list_blocks)):
        W = St.list_blocks[i].m
        St.load_node(i, [0], W)  # Horizontal force proportional to mass
        Total_mass += W

    print(f"Total mass: {Total_mass:.2f} kg")

    # ==========================================================================
    # Modal Analysis
    # ==========================================================================
    print(f"\nRunning modal analysis...")
    n_modes = 11

    St = Modal.solve_modal(St, modes=n_modes, save=False, initial=True)

    # Print eigenvalues (natural frequencies squared)
    print(f"\nEigenvalues (ω²):")
    print(np.around(St.eig_vals, 3))

    # ==========================================================================
    # Modal Contribution Factors - Displacement
    # ==========================================================================
    print("\n" + "="*60)
    print("Modal Contribution Factors - Corner Displacement")
    print("="*60)

    nb_modal_contributions = 10

    # Get reaction forces and initial stiffness
    St.get_P_r()
    St.get_K_str0()

    sum_contr = 0

    # Reference displacement from static analysis
    U_ref = np.linalg.solve(
        St.K0[np.ix_(St.dof_free, St.dof_free)],
        St.P[St.dof_free]
    )
    print(f"Reference corner displacement: {U_ref[-3] * 1000:.4f} mm")

    for i in range(nb_modal_contributions):
        # Modal mass
        M_i = St.eig_modes[:, i].T @ St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, i]
        # Modal participation factor
        G_i = St.eig_modes[:, i].T @ St.P[St.dof_free] / M_i
        # Reference force for mode i
        P_ref_i = G_i * St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, i]
        # Displacement from mode i
        U_i = np.linalg.solve(St.K0[np.ix_(St.dof_free, St.dof_free)], P_ref_i)

        contribution = U_i[-3] / U_ref[-3]
        sum_contr += contribution
        print(f"Mode {i + 1}: {contribution * 100:.3f}% (Cumulative: {sum_contr * 100:.3f}%)")

    # ==========================================================================
    # Modal Contribution Factors - Base Shear
    # ==========================================================================
    print("\n" + "="*60)
    print("Modal Contribution Factors - Base Shear")
    print("="*60)

    # Set reference displacement to get base shear
    St.U[St.dof_free] = U_ref.copy()
    St.get_P_r()
    V_ref = St.P_r[0]
    print(f"Reference base shear: {V_ref / 1000:.4f} kN")

    sum_contr = 0

    for i in range(nb_modal_contributions):
        # Modal mass
        M_i = St.eig_modes[:, i].T @ St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, i]
        # Modal participation factor
        G_i = St.eig_modes[:, i].T @ St.P[St.dof_free] / M_i
        # Reference force for mode i
        P_ref_i = G_i * St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, i]
        # Displacement from mode i
        U_i = np.linalg.solve(St.K0[np.ix_(St.dof_free, St.dof_free)], P_ref_i)

        # Get base shear for mode i
        St.U[St.dof_free] = U_i.copy()
        St.get_P_r()
        V_i = St.P_r[0]

        contribution = V_i / V_ref
        sum_contr += contribution
        print(f"Mode {i + 1}: {contribution * 100:.3f}% (Cumulative: {sum_contr * 100:.3f}%)")

    # ==========================================================================
    # Damping Ratio Computation
    # ==========================================================================
    print("\n" + "="*60)
    print("Damping Ratios (with 1% Rayleigh Damping)")
    print("="*60)

    # Set Rayleigh damping
    St.set_damping_properties(xsi=0.01, damp_type='RAYLEIGH')
    St.get_C_str()

    # Recompute modes
    St = Modal.solve_modal(St, modes=nb_modal_contributions, save=False, initial=True)

    for i in range(nb_modal_contributions):
        # Modal mass
        M_i = St.eig_modes[:, i].T @ St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, i]
        # Modal damping
        C_i = St.eig_modes[:, i].T @ St.C[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, i]
        # Damping ratio
        ksi_i = C_i / (2 * St.eig_vals[i] * M_i)
        print(f"Mode {i + 1}: ξ = {ksi_i:.4f}")

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Modal Analysis Complete - F&TL Wall 1")
    print("="*60)
    print(f"Wall dimensions: {Blocks_Bed * L_b}m x {Blocks_Head * H_b}m")
    print(f"Number of blocks: {len(St.list_blocks)}")
    print(f"Number of modes computed: {n_modes}")
    print("="*60)

    return St


if __name__ == "__main__":
    St = main()
