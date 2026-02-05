# -*- coding: utf-8 -*-
"""
Modal Contribution Factors Test - EESD 2024
============================================

Refactored from: Legacy/Examples/EESD_2024/Contribution_Factors_Test/F&TL_Wall1_Modal.py

This example demonstrates modal contribution factor computation using a
simple stacked block model. It verifies the modal superposition principle
by computing contribution factors for displacement and base shear.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Modal analysis with contribution factors

Configuration:
- 6 stacked unit blocks (1m x 1m x 1m)
- Linear elastic contact at interfaces
- Applied lateral forces at blocks 4 and 5
- Modal contribution factors verification
"""

import os
import sys
from pathlib import Path

import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Structure_Block, Modal
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.ConstitutiveLaw.Contact import Contact


def main():
    # ==========================================================================
    # Parameters
    # ==========================================================================
    N1 = np.array([0, 0], dtype=float)

    # Block dimensions (unit cube)
    H_b = 1.0     # Block height [m]
    L_b = 1.0     # Block length [m]
    B = 1.0       # Out-of-plane thickness [m]

    # Contact stiffness (unit stiffness)
    kn = 1.0      # Normal stiffness [N/m]
    ks = 1.0      # Shear stiffness [N/m]

    # Material properties (unit density)
    RHO = 1.0     # Density [kg/m³]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=RHO)

    # Create 6 stacked unit blocks
    for i in range(6):
        vertices = np.array([N1, N1, N1, N1])
        vertices += np.array([
            [0, i * H_b],
            [0, i * H_b],
            [0, i * H_b],
            [0, i * H_b]
        ], dtype=float)
        vertices += np.array([
            [L_b / 2, -H_b / 2],
            [L_b / 2, H_b / 2],
            [-L_b / 2, H_b / 2],
            [-L_b / 2, -H_b / 2]
        ])
        St.add_block_from_vertices(vertices, b=B, material=material)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with linear elastic contact
    St.make_cfs(
        lin_geom=True,
        nb_cps=2,
        contact=Contact(kn, ks),
        offset=0.0
    )

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix base block completely
    St.fix_node(0, [0, 1, 2])

    # Fix vertical and rotational DOFs for remaining blocks
    # (only horizontal translation allowed)
    for i in range(1, len(St.list_blocks)):
        St.fix_node(i, [1, 2])

    # Apply horizontal loads
    St.load_node(5, [0], 2)   # 2N at top block
    St.load_node(4, [0], -1)  # -1N at block below

    print(f"Applied loads: 2N at block 5, -1N at block 4")

    # ==========================================================================
    # Modal Analysis
    # ==========================================================================
    print(f"\nRunning modal analysis...")

    St = Modal.solve_modal(St, modes=5, save=False, initial=True)

    # Print eigenvalues
    print(f"\nEigenvalues (ω²):")
    print(np.around(St.eig_vals, 3))

    # ==========================================================================
    # Modal Contribution Factors - Displacement
    # ==========================================================================
    print("\n" + "="*60)
    print("Modal Contribution Factors - Top Block Displacement")
    print("="*60)

    nb_modal_contributions = 5

    # Get reaction forces and initial stiffness
    St.get_P_r()
    St.get_K_str0()

    sum_contr = 0

    # Reference displacement from static analysis
    U_ref = np.linalg.solve(
        St.K0[np.ix_(St.dof_free, St.dof_free)],
        St.P[St.dof_free]
    )
    print(f"Reference top displacement: {U_ref[-1] * 1000:.4f} mm")

    for i in range(nb_modal_contributions):
        # Modal mass
        M_i = St.eig_modes[:, i].T @ St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, i]
        # Modal participation factor
        G_i = St.eig_modes[:, i].T @ St.P[St.dof_free] / M_i
        # Reference force for mode i
        P_ref_i = G_i * St.M[np.ix_(St.dof_free, St.dof_free)] @ St.eig_modes[:, i]
        # Displacement from mode i
        U_i = np.linalg.solve(St.K0[np.ix_(St.dof_free, St.dof_free)], P_ref_i)

        contribution = U_i[-1] / U_ref[-1]
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
    print("Modal Contribution Factors Test Complete")
    print("="*60)
    print(f"Number of blocks: {len(St.list_blocks)}")
    print(f"Number of modes: {nb_modal_contributions}")
    print("Modal superposition should sum to 100%")
    print("="*60)

    return St


if __name__ == "__main__":
    St = main()
