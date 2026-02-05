# -*- coding: utf-8 -*-
"""
Modal Analysis - Four Brick Masonry Wallet
==========================================

Refactored from: Legacy/Examples/Modal Analysis/Masonry_Wallet/EigVals_FourBricks.py

This example demonstrates modal analysis of a simple 4-brick masonry wallet
with running bond pattern, comparing numerical results against reference
discrete element and finite element solutions.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Modal analysis (eigenvalue problem)

Configuration:
- 4 bricks in running bond (2 rows, 3 columns)
- Block dimensions: 0.25m x 0.055m
- Joint thickness: 10mm
- Material: E = 1 GPa, nu = 0.2, rho = 1800 kg/m³
- Linear elastic contact at interfaces
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
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.ConstitutiveLaw.Contact import Contact


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
    N1 = np.array([0, 0])

    # Brick dimensions
    H = 0.055   # Brick height [m]
    L = 0.25    # Brick length [m]
    B = 0.12    # Out-of-plane thickness [m]
    t = 0.01    # Joint thickness [m]

    CPS = 10    # Contact points per interface

    # Material properties
    E = 1e9     # Young's modulus [Pa]
    NU = 0.2    # Poisson's ratio
    G = E / (2 * (1 + NU))

    # Contact stiffness (derived from material)
    kn = E / (t / 2)    # Normal stiffness
    ks = G / (t / 2)    # Shear stiffness

    RHO = 1800  # Density [kg/m³]

    # ==========================================================================
    # Create Structure - 4 Brick Running Bond
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=E, nu=NU, rho=RHO)

    # Bottom row: 1 full brick
    vertices1 = np.array([[L, 0.], [L, H], [0., H], [0.0, 0.0]])
    St.add_block_from_vertices(vertices1, b=B, material=material)

    # Middle row: 2 half bricks with reference points at corners
    # Left half brick
    vertices2 = np.array([[L/2, H], [L/2, 2*H], [0, 2*H], [0, H]])
    St.add_block_from_vertices(vertices2, b=B, material=Material(E=E, nu=NU, rho=2*RHO),
                               ref_point=np.array([0, 3*H/2]))

    # Right half brick
    vertices3 = np.array([[L, H], [L, 2*H], [L/2, 2*H], [L/2, H]])
    St.add_block_from_vertices(vertices3, b=B, material=Material(E=E, nu=NU, rho=2*RHO),
                               ref_point=np.array([L, 3*H/2]))

    # Top row: 1 full brick
    vertices4 = np.array([[L, 2*H], [L, 3*H], [0.0, 3*H], [0.0, 2*H]])
    St.add_block_from_vertices(vertices4, b=B, material=material)

    # Initialize structure
    St.make_nodes()

    # Create contact faces with linear elastic contact
    # Using nb_cps as integer, so use contact= parameter
    St.make_cfs(lin_geom=True, nb_cps=CPS, contact=Contact(kn, ks))

    # ==========================================================================
    # Modal Analysis
    # ==========================================================================
    print("Running modal analysis...")
    n_modes = 12
    St = Modal.solve(St, n_modes)

    # Plot mode shapes
    Modal.plot_modes(St, n_modes, scale=0.05, title="Four Bricks - Mode Shapes")

    # ==========================================================================
    # Compare with Reference Solutions
    # ==========================================================================
    # Expected frequencies from discrete elements (Hz)
    Expected_DE = np.array([2496, 3430, 4184, 4325, 4732, 5163, 5732, 8755, 9172]) * np.pi * 2
    # Expected frequencies from finite elements (Hz)
    Expected_FE = np.array([2514, 3345, 4043, 4118, 4697, 4976, 5401, 8265, 8549]) * np.pi * 2

    print("\nNatural frequencies (rad/s):")
    freqs = St.eig_vals / (2 * np.pi)
    print(f"  HybriDFEM: {np.around(freqs, 0)}")

    if len(St.eig_vals) > 3:
        # Skip first 3 rigid body modes
        Error_DE = (St.eig_vals[3:3+len(Expected_DE)] - Expected_DE) / Expected_DE * 100
        Error_FE = (St.eig_vals[3:3+len(Expected_FE)] - Expected_FE) / Expected_FE * 100

        print("\nRelative error vs Discrete Elements [%]:")
        print(f"  {np.around(Error_DE, 2)}")
        print("\nRelative error vs Finite Elements [%]:")
        print(f"  {np.around(Error_FE, 2)}")

    # ==========================================================================
    # Plot Comparison
    # ==========================================================================
    plt.figure(figsize=(10, 6), dpi=150)
    modes = np.arange(4, 4 + len(Expected_DE))

    plt.plot(modes, St.eig_vals[3:3+len(Expected_DE)], 'ko-', label='HybriDFEM', markersize=6)
    plt.plot(modes, Expected_DE, 'b^--', label='Discrete Elements (ref)', markersize=6)
    plt.plot(modes, Expected_FE, 'rs--', label='Finite Elements (ref)', markersize=6)

    plt.xlabel('Mode number')
    plt.ylabel('Natural frequency (rad/s)')
    plt.title('Four Brick Wallet - Frequency Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'FourBricks_Comparison.png'), dpi=150)

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Modal Analysis Results - Four Brick Wallet")
    print("="*60)
    print(f"Wall dimensions: {L}m x {3*H}m")
    print(f"Number of bricks: 4")
    print(f"Brick dimensions: {L}m x {H}m x {B}m")
    print(f"Joint thickness: {t*1000}mm")
    print(f"Contact stiffness: kn = {kn:.2e}, ks = {ks:.2e}")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
