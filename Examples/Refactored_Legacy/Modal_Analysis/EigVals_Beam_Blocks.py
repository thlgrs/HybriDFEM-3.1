# -*- coding: utf-8 -*-
"""
Modal Analysis - Shear Beam Block Convergence Study
====================================================

Refactored from: Legacy/Examples/Modal Analysis/Linear_Elastic_ShearBeam/EigVals_Beam_Blocks.py

This example performs a convergence study comparing the natural frequencies
of a Timoshenko-Ehrenfest beam modeled with discrete blocks against
analytical solutions.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Modal analysis (eigenvalue problem)

Configuration:
- Simply supported beam: 3m length
- Cross-section: 0.5m x 0.2m
- Material: E = 30 GPa, rho = 7000 kg/m³
- Convergence study: 10 to 200 blocks

Reference:
Timoshenko-Ehrenfest beam theory analytical frequencies
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
from Core.Structures import BeamBlock
from Core.Objects.ConstitutiveLaw.Material import Material


def compute_analytical_frequencies(E, NU, RHO, H, B, L, n_modes=10):
    """
    Compute Timoshenko-Ehrenfest beam analytical natural frequencies.

    Returns array of first n_modes natural frequencies (rad/s).
    """
    G = E / 2 / (1 + NU)
    k = 5 * (1 + NU) / (6 + 5 * NU)  # Shear correction factor
    A = B * H
    I = B * H ** 3 / 12

    k2 = np.arange(1, n_modes + 1, 1)

    # Timoshenko beam frequency equation coefficients
    b_star = -G * k / RHO * (A / I + (k2 * np.pi / L) ** 2 * (1 + E / (G * k)))
    c_star = E * G * k / (RHO ** 2) * (k2 * np.pi / L) ** 4

    # Two branches of solutions
    w2 = np.sqrt(1 / 2 * (-b_star - np.sqrt(b_star ** 2 - 4 * c_star)))
    w1 = np.sqrt(1 / 2 * (-b_star + np.sqrt(b_star ** 2 - 4 * c_star)))

    # Torsional frequency
    w_t = np.sqrt(G * k * A / (RHO * I))

    # Sort all frequencies
    w_T = np.sort(np.concatenate((w2, w1, np.array([w_t]))))

    return w_T[:n_modes]


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
    N1 = np.array([0, 0], dtype=float)
    N2 = np.array([3, 0], dtype=float)

    H = 0.5    # Cross-section height [m]
    B = 0.2    # Cross-section width [m]
    L = 3.0    # Length [m]

    CPS = 100  # Contact points per interface

    E = 30e9   # Young's modulus [Pa]
    NU = 0.0   # Poisson's ratio
    RHO = 7000 # Density [kg/m³]

    # List of block counts for convergence study
    list_Blocks = [10, 25, 50, 100, 200]
    markers = ['o', 's', '^', 'D', '*']
    n_modes = 10

    # ==========================================================================
    # Analytical Solution
    # ==========================================================================
    w_T = compute_analytical_frequencies(E, NU, RHO, H, B, L, n_modes)

    print("Timoshenko-Ehrenfest analytical frequencies:")
    for i in range(n_modes):
        print(f"  Mode {i+1}: {w_T[i]:.3f} rad/s")

    # ==========================================================================
    # Convergence Study
    # ==========================================================================
    print("\nRunning convergence study...")

    # Setup plot
    plt.figure(figsize=(8, 6), dpi=150)
    modes = np.arange(1, n_modes + 1)
    plt.plot(modes, np.zeros(n_modes), color='black', linewidth=0.5, linestyle='dashed')

    for i, BLOCKS in enumerate(list_Blocks):
        print(f"  Testing {BLOCKS} blocks...")

        # Create material with shear correction factor
        material = Material(E, NU, rho=RHO, corr_fact=6/5)

        # Create beam structure
        Beam_temp = BeamBlock(N1, N2, BLOCKS, H, RHO, b=B, material=material)

        St = Structure_Block()
        for block in Beam_temp.list_blocks:
            St.list_blocks.append(block)

        St.make_nodes()
        St.make_cfs(lin_geom=True, nb_cps=CPS)

        # Boundary conditions: Simply supported (pin-roller)
        # Fix vertical displacement at both ends
        N1_idx = St.get_node_at(N1)
        N2_idx = St.get_node_at(N2)

        St.fix_node(N1_idx, [1])  # Pin: vertical fixed
        St.fix_node(N2_idx, [1])  # Roller: vertical fixed

        # Fix horizontal displacement for all blocks (pure bending)
        for j in range(BLOCKS):
            St.fix_node(j, [0])

        # Modal analysis
        St = Modal.solve(St, n_modes)

        # Compute error
        error = (St.eig_vals[:n_modes] - w_T) / w_T * 100

        plt.plot(modes, error, label=f'{BLOCKS} Blocks', color='black',
                marker=markers[i], linewidth=1, markersize=4)

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    plt.grid(True, color='gainsboro', linewidth=0.5)
    plt.xlabel('Mode number')
    plt.ylabel('Relative error [%]')
    plt.xlim([1, n_modes])
    plt.ylim([-5, 0.5])
    plt.xticks(np.arange(1, n_modes + 1, 1))
    plt.legend()
    plt.title('Shear Beam - Block Discretization Convergence')
    plt.savefig(os.path.join(save_path, 'Convergence_Blocks.png'), dpi=150)

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Modal Analysis - Shear Beam Block Convergence Study")
    print("="*60)
    print(f"Beam: {L}m x {H}m x {B}m")
    print(f"Material: E = {E/1e9} GPa, rho = {RHO} kg/m³")
    print(f"Block counts tested: {list_Blocks}")
    print(f"Number of modes: {n_modes}")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()


if __name__ == "__main__":
    main()
