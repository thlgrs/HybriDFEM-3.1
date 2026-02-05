# -*- coding: utf-8 -*-
"""
Two Blocks Coulomb Sliding - IABSE 2024
=======================================

Refactored from: Legacy/Examples/IABSE_2024/Masonry Wall/2Blocks_Coulomb_Lin.py

Simple two-block sliding test with cyclic displacement control to
demonstrate Coulomb friction behavior.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear static (cyclic displacement control)

Configuration:
- 2 stacked blocks (0.2m x 0.2m)
- Coulomb friction contact (kn=ks=1e6 N/m, mu=0.65)
- Cyclic horizontal displacement at top
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import h5py

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Structure_Block, StaticNonLinear
from Core.Structures import WallBlock
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.ConstitutiveLaw.Contact import Coulomb


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

    H_b = 0.2     # Block height [m]
    L_b = 0.2     # Block length [m]
    B = 1.0       # Out-of-plane thickness [m]

    kn = 1e6      # Normal stiffness [N/m]
    ks = 1e6      # Shear stiffness [N/m]
    mu = 0.65     # Friction coefficient

    RHO = 2000    # Density [kg/m³]

    # Simple 2-block pattern
    PATTERN = [[1], [1]]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=RHO)

    # Create wall (2 stacked blocks)
    Wall_temp = WallBlock(N1, L_b, H_b, PATTERN, RHO, b=B, material=material)
    for block in Wall_temp.list_blocks:
        St.list_blocks.append(block)

    St.make_nodes()
    St.make_cfs(lin_geom=True, nb_cps=2, contact=Coulomb(kn, ks, mu), offset=0.02)

    # ==========================================================================
    # Boundary Conditions and Loading
    # ==========================================================================
    # Fix bottom block
    for i in range(len(PATTERN[0])):
        St.fix_node(i, [0, 1, 2])

    # Apply gravity and horizontal load to top block
    for i in range(len(PATTERN[0]), len(St.list_blocks)):
        W = St.list_blocks[i].m * 9.81
        St.load_node(i, [1], -W, fixed=True)
        St.load_node(i, [0], W)

    St.plot(show_contact_faces=True, title="Two Blocks - Initial")

    # ==========================================================================
    # Cyclic Displacement Control
    # ==========================================================================
    Node = len(St.list_blocks) - 1
    MAX_D = L_b / 8
    n_incr = 2
    delta_D = MAX_D / n_incr
    STEPS = 200

    # Build cyclic displacement history
    LIST_D = np.array([])
    for i in range(n_incr):
        LIST_D = np.append(LIST_D, np.linspace(-delta_D * i, delta_D * (i + 1), STEPS * (i + 1)))
        LIST_D = np.append(LIST_D, np.linspace(delta_D * (i + 1), -delta_D * (i + 1), 2 * STEPS * (i + 1)))
    LIST_D = LIST_D.tolist()

    print(f"\nRunning cyclic sliding test...")
    print(f"  Max displacement: ±{MAX_D * 1000:.2f} mm")
    print(f"  Number of cycles: {n_incr}")

    St = StaticNonLinear.solve_dispcontrol(
        St, LIST_D, 0, Node, 0,
        tol=1,
        filename='Sliding',
        dir_name=save_path
    )

    St.save_structure(os.path.join(save_path, 'TwoBlocks_Sliding'))

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    result_file = os.path.join(save_path, 'Sliding.h5')
    if os.path.exists(result_file):
        with h5py.File(result_file, 'r') as hf:
            P = hf['P_r_conv'][3 * Node] / (0.2 * 0.2 * 9.81 * RHO)
            U = hf['U_conv'][3 * Node] * 1000

        plt.figure(figsize=(8, 6))
        plt.plot(U, P, 'k-', linewidth=0.75, label='F-Δ')
        plt.xlabel('Top horizontal displacement [mm]')
        plt.ylabel('Load multiplier α')
        plt.title('Two Blocks - Cyclic Sliding (Coulomb Friction)')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, 'sliding_hysteresis.png'), dpi=150)

    print("\n" + "="*60)
    print("Two Blocks Coulomb Sliding Test Complete")
    print("="*60)

    plt.show()
    return St


if __name__ == "__main__":
    St = main()
