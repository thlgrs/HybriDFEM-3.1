# -*- coding: utf-8 -*-
"""
Three Blocks Sliding - Displacement Control
===========================================

Refactored from: Legacy/Examples/Structures_2024/Rocking_Blocks/ThreeBlocks_Sliding.py

This example demonstrates displacement-controlled analysis of three stacked
blocks with the middle block pushed horizontally while the top and bottom
blocks are fixed.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear static (displacement control)

Configuration:
- Three stacked blocks: 0.5m x 0.5m each
- Contact: Coulomb friction (kn=ks=1e7 N/m, mu=1.0)
- Loading: Horizontal displacement of middle block
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
    H = 0.5    # Block height [m]
    L = 0.5    # Block width [m]
    B = 1.0    # Thickness [m]

    # Contact stiffness
    kn = 1e7   # Normal stiffness [N/m]
    ks = 1e7   # Shear stiffness [N/m]
    mu = 1.0   # Friction coefficient

    RHO = 1000.0  # Density [kg/m3]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=RHO)

    # Bottom block (fixed)
    vertices_1 = np.array([
        [L, 0],
        [L, H],
        [0, H],
        [0, 0]
    ])
    St.add_block_from_vertices(vertices_1, b=B, material=material)

    # Middle block (sliding)
    vertices_2 = np.array([
        [L, H],
        [L, 2 * H],
        [0, 2 * H],
        [0, H]
    ])
    St.add_block_from_vertices(vertices_2, b=B, material=material)

    # Top block (fixed)
    vertices_3 = np.array([
        [L, 2 * H],
        [L, 3 * H],
        [0, 3 * H],
        [0, 2 * H]
    ])
    St.add_block_from_vertices(vertices_3, b=B, material=material)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with Coulomb friction
    St.make_cfs(lin_geom=False, nb_cps=2, contact=Coulomb(kn, ks, mu), offset=0.0)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix bottom block
    St.fix_node(0, [0, 1, 2])

    # Fix top block
    St.fix_node(2, [0, 1, 2])

    # Apply loading to middle block
    W = St.list_blocks[1].m * 9.81  # Weight
    St.load_node(1, [0], W)               # Horizontal force
    St.load_node(1, [1], -W, fixed=True)  # Self-weight (fixed)
    W_c = W  # Reference weight for normalization

    # ==========================================================================
    # Plot Initial Structure
    # ==========================================================================
    St.plot(show_contact_faces=True, title="Three Blocks - Initial")

    # ==========================================================================
    # Displacement Control Analysis
    # ==========================================================================
    Node = 1  # Control node (middle block)

    # Displacement steps (0 to 10mm)
    LIST = np.linspace(0, 1e-2, 100).tolist()

    print(f"\nRunning displacement control analysis...")
    print(f"  Control node: {Node} (middle block)")
    print(f"  Max displacement: 10 mm")
    print(f"  Number of steps: {len(LIST)}")

    St = StaticNonLinear.solve_dispcontrol(
        St, LIST, 0, Node, 0,
        tol=1e-4,
        filename='ThreeBlocks_DispControl',
        dir_name=save_path,
        max_iter=100
    )

    # Save structure
    St.save_structure(os.path.join(save_path, 'ThreeBlocks'))

    # ==========================================================================
    # Plot Deformed Structure
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=False,
            title="Three Blocks - Deformed")

    # ==========================================================================
    # Plot Force-Displacement Curve
    # ==========================================================================
    result_file = os.path.join(save_path, 'ThreeBlocks_DispControl.h5')
    if os.path.exists(result_file):
        with h5py.File(result_file, 'r') as hf:
            U = hf['U_conv'][3 * Node] * 1000   # Displacement [mm]
            P = hf['P_r_conv'][3 * Node] / W_c  # Normalized force

        plt.figure(figsize=(8, 6))
        plt.plot(U, P, 'k-o', linewidth=0.75, markersize=3)
        plt.xlabel('Horizontal Displacement [mm]')
        plt.ylabel('Force / Weight')
        plt.title('Three Blocks - Force-Displacement')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'force_displacement.png'), dpi=150)

        print(f"\nMax load multiplier: {np.max(P):.3f}")

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Displacement Control Results - Three Blocks Sliding")
    print("="*60)
    print(f"Block dimensions: {L}m x {H}m")
    print(f"Friction coefficient: {mu}")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
