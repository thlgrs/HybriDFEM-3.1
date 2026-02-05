# -*- coding: utf-8 -*-
"""
Ferris & Tin-Loi Wall 1 - Pushover Analysis
============================================

Refactored from: Legacy/Examples/F&TL Walls (WCEE 2024)/Lin_Geom/Ferris&Tin-Loi_Wall1/Pushover_F&TL-1.py

This example demonstrates pushover analysis of a masonry wall using the
benchmark configuration from Ferris & Tin-Loi (2001).

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear static (displacement control)

Configuration:
- Running bond pattern: 5 blocks per course, 6 courses
- Block dimensions: 0.4m x 0.175m
- Coulomb friction contact (kn=ks=1e10 N/m, mu=0.65)
- Horizontal displacement control at top
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

    # Block dimensions
    H_b = 0.175   # Block height [m]
    L_b = 0.4     # Block length [m]
    B = 1.0       # Out-of-plane thickness [m]

    # Contact parameters
    kn = 1e10     # Normal stiffness [N/m]
    ks = 1e10     # Shear stiffness [N/m]
    mu = 0.65     # Friction coefficient
    psi = 0.0     # Dilatancy angle [rad]
    r_b = 0.02    # Contact offset [m]

    # Material properties
    RHO = 1000.0  # Density [kg/m³]

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

    # Create contact faces with Coulomb friction
    St.make_cfs(
        lin_geom=True,
        nb_cps=2,
        contact=Coulomb(kn, ks, mu, psi=psi),
        offset=r_b
    )

    # ==========================================================================
    # Boundary Conditions and Loading
    # ==========================================================================
    # Fix base block
    St.fix_node(0, [0, 1, 2])

    # Control node (top of wall)
    Node = len(St.list_blocks) - 1

    # Apply horizontal and gravity loads to all wall blocks
    W_c = 0  # Reference weight for normalization
    for i in range(1, len(St.list_blocks)):
        W = St.list_blocks[i].m * 10  # Weight with g=10 m/s²
        St.load_node(i, [0], W)                # Horizontal force
        St.load_node(i, [1], -W, fixed=True)   # Self-weight (fixed)
        W_c = W  # Store last weight for normalization

    # ==========================================================================
    # Plot Initial Structure
    # ==========================================================================
    St.plot(show_contact_faces=True, title="F&TL Wall 1 - Initial")

    # ==========================================================================
    # Displacement Control Analysis
    # ==========================================================================
    # Displacement steps (0 to 0.1mm)
    LIST = np.linspace(0, 1e-4, 1000).tolist()

    print(f"\nRunning pushover analysis (F&TL Wall 1)...")
    print(f"  Control node: {Node}")
    print(f"  Max displacement: 0.1 mm")
    print(f"  Number of steps: {len(LIST)}")
    print(f"  Friction coefficient: {mu}")
    print(f"  Dilatancy angle: {psi}")

    St = StaticNonLinear.solve_dispcontrol(
        St, LIST, 0, Node, 0,
        tol=1e-3,
        filename=f'Wall1_rb={r_b}_psi={psi}',
        dir_name=save_path,
        max_iter=1000
    )

    # Save structure
    St.save_structure(os.path.join(save_path, 'FTL_Wall1'))

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1000, show_contact_faces=False,
            title="F&TL Wall 1 - Deformed")

    # Plot force-displacement curve
    result_file = os.path.join(save_path, f'Wall1_rb={r_b}_psi={psi}.h5')
    if os.path.exists(result_file):
        with h5py.File(result_file, 'r') as hf:
            U = hf['U_conv'][-3] * 1000    # Horizontal displacement [mm]
            P = hf['P_r_conv'][-3] / W_c   # Normalized load

        plt.figure(figsize=(8, 6), dpi=150)
        plt.plot(U, P, 'k-o', linewidth=0.5, markersize=2, label='HybriDFEM')
        plt.xlabel('Control displacement [mm]')
        plt.ylabel('Load multiplier')
        plt.title('F&TL Wall 1 - Pushover Curve')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, 'pushover_curve.png'), dpi=150)

        print(f"\nMax load multiplier: {np.max(P):.4f}")

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Pushover Analysis Results - F&TL Wall 1")
    print("="*60)
    print(f"Wall dimensions: {Blocks_Bed * L_b}m x {Blocks_Head * H_b}m")
    print(f"Number of blocks: {len(St.list_blocks)}")
    print(f"Friction coefficient: {mu}")
    print(f"Dilatancy angle: {psi}")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
