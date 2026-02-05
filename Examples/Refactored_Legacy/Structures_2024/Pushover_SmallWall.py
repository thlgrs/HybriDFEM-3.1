# -*- coding: utf-8 -*-
"""
Pushover Analysis - Small Masonry Wall
======================================

Refactored from: Legacy/Examples/Structures_2024/SmallWall_Pushover/Pushover_SmallWall.py

This example demonstrates displacement-controlled pushover analysis of a
small masonry wall with Coulomb friction contact.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear static (displacement control)

Configuration:
- Wall: 3x3 blocks in running bond pattern
- Block size: 0.4m x 0.2m (L x H)
- Base: Fixed rigid block
- Contact: Coulomb friction (kn=ks=1e8 N/m, mu=0.65, psi=0.2)
- Loading: Self-weight + horizontal displacement at top
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
    N1 = np.array([0, 0])  # Wall origin

    H_b = 0.2   # Block height [m]
    L_b = 0.4   # Block length [m]
    B = 1.0     # Wall thickness [m]

    # Contact stiffness
    kn = 1e8    # Normal stiffness [N/m]
    ks = 1e8    # Shear stiffness [N/m]
    mu = 0.65   # Friction coefficient
    psi = 0.2   # Dilatancy angle

    RHO = 1000.0  # Density [kg/m3]

    Blocks_Bed = 3   # Blocks per row
    Blocks_Head = 3  # Number of rows

    # ==========================================================================
    # Create Wall Pattern (Running Bond)
    # ==========================================================================
    Line1 = []  # Full blocks
    Line2 = []  # Half blocks at ends

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
            PATTERN.append(Line2)  # Offset rows
        else:
            PATTERN.append(Line1)  # Full rows

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=RHO)

    # Create base block (fixed)
    vertices_base = np.array([
        [Blocks_Bed * L_b, -H_b],
        [Blocks_Bed * L_b, 0],
        [0, 0],
        [0, -H_b]
    ])
    St.add_block_from_vertices(vertices_base, b=B, material=material)

    # Create wall blocks
    wall = WallBlock(N1, L_b, H_b, PATTERN, rho=RHO, b=B, material=material)
    for block in wall.list_blocks:
        St.list_blocks.append(block)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with Coulomb friction
    St.make_cfs(lin_geom=False, nb_cps=2, contact=Coulomb(kn, ks, mu, psi=psi), offset=0.02)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix base block
    St.fix_node(0, [0, 1, 2])

    # Apply self-weight as fixed load and horizontal force for pushover
    for i in range(1, len(St.list_blocks)):
        W = St.list_blocks[i].m * 10  # Weight
        St.load_node(i, [0], W)        # Horizontal load
        St.load_node(i, [1], -W, fixed=True)  # Self-weight (fixed)

    W_c = St.list_blocks[-1].m * 10  # Reference weight for normalization

    # ==========================================================================
    # Plot Initial Structure
    # ==========================================================================
    St.plot(show_contact_faces=True, title="Small Wall - Initial Configuration")

    # ==========================================================================
    # Displacement Control Pushover Analysis
    # ==========================================================================
    # Control node (top of wall)
    Node = len(St.list_blocks) - 1

    # Displacement steps (0 to 327mm)
    LIST = np.linspace(0, 3.27e-1, 9000).tolist()

    print(f"\nRunning displacement-controlled pushover analysis...")
    print(f"  Control node: {Node}")
    print(f"  Control DOF: 0 (horizontal)")
    print(f"  Max displacement: 327 mm")
    print(f"  Number of steps: {len(LIST)}")

    St = StaticNonLinear.solve_dispcontrol(
        St, LIST, 0, Node, 0,
        tol=1e-1,
        filename=f'Wallet_DispControl_psi={psi}',
        dir_name=save_path,
        max_iter=100
    )

    # Save structure
    St.save_structure(os.path.join(save_path, 'Wallet'))

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Pushover Analysis Results - Small Masonry Wall")
    print("="*60)
    print(f"Wall: {Blocks_Bed}x{Blocks_Head} blocks")
    print(f"Friction coefficient: {mu}")
    print(f"Dilatancy angle: {psi}")
    print(f"Results saved to: {save_path}")
    print("="*60)

    # Plot deformed structure
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=False,
            title="Small Wall - Deformed Configuration")

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
