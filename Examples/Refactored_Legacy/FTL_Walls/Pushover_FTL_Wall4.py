# -*- coding: utf-8 -*-
"""
Ferris & Tin-Loi Wall 4 - Wall with Multiple Window Openings
=============================================================

Refactored from: Legacy/Examples/F&TL Walls (WCEE 2024)/Lin_Geom/Ferris&Tin-Loi_Wall4/Pushover_F&TL-4.py

Wall 4 features a more complex pattern with multiple window openings
distributed across different courses.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear static (displacement control)

Configuration:
- 11 courses with multiple window openings
- Block dimensions: 0.4m x 0.175m
- Coulomb friction contact (kn=ks=1e10 N/m, mu=psi=0.65)
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

    H_b = 0.175   # Block height [m]
    L_b = 0.4     # Block length [m]
    B = 1.0       # Out-of-plane thickness [m]

    kn = 1e10     # Normal stiffness [N/m]
    ks = 1e10     # Shear stiffness [N/m]
    mu = 0.65     # Friction coefficient
    psi = mu      # Dilatancy angle
    r_b = 0.00    # Contact offset [m]

    RHO = 1000.0  # Density [kg/m³]
    Blocks_Bed = 5

    # ==========================================================================
    # Generate Wall Pattern with Multiple Windows
    # ==========================================================================
    # Full course patterns
    Full_e = [0.5, 1., 1., 1., 1., 0.5]   # Even full course with half blocks at edges
    Full_u = [1.] * 5                      # Odd full course

    # Window course patterns (-1 = gap)
    Wind_u = [1., -1., 1., -1., 1.]       # Odd with windows
    Wind_e = [0.5, 0.5, -1., 0.5, 0.5, -1., 0.5, 0.5]  # Even with windows

    # Build pattern: 11 courses with alternating windows
    PATTERN = [
        Full_e, Full_u, Full_e,    # Courses 1-3: solid base
        Wind_u, Wind_e, Wind_u,    # Courses 4-6: window section
        Wind_e, Wind_u, Full_e,    # Courses 7-9: more windows
        Full_u, Full_e             # Courses 10-11: solid top
    ]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=RHO)

    # Add base block
    vertices = np.array([
        [Blocks_Bed * L_b, -H_b],
        [Blocks_Bed * L_b, 0],
        [0, 0],
        [0, -H_b]
    ])
    St.add_block_from_vertices(vertices, b=B, material=material)

    # Create wall with window openings
    Wall_temp = WallBlock(N1, L_b, H_b, PATTERN, RHO, b=B, material=material)
    for block in Wall_temp.list_blocks:
        St.list_blocks.append(block)

    St.make_nodes()
    St.make_cfs(lin_geom=True, nb_cps=2, contact=Coulomb(kn, ks, mu, psi=psi), offset=r_b)

    # ==========================================================================
    # Boundary Conditions and Loading
    # ==========================================================================
    St.fix_node(0, [0, 1, 2])
    Node = len(St.list_blocks) - 1

    W_c = 0
    for i in range(1, len(St.list_blocks)):
        W = St.list_blocks[i].m * 10
        St.load_node(i, [0], W)
        St.load_node(i, [1], -W, fixed=True)
        W_c = W

    St.plot(show_contact_faces=True, title="F&TL Wall 4 - Multiple Windows")

    # ==========================================================================
    # Displacement Control Analysis
    # ==========================================================================
    LIST = np.linspace(0, 5e-4, 2000).tolist()

    print(f"\nRunning pushover analysis (F&TL Wall 4 - Multiple Windows)...")
    print(f"  Number of blocks: {len(St.list_blocks)}")

    St = StaticNonLinear.solve_dispcontrol(
        St, LIST, 0, Node, 0,
        tol=1e-3,
        filename=f'Wall4_rb={r_b}_psi={psi}',
        dir_name=save_path,
        max_iter=1000
    )

    St.save_structure(os.path.join(save_path, 'FTL_Wall4'))
    St.plot(show_deformed=True, deformation_scale=1000, title="F&TL Wall 4 - Deformed")

    print("\n" + "="*60)
    print("F&TL Wall 4 - Multiple Window Openings Complete")
    print("="*60)

    plt.show()
    return St


if __name__ == "__main__":
    St = main()
