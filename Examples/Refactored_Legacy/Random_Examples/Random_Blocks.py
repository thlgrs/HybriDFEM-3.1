# -*- coding: utf-8 -*-
"""
Random Blocks - Arbitrary Polygon Block Assembly
=================================================

Refactored from: Legacy/Examples/Random examples/Random_Blocks.py

This example demonstrates creating a structure from arbitrary polygon blocks
with different numbers of vertices (triangular, quadrilateral, pentagonal).

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Structure visualization (contact face detection)

Configuration:
- 3 blocks with different shapes:
  - Block 1: Quadrilateral
  - Block 2: Triangle
  - Block 3: Pentagon
- Automatic contact face detection
- No-tension contact at interfaces
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
from Core import Structure_Block
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.ConstitutiveLaw.Contact import NoTension_EP


def main():
    # ==========================================================================
    # Output Directory Setup
    # ==========================================================================
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # ==========================================================================
    # Node Coordinates
    # ==========================================================================
    N0 = np.array([0, 0], dtype=float)
    N1 = np.array([3, 0], dtype=float)
    N2 = np.array([0, 2], dtype=float)
    N3 = np.array([2, 1], dtype=float)
    N4 = np.array([0, 3], dtype=float)
    N5 = np.array([3, 3], dtype=float)
    N6 = np.array([2, 4], dtype=float)

    # ==========================================================================
    # Material and Contact Parameters
    # ==========================================================================
    rho = 100.0       # Density [kg/m³]
    kn = 100.0        # Normal contact stiffness [N/m]
    ks = 100.0        # Shear contact stiffness [N/m]

    material = Material(E=30e9, nu=0.2, rho=rho)

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()

    # Block 1: Quadrilateral (4 vertices)
    vertices1 = np.array([N0, N1, N5, N3])
    St.add_block_from_vertices(vertices1, b=1.0, material=material)
    print(f"Block 1: Quadrilateral with {len(vertices1)} vertices")

    # Block 2: Triangle (3 vertices)
    vertices2 = np.array([N0, N3, N2])
    St.add_block_from_vertices(vertices2, b=1.0, material=material)
    print(f"Block 2: Triangle with {len(vertices2)} vertices")

    # Block 3: Pentagon (5 vertices)
    vertices3 = np.array([N2, N3, N5, N6, N4])
    St.add_block_from_vertices(vertices3, b=1.0, material=material)
    print(f"Block 3: Pentagon with {len(vertices3)} vertices")

    # ==========================================================================
    # Initialize Structure
    # ==========================================================================
    St.make_nodes()

    # Create contact faces with no-tension contact
    # nb_cps=2 is integer, so use contact= parameter
    St.make_cfs(
        lin_geom=True,
        nb_cps=2,
        offset=0.0,
        contact=NoTension_EP(kn, ks)
    )

    # ==========================================================================
    # Print Structure Information
    # ==========================================================================
    print(f"\nStructure Summary:")
    print(f"  Number of blocks: {len(St.list_blocks)}")
    print(f"  Number of nodes: {len(St.list_nodes)}")
    print(f"  Number of contact faces: {len(St.list_cfs)}")
    print(f"  Total DOFs: {len(St.U)}")

    for i, block in enumerate(St.list_blocks):
        print(f"\n  Block {i}:")
        print(f"    Vertices: {len(block.vertices)}")
        print(f"    Reference point: {block.ref_point}")
        print(f"    Mass: {block.m:.2f} kg")

    for i, cf in enumerate(St.list_cfs):
        print(f"\n  Contact Face {i}:")
        print(f"    Block A: {St.list_blocks.index(cf.bl_A)}")
        print(f"    Block B: {St.list_blocks.index(cf.bl_B)}")
        print(f"    Contact pairs: {len(cf.cps)}")

    # ==========================================================================
    # Plot Structure
    # ==========================================================================
    St.plot(show_contact_faces=True, title="Random Blocks - Arbitrary Polygons")
    plt.savefig(os.path.join(save_path, 'random_blocks.png'), dpi=150)

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Random Blocks Example Complete")
    print("="*60)
    print("This example demonstrates:")
    print("  - Creating blocks with arbitrary polygon shapes")
    print("  - Automatic contact face detection")
    print("  - Mixed polygon types (triangle, quad, pentagon)")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
