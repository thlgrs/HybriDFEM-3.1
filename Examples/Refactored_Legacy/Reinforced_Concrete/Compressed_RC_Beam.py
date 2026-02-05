# -*- coding: utf-8 -*-
"""
Reinforced Concrete - Compressed RC Column
===========================================

Refactored from: Legacy/Examples/Reinforced Concrete/Elastic_RC_Beam/Compressed_RC_Beam.py

This example demonstrates compression analysis of a reinforced concrete
column with symmetric reinforcement, checking stress distribution.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear static (force control)

Configuration:
- Column: 3m height, 0.4m section, circular area equivalent
- Concrete: E=35 GPa, nu=0
- Steel: E=200 GPa (8x 16mm bars symmetric)
- Loading: 2000 kN axial compression
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
from Core import StaticNonLinear
from Core.Structures import BeamBlock
from Core.Objects.ConstitutiveLaw.Material import Material


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
    N1 = np.array([0, 0], dtype=float)  # Bottom
    N2 = np.array([0, 3], dtype=float)  # Top

    # Material properties
    E_c = 35e9     # Concrete Young's modulus [Pa]
    E_s = 200e9    # Steel Young's modulus [Pa]
    NU = 0.0       # Poisson's ratio

    # Discretization
    CPS = 10       # Contact pairs per interface
    BLOCKS = 2     # Number of blocks

    # Cross-section (circular equivalent)
    H = 0.4                    # Section depth [m]
    B = np.pi * H / 4          # Width for equivalent circular area

    # Reinforcement (8x 16mm bars)
    d = 16e-3                  # Bar diameter [m]
    A = 8 * np.pi * d ** 2 / 4 # Total steel area [m²]

    # ==========================================================================
    # Create Materials
    # ==========================================================================
    CONCR = Material(E_c, NU)
    STEEL = Material(E_s, NU, rho=7850)

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = BeamBlock(N1, N2, BLOCKS, H, rho=2500, b=B, material=CONCR)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces (linear geometry)
    St.make_cfs(lin_geom=True, nb_cps=CPS)

    # Add symmetric reinforcement to each contact face
    # Reinforcement at ±90% of section depth (near outer fibers)
    for cf in St.list_cfs:
        cf.add_reinforcement([-0.9, 0.9], A / 2, material=STEEL, height=d)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix bottom
    St.fix_node(0, [0, 1, 2])

    # Fix rotation at top (prevent lateral sway)
    St.fix_node(BLOCKS - 1, [2])

    # Apply compression load
    N = 2000e3  # Axial force [N]
    St.load_node(BLOCKS - 1, [1], -N)

    # ==========================================================================
    # Nonlinear Static Analysis (Force Control)
    # ==========================================================================
    print(f"\nRunning compression analysis of RC column...")
    print(f"  Column: {N2[1]-N1[1]}m height")
    print(f"  Section: {H}m (circular equivalent)")
    print(f"  Steel area: {A*1e6:.2f} mm²")
    print(f"  Applied load: {N/1000} kN")

    St = StaticNonLinear.solve_forcecontrol(
        St, steps=1,
        filename='Compressed_RC',
        dir_name=save_path,
        tol=1e-3
    )

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Compression Analysis Results - RC Column")
    print("="*60)

    # Displacement at top
    top_disp = St.U[-2] * 1000  # Convert to mm
    print(f"Top displacement: {top_disp:.4f} mm")

    # Expected stress (assuming all load taken by steel in compression)
    sigma_expected = N / A
    print(f"Expected steel stress (if all load on steel): {sigma_expected/1e6:.2f} MPa")

    # Actual stress distribution can be checked from contact face forces
    print(f"Results saved to: {save_path}")
    print("="*60)

    # ==========================================================================
    # Visualization
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=100, show_contact_faces=False,
            title="RC Column - Deformed Shape")
    plt.savefig(os.path.join(save_path, 'compressed_def_shape.png'), dpi=150)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
