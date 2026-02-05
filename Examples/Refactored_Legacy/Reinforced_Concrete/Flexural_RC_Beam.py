# -*- coding: utf-8 -*-
"""
Reinforced Concrete - Flexural Analysis of RC Beam
===================================================

Refactored from: Legacy/Examples/Reinforced Concrete/Elastic_RC_Beam/Flexural_RC_Beam.py

This example demonstrates flexural analysis of a reinforced concrete beam
using blocks with no-tension material for concrete and steel reinforcement.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear static (force control)

Configuration:
- Beam: 3m span, 0.5m height, 0.2m width
- Concrete: E=30 GPa, nu=0.2 (no-tension material)
- Steel: E=200 GPa, nu=0.3 (2x 10mm bars at 80% depth)
- Loading: Cantilever with 25 kN tip load
"""

import os
import sys
from pathlib import Path

import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import StaticNonLinear
from Core.Structures import BeamBlock
from Core.Objects.ConstitutiveLaw.Material import Material, NoTension_Mat


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
    N1 = np.array([0, 0], dtype=float)  # Fixed end
    N2 = np.array([3, 0], dtype=float)  # Free end

    # Material properties
    E_c = 30e9     # Concrete Young's modulus [Pa]
    E_s = 200e9    # Steel Young's modulus [Pa]
    NU_c = 0.2     # Concrete Poisson's ratio
    NU_s = 0.3     # Steel Poisson's ratio

    # Discretization
    CPS = 35       # Contact pairs per interface
    BLOCKS = 50    # Number of blocks

    # Cross-section
    H = 0.5        # Height [m]
    B = 0.2        # Width [m]

    # Reinforcement (2x 10mm bars)
    r = 10e-3      # Bar radius [m]
    A = 2 * np.pi * r ** 2  # Total steel area [m2]

    # ==========================================================================
    # Create Materials
    # ==========================================================================
    # Concrete: No-tension material (cracks in tension)
    CONCR = NoTension_Mat(E_c, NU_c)

    # Steel: Linear elastic material
    STEEL = Material(E_s, NU_s, rho=7850)

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    # Create beam with no-tension concrete
    St = BeamBlock(N1, N2, BLOCKS, H, rho=0, b=B, material=CONCR)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces (linear geometry)
    St.make_cfs(lin_geom=True, nb_cps=CPS)

    # Add steel reinforcement to each contact face
    # Reinforcement at 80% of height (near bottom fiber for positive bending)
    for cf in St.list_cfs:
        cf.add_reinforcement([0.8], A, material=STEEL, height=r)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix left end (cantilever)
    St.fix_node(0, [0, 1, 2])

    # Apply tip load at free end
    F = 25e3  # Force [N]
    St.load_node(BLOCKS - 1, [1], -F)

    # ==========================================================================
    # Nonlinear Static Analysis (Force Control)
    # ==========================================================================
    print(f"\nRunning flexural analysis of RC beam...")
    print(f"  Beam: {N2[0]-N1[0]}m x {H}m x {B}m")
    print(f"  Blocks: {BLOCKS}")
    print(f"  Steel area: {A*1e6:.2f} mm²")
    print(f"  Applied load: {F/1000} kN")

    St = StaticNonLinear.solve_forcecontrol(
        St, steps=2,
        filename='Flexural',
        dir_name=save_path,
        tol=10
    )

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Flexural Analysis Results - RC Beam")
    print("="*60)

    # Analytical solution for cracked section
    alph = E_s / E_c  # Modular ratio
    D = H - 0.05      # Effective depth [m]
    rho = A / (B * D) # Reinforcement ratio

    # Neutral axis depth (cracked section)
    x = D * alph * rho * (np.sqrt(1 + 2 / (alph * rho)) - 1)
    print(f"Neutral axis depth (analytical): {x*100:.2f} cm")

    # Stresses
    M = N2[0] * F     # Bending moment at support [Nm]
    s_max = M / ((1 - x / (3 * D)) * A * D)  # Steel stress
    c_max = 2 * M / ((1 - x / (3 * D)) * B * D ** 2 * x / D)  # Concrete stress

    print(f"Steel stress (analytical): {s_max/1e6:.2f} MPa")
    print(f"Concrete stress (analytical): {c_max/1e6:.2f} MPa")
    print(f"Results saved to: {save_path}")
    print("="*60)

    # ==========================================================================
    # Visualization
    # ==========================================================================
    import matplotlib.pyplot as plt

    # Plot deformed shape
    St.plot(show_deformed=True, deformation_scale=20, show_contact_faces=False,
            title="RC Beam - Deformed Shape")
    plt.savefig(os.path.join(save_path, 'def_shape.png'), dpi=150)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
