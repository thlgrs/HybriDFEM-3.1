# -*- coding: utf-8 -*-
"""
Cyclic Loading - Coulomb Friction with Linear Contact
=====================================================

Refactored from: Legacy/Examples/Structures_2024/Rocking_Blocks/Coulomb_Lin_Cont.py

This example demonstrates cyclic displacement-controlled loading of two
stacked blocks with Coulomb friction contact.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear static (displacement control, cyclic)

Configuration:
- Two stacked blocks: 2m x 2m each
- Contact: Coulomb friction (kn=ks=200 N/m, mu=0.5, psi=1)
- Loading: Cyclic horizontal displacement at top
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
from Core import StaticNonLinear
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
    N1 = np.array([0, 0])  # Origin

    H = 2.0   # Block height [m]
    B = 2.0   # Block width [m]

    # Contact stiffness
    kn = 2e2  # Normal stiffness [N/m]
    ks = 2e2  # Shear stiffness [N/m]
    mu = 0.5  # Friction coefficient
    psi = 1.0 # Dilatancy angle (associated flow rule)

    # Density chosen so that each block has unit weight
    RHO = 1 / (H * B * 9.81)

    # Pattern: 2 stacked blocks
    PATTERN = [[1], [1]]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    material = Material(E=30e9, nu=0.2, rho=RHO)
    St = WallBlock(N1, B, H, PATTERN, rho=RHO, b=1.0, material=material)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with Coulomb friction
    St.make_cfs(lin_geom=False, nb_cps=2, contact=Coulomb(kn, ks, mu, psi=psi), offset=0.0)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix bottom block
    St.fix_node(0, [0, 1, 2])

    # Apply loading
    W = 1.0  # Reference force
    St.load_node(1, [0], W)               # Horizontal force
    St.load_node(1, [1], -W, fixed=True)  # Self-weight (fixed)

    # ==========================================================================
    # Plot Initial Structure
    # ==========================================================================
    St.plot(title="Two Blocks - Initial")

    # ==========================================================================
    # Cyclic Displacement Control
    # ==========================================================================
    Node = 1      # Control node (top block)
    d_end = 2e-2  # Maximum displacement [m]

    # Create cyclic loading protocol: 4 cycles of push-pull
    LIST = np.array([])
    for _ in range(4):
        LIST = np.append(LIST, np.linspace(0, d_end, 50))
        LIST = np.append(LIST, np.linspace(d_end, 0, 50))
    LIST = LIST.tolist()

    print(f"\nRunning cyclic displacement control...")
    print(f"  Control node: {Node}")
    print(f"  Max displacement: {d_end*1000} mm")
    print(f"  Number of cycles: 4")

    St = StaticNonLinear.solve_dispcontrol(
        St, LIST, 0, Node, 0,
        tol=1e-5,
        filename='Coulomb_Lin_Cont',
        dir_name=save_path,
        max_iter=100
    )

    print(f"\nFinal displacements (mm): {St.U[-3:] * 1000}")

    # Save structure
    St.save_structure(os.path.join(save_path, 'Wallet'))

    # ==========================================================================
    # Plot Deformed Structure
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=10, show_contact_faces=True,
            title="Two Blocks - Deformed")

    # ==========================================================================
    # Plot Force-Displacement Curve
    # ==========================================================================
    result_file = os.path.join(save_path, 'Coulomb_Lin_Cont.h5')
    if os.path.exists(result_file):
        with h5py.File(result_file, 'r') as hf:
            U = hf['U_conv'][-3] * 1000   # Horizontal displacement [mm]
            U_v = hf['U_conv'][-2] * 1000 # Vertical displacement [mm]
            P = hf['P_r_conv'][-3]        # Horizontal reaction [N]

        plt.figure(figsize=(8, 6))
        plt.plot(U, P, 'k-', linewidth=1.0, label='HybriDFEM')
        plt.axhline(y=mu, color='g', linestyle='--', label=f'Sliding limit (μ={mu})')
        plt.xlabel('Horizontal Displacement [mm]')
        plt.ylabel('Horizontal Force [N]')
        plt.title('Force-Displacement Curve - Cyclic Loading')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'force_displacement.png'), dpi=150)

        # Dilatancy plot
        plt.figure(figsize=(8, 6))
        plt.plot(U, U_v, 'b-', linewidth=1.0)
        plt.xlabel('Horizontal Displacement [mm]')
        plt.ylabel('Vertical Displacement [mm]')
        plt.title('Dilatancy - Cyclic Loading')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'dilatancy.png'), dpi=150)

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Cyclic Loading Results - Coulomb Friction")
    print("="*60)
    print(f"Friction coefficient: {mu}")
    print(f"Dilatancy angle: {psi}")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
