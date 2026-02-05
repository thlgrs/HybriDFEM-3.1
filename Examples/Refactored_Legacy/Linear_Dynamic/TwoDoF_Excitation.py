# -*- coding: utf-8 -*-
"""
Linear Dynamic - 2 DOF System Under Harmonic Excitation
=======================================================

Refactored from: Legacy/Examples/Linear_Dynamic/Linear_2DoF/2DoF_Excitation.py

This example demonstrates linear dynamic analysis of a simple 2-DOF system
(3 blocks stacked vertically) under harmonic excitation.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Linear dynamic (HHT-alpha integration)

Configuration:
- Geometry: 3 stacked blocks (1m x 1m each)
- Material: rho=6000 kg/m3
- Contact: Linear elastic surface (kn=ks=400 kN/m3)
- Loading: Harmonic force P(t) = P0 * sin(w*t)
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
from Core import Dynamic
from Core.Structures import WallBlock
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.DFEM.Surface import Surface


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
    N1 = np.array([0, 0], dtype=float)  # Origin

    BLOCKS = 3   # Number of blocks (stacked vertically)
    CPS = 100    # Contact pairs per interface

    # Contact stiffness
    kn = 200e3 * 2  # Normal stiffness [N/m3]
    ks = kn         # Shear stiffness [N/m3]

    # Loading parameters
    w_s = 10       # Excitation frequency [rad/s]
    p0 = 1.66e4    # Force amplitude [N]

    RHO = 6000     # Density [kg/m3]

    # Integration method: HHT-alpha with alpha=0 (equivalent to Newmark)
    Meth = ['HHT', 0.0]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    # Create wall pattern (single column of 3 blocks)
    pattern = [[1], [1], [1]]  # 3 rows, 1 block each

    material = Material(E=30e9, nu=0.2, rho=RHO)
    St = WallBlock(N1, 1/3, 1.0, pattern, rho=RHO, b=1, material=material)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with linear elastic surface law
    St.make_cfs(lin_geom=True, nb_cps=CPS, surface=Surface(kn, ks))

    # ==========================================================================
    # Excitation Function
    # ==========================================================================
    def excitation(t):
        """Harmonic excitation: sin(w*t)"""
        return np.sin(w_s * t)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix base block
    St.fix_node(0, [0, 1, 2])

    # Apply horizontal load to middle block (node 1)
    St.load_node(1, [0], p0)

    # ==========================================================================
    # Set Damping
    # ==========================================================================
    St.set_damping_properties(xsi=0.05, damp_type='RAYLEIGH')

    # ==========================================================================
    # Plot Initial Structure
    # ==========================================================================
    St.plot(show_contact_faces=True, title="2-DOF System - Initial")

    # ==========================================================================
    # Linear Dynamic Analysis
    # ==========================================================================
    print(f"\nRunning linear dynamic analysis...")
    print(f"  Method: HHT-alpha (alpha=0)")
    print(f"  Time step: 0.01 s")
    print(f"  Total time: 10 s")
    print(f"  Excitation frequency: {w_s} rad/s")

    solver = Dynamic(
        T=10.0,
        dt=1e-2,
        Meth=Meth,
        lmbda=excitation,
        filename='2DoF_Vibration',
        dir_name=save_path
    )

    St = solver.linear(St)

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Linear Dynamic Analysis Results - 2 DOF System")
    print("="*60)
    print(f"Excitation frequency: {w_s} rad/s = {w_s/(2*np.pi):.2f} Hz")
    print(f"Force amplitude: {p0} N")
    print(f"Damping ratio: 5%")
    print(f"Results saved to: {save_path}")
    print("="*60)

    # Save structure
    St.save_structure(os.path.join(save_path, '2DoF_Vibration'))

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
