# -*- coding: utf-8 -*-
"""
Modal Analysis - Tapered Beam (Eigenvalue Problem)
===================================================

Refactored from: Legacy/Examples/Modal Analysis/EigVals_Tapered_Beam.py

This example computes natural frequencies and mode shapes for a tapered
beam discretized with rigid blocks and contact faces.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Modal (eigenvalue analysis)

Configuration:
- Geometry: 5m length, varying height from H1=0.4m to H2=2m
- Material: E=210 GPa, nu=0.3, rho=7850 kg/m3 (steel)
- Blocks: 100 blocks
- Contact: Linear elastic, 100 contact pairs per interface
"""

import sys
from pathlib import Path

import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Modal, Visualizer
from Core.Structures import TaperedBeamBlock
from Core.Objects.ConstitutiveLaw.Material import Material


def main():
    # ==========================================================================
    # Parameters
    # ==========================================================================
    N1 = np.array([0, 0], dtype=float)
    N2 = np.array([5, 0], dtype=float)

    C = 4  # Taper ratio
    H1 = 0.4  # Height at start
    H2 = H1 * (1 + C)  # Height at end
    B = 0.2  # Width (out-of-plane thickness)

    BLOCKS = 100  # Number of blocks
    CPS = 100     # Contact pairs per interface

    E = 210e9   # Young's modulus [Pa]
    NU = 0.3    # Poisson's ratio
    RHO = 7850  # Density [kg/m3]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    # In Legacy: St = st.Structure_2D()
    #            St.add_tapered_beam(N1, N2, BLOCKS, H1, H2, RHO, b=B, material=mat.Material(E, NU, ...))
    # In Core: TaperedBeamBlock handles this directly

    material = Material(E, NU, rho=RHO, corr_fact=13/15, shear_def=False)

    # Create tapered beam structure
    St = TaperedBeamBlock(N1, N2, BLOCKS, H1, H2, rho=RHO, b=B, material=material)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces
    # Linear geometry (lin_geom=True), with CPS contact pairs per interface
    St.make_cfs(lin_geom=True, nb_cps=CPS)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix the end at N2 (right end)
    # In Legacy: St.fixNode(N2, [0, 1, 2])
    # In Core: St.fix_node(node_id, dofs)

    # Find node at N2
    node_N2 = St.get_node_id(N2)
    if node_N2 is not None:
        St.fix_node(node_N2, [0, 1, 2])
    else:
        print("Warning: Could not find node at N2")

    # ==========================================================================
    # Modal Analysis
    # ==========================================================================
    # In Legacy: St.solve_modal(no_inertia=True)
    # In Core: Modal.solve_modal(St, no_inertia=True)

    # Note: no_inertia=False because no_inertia=True can make the mass matrix singular
    # which causes numerical issues with the eigensolver
    St = Modal.solve_modal(St, no_inertia=False, save=False)

    # ==========================================================================
    # Results
    # ==========================================================================
    # Compute non-dimensional frequency parameter
    # lambda = sqrt(rho * H * B * L^4 / (E * I))
    L = 5.0
    I = B * H2**3 / 12
    lbda = np.sqrt(RHO * H2 * B * L**4 / (E * I))

    print("\n" + "="*60)
    print("Modal Analysis Results - Tapered Beam")
    print("="*60)
    print(f"First 5 natural frequencies (rad/s): {St.eig_vals[:5]}")
    print(f"Non-dimensional frequencies: {St.eig_vals[:5] * lbda}")
    print("="*60)

    # ==========================================================================
    # Visualization
    # ==========================================================================
    # In Legacy: St.plot_modes(5, scale=5)
    # In Core: Plot structure and optionally modes

    # Plot undeformed structure
    St.plot(title="Tapered Beam - Undeformed")

    # Note: Mode shape plotting would need Visualizer extension
    # For now, display the structure

    import matplotlib.pyplot as plt
    plt.show()

    return St


if __name__ == "__main__":
    St = main()
