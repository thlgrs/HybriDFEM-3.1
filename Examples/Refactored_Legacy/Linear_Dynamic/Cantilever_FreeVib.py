# -*- coding: utf-8 -*-
"""
Linear Dynamic - Cantilever Free Vibration
===========================================

Refactored from: Legacy/Examples/Linear_Dynamic/Linear_Cantilever/Cantilever_FreeVib.py

This example demonstrates linear dynamic analysis of a cantilever beam
discretized with rigid blocks, subjected to an initial load and then
released to vibrate freely.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Linear dynamic (time-history using Linear Acceleration method)

Configuration:
- Geometry: 3m length, 0.5m height, 0.2m thickness
- Material: E=30 GPa, nu=0.0, rho=7000 kg/m3
- Blocks: 5 blocks
- Contact: Linear elastic, 15 contact pairs per interface
- Loading: Initial tip load of -100 kN
"""

import os
import sys
from pathlib import Path

import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core import Static, Modal, Dynamic, Visualizer
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
    N1 = np.array([0, 0], dtype=float)  # Fixed end
    N2 = np.array([3, 0], dtype=float)  # Free end

    H = 0.5   # Height [m]
    B = 0.2   # Width (out-of-plane) [m]

    BLOCKS = 5   # Number of blocks
    CPS = 15     # Contact pairs per interface

    E = 30e9     # Young's modulus [Pa]
    NU = 0.0     # Poisson's ratio
    RHO = 7000   # Density [kg/m3]

    F = -100e3   # Initial tip load [N]

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    # In Legacy: St = st.Structure_2D()
    #            St.add_beam(N1, N2, BLOCKS, H, RHO, b=B, material=mat.Material(E, NU, ...))
    # In Core: BeamBlock handles this directly

    material = Material(E, NU, rho=RHO, shear_def=True)

    # Create beam block structure
    St = BeamBlock(N1, N2, BLOCKS, H, rho=RHO, b=B, material=material)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces
    St.make_cfs(lin_geom=True, nb_cps=CPS)

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Fix the left end at N1
    # In Legacy: St.fixNode(N1, [0, 1, 2])
    # In Core: St.fix_node(node_id, dofs)

    node_N1 = St.get_node_id(N1)
    node_N2 = St.get_node_id(N2)

    if node_N1 is not None:
        St.fix_node(node_N1, [0, 1, 2])
    else:
        print("Warning: Could not find node at N1")

    # ==========================================================================
    # Initial Static Analysis (for initial displacement)
    # ==========================================================================
    # Apply tip load to get initial deformed shape
    # In Legacy: St.loadNode(N2, [1], F, fixed=True)
    # In Core: St.load_node(node_id, dofs, force, fixed=True)

    if node_N2 is not None:
        St.load_node(node_N2, [1], F, fixed=True)

    # Solve linear static
    # In Legacy: St.solve_linear()
    # In Core: Static.solve(St)
    St = Static.solve(St)

    print(f"Initial tip displacement: {St.U[-4:]}")

    # ==========================================================================
    # Modal Analysis
    # ==========================================================================
    # In Legacy: St.solve_modal(filename=f'{BLOCKS}', dir_name='conv_blocks')
    # In Core: Modal.solve_modal(St, ...)

    St = Modal.solve_modal(St, save=False)

    print(f"\nFirst 4 natural frequencies (rad/s): {St.eig_vals[:4]}")
    print(f"Maximum frequency: {max(St.eig_vals)}")

    for i in range(4):
        print(f"Mode {i+1} tip DOFs: {St.eig_modes.T[i, -4:]}")

    # ==========================================================================
    # Reset Loading for Dynamic Analysis
    # ==========================================================================
    # In Legacy: St.reset_loading()
    # In Core: St.reset_loading()
    St.reset_loading()

    # Set up damping
    # In Legacy: St.set_damping_properties(xsi=0.00, damp_type='STIFF')
    # In Core: St.set_damping_properties(xsi=0.00, damp_type='STIFF')
    St.set_damping_properties(xsi=0.00, damp_type='STIFF')

    # ==========================================================================
    # Linear Dynamic Analysis
    # ==========================================================================
    # In Legacy: St.solve_dyn_linear(5, dt, Meth='LA', dir_name=..., filename=...)
    # In Core: Dynamic(T, dt, ...).linear(St)

    Meth = 'LA'  # Linear Acceleration method
    dt = 1.61e-4  # Time step [s]
    T_end = 5.0   # Total simulation time [s]

    print(f"\nRunning linear dynamic analysis ({BLOCKS} blocks, Meth={Meth}, dt={dt})")

    solver = Dynamic(
        T=T_end,
        dt=dt,
        Meth=Meth,
        filename=f'dt_{BLOCKS}',
        dir_name=save_path
    )

    St = solver.linear(St)

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Linear Dynamic Analysis Results - Cantilever Free Vibration")
    print("="*60)
    print(f"Number of blocks: {BLOCKS}")
    print(f"Time step: {dt} s")
    print(f"Integration method: {Meth}")
    print(f"Results saved to: {save_path}")
    print("="*60)

    # ==========================================================================
    # Visualization
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=20, title="Cantilever Beam - Deformed")

    import matplotlib.pyplot as plt
    plt.show()

    return St


if __name__ == "__main__":
    St = main()
