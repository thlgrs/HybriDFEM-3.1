# -*- coding: utf-8 -*-
"""
Nonlinear Dynamic - Single Block Rocking
=========================================

Refactored from: Legacy/Examples/Nonlinear_Dynamic/OneBlock/Oneblock_Rocking.py

This example demonstrates nonlinear dynamic analysis of a single rigid block
rocking on a fixed base, using Coulomb friction contact law.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear dynamic (CDM or Newmark integration)

Configuration:
- Upper Block: 0.4m x 0.4m (H x L), rho=1000 kg/m3
- Base Block: 1.0m x 0.2m (L x H), fixed
- Contact: Coulomb friction (kn=ks=1e8 N/m, mu=100 - essentially infinite friction)
- Loading: Self-weight + sinusoidal horizontal excitation
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
from Core import Structure_Block, StaticNonLinear, Dynamic
from Core.Objects.DFEM import Coulomb
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
    # Integration method: 'CDM' (Central Difference) or 'NWK' (Newmark)
    Meth = 'CDM'

    # Contact stiffness
    kn = 1e8  # Normal stiffness [N/m]
    ks = kn   # Shear stiffness [N/m]

    # Upper block geometry
    H = 0.4  # Height [m]
    L = 0.4  # Length [m]
    B = 1    # Thickness (out-of-plane) [m]

    rho = 1000  # Density [kg/m3]

    # Base block geometry
    L_base = 1.0   # Length [m]
    H_base = 0.2   # Height [m]

    # Reference points for blocks
    N1 = np.array([0, -H_base / 2], dtype=float)  # Base block center
    N2 = np.array([0, H / 2], dtype=float)        # Upper block center

    # Unit vectors
    x = np.array([0.5, 0])
    y = np.array([0, 0.5])

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    # In Legacy: St = st.Structure_2D()
    # In Core: St = Structure_Block()
    St = Structure_Block()

    # Create base block vertices (fixed)
    # In Legacy: St.add_block(vertices, rho, b=B)
    # In Core: St.add_block_from_vertices(vertices, b=B, material=Material(rho=rho))
    vertices_base = np.array([N1, N1, N1, N1])
    vertices_base[0] += L_base * x - H_base * y
    vertices_base[1] += L_base * x + H_base * y
    vertices_base[2] += -L_base * x + H_base * y
    vertices_base[3] += -L_base * x - H_base * y

    St.add_block_from_vertices(vertices_base, b=B, material=Material(E=1e9, nu=0.3, rho=rho))

    # Create upper block vertices (rocking block)
    vertices_upper = np.array([N2, N2, N2, N2])
    vertices_upper[0] += L * x - H * y
    vertices_upper[1] += L * x + H * y
    vertices_upper[2] += -L * x + H * y
    vertices_upper[3] += -L * x - H * y

    St.add_block_from_vertices(vertices_upper, b=B, material=Material(E=1e9, nu=0.3, rho=rho))

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with Coulomb friction
    # In Legacy: St.make_cfs(False, nb_cps=2, offset=0.0, contact=cont.Coulomb(kn, ks, 100))
    # In Core: St.make_cfs(lin_geom=False, nb_cps=2, offset=0.0, contact=Coulomb(kn, ks, mu, c, psi))
    St.make_cfs(
        lin_geom=False,  # Nonlinear geometry
        nb_cps=2,
        offset=0.0,
        contact=Coulomb(kn=kn, ks=ks, mu=100, c=0, psi=0)  # High friction
    )

    # ==========================================================================
    # Get Block Properties
    # ==========================================================================
    M = St.list_blocks[1].m  # Mass of upper block
    W = 9.81 * M             # Weight
    I = St.list_blocks[1].I  # Rotational inertia

    print(f"Weight of upper block: {W:.2f} N")
    print(f"Rotational inertia: {I:.4f} kg.m^2")

    # ==========================================================================
    # Boundary Conditions
    # ==========================================================================
    # Apply self-weight to upper block
    # In Legacy: St.loadNode(1, [1], -W, fixed=True)
    # In Core: St.load_node(1, [1], -W, fixed=True)
    St.load_node(1, [1], -W, fixed=True)

    # Fix base block (node 0)
    # In Legacy: St.fixNode(0, [0, 1, 2])
    # In Core: St.fix_node(0, [0, 1, 2])
    St.fix_node(0, [0, 1, 2])

    # ==========================================================================
    # Initial Static Analysis
    # ==========================================================================
    # Settle under self-weight
    # In Legacy: St.solve_forcecontrol(10)
    # In Core: StaticNonLinear.solve_forcecontrol(St, steps=10)
    St = StaticNonLinear.solve_forcecontrol(St, steps=10, dir_name=save_path)

    # ==========================================================================
    # Excitation Setup
    # ==========================================================================
    # Apply horizontal load for excitation
    # In Legacy: St.loadNode(1, [0], W)
    # In Core: St.load_node(1, [0], W)
    St.load_node(1, [0], W)

    # Calculate rocking period
    R = 2 * 0.141  # Effective radius
    period = 2 * np.sqrt(I / (W * R))

    AMP = np.pi / 2 * 1.2  # Amplitude
    lag = 0

    def lmbda(t):
        """Sinusoidal load multiplier."""
        if t < period + lag:
            return AMP * np.sin((t - lag) * np.pi / period)
        return 0

    # Plot loading function
    t_plot = np.linspace(0, 2, 100)
    load_plot = [lmbda(t) for t in t_plot]
    plt.figure(figsize=(8, 4))
    plt.plot(t_plot, load_plot)
    plt.xlabel("Time [s]")
    plt.ylabel("Load multiplier")
    plt.title("Sinusoidal Excitation")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "excitation.png"), dpi=150)
    plt.close()

    # ==========================================================================
    # Set Damping
    # ==========================================================================
    # In Legacy: St.set_damping_properties(xsi=0.00, damp_type='RAYLEIGH')
    # In Core: St.set_damping_properties(xsi=0.00, damp_type='RAYLEIGH')
    St.set_damping_properties(xsi=0.00, damp_type='RAYLEIGH')

    # ==========================================================================
    # Nonlinear Dynamic Analysis
    # ==========================================================================
    # In Legacy: St.solve_dyn_nonlinear(2, 1e-4, Meth=Meth, lmbda=lmbda)
    # In Core: Dynamic(T, dt, ...).nonlinear(St)

    T_end = 2.0    # Total time [s]
    dt = 1e-4      # Time step [s]

    print(f"\nRunning nonlinear dynamic analysis (Meth={Meth}, dt={dt})")

    solver = Dynamic(
        T=T_end,
        dt=dt,
        Meth=Meth,
        lmbda=lmbda,
        filename='Rocking_block',
        dir_name=save_path
    )

    St = solver.nonlinear(St)

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Nonlinear Dynamic Analysis Results - Block Rocking")
    print("="*60)
    print(f"Integration method: {Meth}")
    print(f"Time step: {dt} s")
    print(f"Total time: {T_end} s")
    print(f"Results saved to: {save_path}")
    print("="*60)

    # ==========================================================================
    # Save Structure
    # ==========================================================================
    # In Legacy: St.save_structure(filename=...)
    # In Core: St.save_structure(filename)
    St.save_structure(os.path.join(save_path, 'Rocking_block'))

    # ==========================================================================
    # Visualization
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1, title="Rocking Block - Final State")
    plt.savefig(os.path.join(save_path, "rocking_block_final.png"), dpi=150)
    plt.show()

    return St


if __name__ == "__main__":
    St = main()
