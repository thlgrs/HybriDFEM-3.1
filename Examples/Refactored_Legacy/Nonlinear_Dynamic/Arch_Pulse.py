# -*- coding: utf-8 -*-
"""
Nonlinear Dynamic - Arch Under Pulse Loading (Lemos)
====================================================

Refactored from: Legacy/Examples/Nonlinear_Dynamic/Lemos_Arch_Pulse/Arch_Pulse.py

This example demonstrates nonlinear dynamic analysis of a semicircular
masonry arch under horizontal pulse loading, using Coulomb friction contact.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Nonlinear dynamic (Newmark CAA integration)

Configuration:
- Arch: R=3.75m, 17 voussoirs, 0.5m thick
- Base: Fixed rigid block
- Contact: Coulomb friction (kn=20 GPa, ks=8 GPa, mu=0.4)
- Loading: Self-weight + sinusoidal horizontal pulse
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
from Core.Structures import ArchBlock
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.DFEM.Surface import Coulomb as SurfaceCoulomb


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
    # Contact stiffness
    kn = 20e9   # Normal stiffness [N/m3]
    ks = 0.4 * kn  # Shear stiffness [N/m3]
    mu = 0.4    # Friction coefficient

    # Arch geometry
    B = 0.5     # Out-of-plane thickness [m]
    H = 0.5     # Voussoir thickness [m]
    R = 7.5 / 2 + H / 2  # Arch radius to centerline [m]
    nb_blocks = 17  # Number of voussoirs

    rho = 2700  # Density [kg/m3]

    # Base block geometry
    N1 = np.array([0, 0], dtype=float)  # Arch center at base level
    H_base = 0.5
    L_base = 1.1 * 2 * R

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    St = Structure_Block()
    material = Material(E=30e9, nu=0.2, rho=rho)

    # Create base block vertices
    x = np.array([0.5, 0])
    y = np.array([0, 1])
    vertices_base = np.array([N1, N1, N1, N1], dtype=float)
    vertices_base[0] = N1 + L_base * x - H_base * y
    vertices_base[1] = N1 + L_base * x
    vertices_base[2] = N1 - L_base * x
    vertices_base[3] = N1 - L_base * x - H_base * y

    St.add_block_from_vertices(vertices_base, b=B, material=material)

    # Create arch voussoirs
    arch = ArchBlock(N1, 0, np.pi, R, nb_blocks, H, rho=rho, b=B, material=material)
    for block in arch.list_blocks:
        St.list_blocks.append(block)

    # Initialize nodes and DOFs
    St.make_nodes()

    # Create contact faces with Coulomb friction
    nb_cps = [-1, 0, 1]  # Contact points at edges and center
    St.make_cfs(lin_geom=False, nb_cps=nb_cps, offset=-1,
                surface=SurfaceCoulomb(kn, ks, mu))

    # ==========================================================================
    # Boundary Conditions - Fix Base
    # ==========================================================================
    St.fix_node(0, [0, 1, 2])

    # ==========================================================================
    # Apply Self-Weight
    # ==========================================================================
    for i in range(1, nb_blocks + 1):
        M = St.list_blocks[i].m
        W = M * 10  # Weight (g = 10 m/s2)
        St.load_node(i, [1], -W)

    # ==========================================================================
    # Initial Static Analysis (Settle under self-weight)
    # ==========================================================================
    print("Settling arch under self-weight...")
    St = StaticNonLinear.solve_forcecontrol(St, steps=20, tol=1, max_iter=100,
                                             dir_name=save_path)

    # Plot settled configuration
    St.plot(show_deformed=True, deformation_scale=1,
            title="Arch - Settled Under Self-Weight")

    # ==========================================================================
    # Reset Loading for Dynamic Analysis
    # ==========================================================================
    St.reset_loading()

    # Reapply self-weight as fixed load
    for i in range(nb_blocks):
        M = St.list_blocks[i].m
        W = M * 10
        St.load_node(i, [1], -W, fixed=True)
        St.load_node(i, [0], -W)  # Horizontal load for pulse

    # ==========================================================================
    # Excitation Function and Damping
    # ==========================================================================
    t_p = 0.25     # Pulse period [s]
    w_s = np.pi / t_p  # Angular frequency
    a = -0.15      # Amplitude (fraction of g)
    lag = 0        # Delay before pulse starts

    def lmbda(t):
        """Sinusoidal pulse load multiplier."""
        if t < lag:
            return 0
        if t < t_p + lag:
            return a * np.sin(w_s * (t - lag))
        return 0

    # Plot excitation function
    t_plot = np.linspace(0, 1, 200)
    load_plot = [lmbda(t) for t in t_plot]
    plt.figure(figsize=(8, 4))
    plt.plot(t_plot, load_plot)
    plt.xlabel("Time [s]")
    plt.ylabel("Load multiplier")
    plt.title(f"Horizontal Pulse Excitation (a={-a}g, T={t_p}s)")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "arch_pulse_excitation.png"), dpi=150)
    plt.close()

    # Set damping
    damp = 0.05
    stiff_type = 'TAN'
    St.set_damping_properties(xsi=damp, damp_type='STIFF', stiff_type=stiff_type)

    # ==========================================================================
    # Nonlinear Dynamic Analysis
    # ==========================================================================
    print(f"\nRunning nonlinear dynamic analysis...")
    print(f"  Method: CAA (Constant Average Acceleration)")
    print(f"  Time step: 2e-3 s")
    print(f"  Total time: 2.5 s")
    print(f"  Damping: {damp*100}%")

    Meth = 'CAA'
    solver = Dynamic(
        T=2.5,
        dt=2e-3,
        Meth=Meth,
        lmbda=lmbda,
        filename=f'{stiff_type}_Coulomb_{-a}g_{damp}',
        dir_name=save_path
    )

    St = solver.nonlinear(St)

    # ==========================================================================
    # Results
    # ==========================================================================
    print("\n" + "="*60)
    print("Nonlinear Dynamic Analysis Results - Arch Under Pulse")
    print("="*60)
    print(f"Integration method: {Meth}")
    print(f"Pulse amplitude: {-a}g")
    print(f"Pulse period: {t_p} s")
    print(f"Damping ratio: {damp*100}%")
    print(f"Results saved to: {save_path}")
    print("="*60)

    # Plot final configuration
    St.plot(show_deformed=True, deformation_scale=1,
            title="Arch - Final State After Pulse")

    # Save structure
    St.save_structure(os.path.join(save_path, 'Lemos_Arch_Coulomb'))

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
