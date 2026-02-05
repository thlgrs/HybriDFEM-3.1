# -*- coding: utf-8 -*-
"""
Arch Stroobants - Dynamic Analysis
==================================

Refactored from: Legacy/Examples/Giacomo/Arche_Stroobants.py

This example demonstrates static and dynamic analysis of a circular arch
assembly with triangular support blocks under gravitational and seismic loading.

Structure Type: Pure DFEM (rigid blocks with 3 DOF/node)
Analysis: Static force control + Modal + Nonlinear dynamic (CDM)

Configuration:
- Circular arch with 16 voussoirs
- Mean radius: 200mm, thickness: 30mm
- Linear elastic contact
- Base harmonic excitation simulation
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
from Core import Structure_Block, StaticNonLinear, Modal, Dynamic
from Core.Structures import ArchBlock
from Core.Objects.ConstitutiveLaw.Material import Material
from Core.Objects.ConstitutiveLaw.Contact import Contact


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
    R_med = 200e-3    # Mean radius [m]
    t = 30e-3         # Radial thickness [m]

    R_int = R_med - t / 2
    R_ext = R_med + t / 2

    mu = 1.0          # Friction coefficient (not used for Contact)
    rho = 2600        # Density [kg/m³]

    CPS = 2           # Contact pairs per interface
    kn = 40e6         # Normal stiffness [N/m]
    ks = kn           # Shear stiffness [N/m]

    # ==========================================================================
    # Create Arch Structure
    # ==========================================================================
    # Create arch using ArchBlock generator
    N0 = np.array([0, 0], dtype=float)
    material = Material(E=30e9, nu=0.2, rho=rho)

    # Create arch from angle pi/20 to 19*pi/20 (nearly semicircular)
    Arch_temp = ArchBlock(
        c=N0,
        a1=np.pi / 20,
        a2=np.pi * 19 / 20,
        R=R_med,
        n_blocks=16,
        h=t,
        rho=rho,
        b=1.0,
        material=material
    )

    # Create main structure and add arch blocks
    St = Structure_Block()
    for block in Arch_temp.list_blocks:
        St.list_blocks.append(block)

    # ==========================================================================
    # Add Support Blocks
    # ==========================================================================
    # Base support (rectangular block below arch)
    N1 = np.array([-300e-3, -50e-3], dtype=float)
    N2 = np.array([300e-3, -50e-3], dtype=float)
    N3 = np.array([300e-3, 0], dtype=float)
    N4 = np.array([-300e-3, 0], dtype=float)

    vertices_base = np.array([N1, N2, N3, N4])
    St.add_block_from_vertices(vertices_base, b=1.0, material=Material(E=30e9, nu=0.2, rho=100))

    # Left triangular support (fills gap at left springing)
    N5 = np.array([-R_ext, 0], dtype=float)
    N6 = np.array([-R_ext * np.sin(np.pi * 9 / 20), R_ext * np.cos(np.pi * 9 / 20)], dtype=float)

    vertices_left = np.array([N0, N5, N6])
    St.add_block_from_vertices(vertices_left, b=1.0, material=Material(E=30e9, nu=0.2, rho=100))

    # Right triangular support (fills gap at right springing)
    N7 = np.array([R_ext, 0], dtype=float)
    N8 = np.array([R_ext * np.sin(np.pi * 9 / 20), R_ext * np.cos(np.pi * 9 / 20)], dtype=float)

    vertices_right = np.array([N0, N7, N8])
    St.add_block_from_vertices(vertices_right, b=1.0, material=Material(E=30e9, nu=0.2, rho=100))

    # ==========================================================================
    # Initialize Structure
    # ==========================================================================
    St.make_nodes()

    # Create contact faces with linear elastic contact
    # Using nb_cps=10 with list positions requires surface parameter
    cps_list = np.linspace(-1, 1, 10).tolist()
    St.make_cfs(
        lin_geom=True,
        nb_cps=cps_list,
        offset=-1,
        surface=Contact(kn, ks)
    )

    # ==========================================================================
    # Boundary Conditions and Loading
    # ==========================================================================
    # Block indices: 0-15 are arch blocks, 16 is base, 17 is left support, 18 is right support
    St.fix_node(16, [0, 1, 2])  # Fix base block
    St.fix_node(17, [0, 1, 2])  # Fix left support
    St.fix_node(18, [0, 1, 2])  # Fix right support

    # Apply gravity to arch blocks only
    for i in range(0, 16):
        M = St.list_blocks[i].m
        W = 9.81 * M
        St.load_node(i, [1], -W, fixed=False)

    # ==========================================================================
    # Plot Initial Structure
    # ==========================================================================
    St.plot(show_contact_faces=False, title="Arch Stroobants - Initial")

    # ==========================================================================
    # Static Analysis (Force Control)
    # ==========================================================================
    print("\nRunning static force control analysis...")
    St = StaticNonLinear.solve_forcecontrol(
        St, 10,
        dir_name=save_path,
        filename='Results_arch_Stroobants_fc'
    )

    # ==========================================================================
    # Modal Analysis
    # ==========================================================================
    print("\nRunning modal analysis...")
    nb_modes = 4
    St = Modal.solve(St, nb_modes, filename=os.path.join(save_path, 'Arch_Modal'))

    if St.eig_vals is not None and len(St.eig_vals) >= nb_modes:
        freqs = np.sqrt(np.abs(St.eig_vals[:nb_modes])) / (2 * np.pi)
        print(f"Natural frequencies: {freqs} Hz")
        Modal.plot_modes(St, nb_modes, scale=0.1, title="Arch Modes")

    # ==========================================================================
    # Dynamic Analysis with Harmonic Excitation
    # ==========================================================================
    T = 0.18        # Period [s]
    A_max = 0.28    # Maximum acceleration [g]
    w = 2 * np.pi / T

    # Define excitation function (base horizontal acceleration)
    def excitation(t):
        return -A_max * np.sin(w * t)

    # Set damping properties
    St.set_damping_properties(xsi=0.0, damp_type='RAYLEIGH')

    print("\nRunning nonlinear dynamic analysis (CDM)...")
    print(f"  Duration: 0.6 s")
    print(f"  Time step: 0.001 s")
    print(f"  Excitation period: {T} s")
    print(f"  Max acceleration: {A_max} g")

    St = Dynamic.solve_nonlinear(
        St, 0.6, 1e-3,
        lmbda=excitation,
        Meth='CDM',
        filename='Arch_Dynamic',
        dir_name=save_path
    )

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=1, show_contact_faces=False,
            title="Arch Stroobants - Deformed (Dynamic)")

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("Arch Stroobants Analysis Complete")
    print("="*60)
    print(f"Arch dimensions:")
    print(f"  Mean radius: {R_med * 1000:.0f} mm")
    print(f"  Thickness: {t * 1000:.0f} mm")
    print(f"  Number of voussoirs: 16")
    print(f"Analysis types:")
    print(f"  - Static force control")
    print(f"  - Modal analysis ({nb_modes} modes)")
    print(f"  - Nonlinear dynamic (harmonic excitation)")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
