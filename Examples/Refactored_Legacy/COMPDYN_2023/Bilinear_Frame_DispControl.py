# -*- coding: utf-8 -*-
"""
Bilinear Frame - Displacement Control - COMPDYN 2023
=====================================================

Refactored from: Legacy/Examples/COMPDYN_2023/Bilinear_Frame/Bilinear_Frame_DispControl.py

This example demonstrates pushover analysis of a portal frame with bilinear
material behavior. The frame can be modeled as either:
- Hybrid: FE columns + block beam (coupled)
- Full blocks: All members as block assemblies

Structure Type: Either Hybrid or Pure DFEM
Analysis: Nonlinear static (displacement control)

Configuration:
- Frame: 3m x 3m portal
- Columns: H_C = 0.2 * 2^(1/3) m depth (Timoshenko or blocks)
- Beam: H_B = 0.2m depth (always blocks with bilinear material)
- Bilinear material: E=30 GPa, fy=20 MPa
- P-Delta effects included (geometric nonlinearity)
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
from Core import Structure_Block, StaticNonLinear, Hybrid
from Core.Structures import BeamBlock
from Core.Objects.FEM.Timoshenko import Timoshenko, GeometryBeam
from Core.Objects.ConstitutiveLaw.Material import Material, TimoshenkoMaterial, Bilinear_Mat


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
    # Node coordinates
    N1 = np.array([0, 0], dtype=float)   # Bottom left
    N2 = np.array([0, 3], dtype=float)   # Top left
    N3 = np.array([3, 3], dtype=float)   # Top right
    N4 = np.array([3, 0], dtype=float)   # Bottom right

    # Cross-section dimensions
    H_B = 0.2                    # Beam depth [m]
    H_C = 0.2 * 2 ** (1 / 3)     # Column depth [m] (slightly larger)
    B = 0.2                      # Section width [m]

    # Discretization
    BLOCKS = 40       # Number of blocks per member
    CPS = 30          # Contact pairs per interface

    # Material properties
    E = 30e9          # Young's modulus [Pa]
    NU = 0.0          # Poisson's ratio
    FY = 20e6         # Yield stress [Pa]
    A = 0.0           # Hardening ratio

    # Analysis options
    Lin_Geom = False  # False = P-Delta effects (geometric nonlinearity)
    FEs = False       # True = FE columns (hybrid), False = block columns (full)

    # Filename suffix
    text_lin_geom = 'Linear' if Lin_Geom else 'P-Delta'
    text_fes = '_Coupled' if FEs else '_Full'
    filename = f'Frame_BilinMat_{text_lin_geom}{text_fes}'

    print(f"\nFrame Configuration:")
    print(f"  Geometry: {'Linear' if Lin_Geom else 'P-Delta (geometric nonlinearity)'}")
    print(f"  Columns: {'FE (Timoshenko)' if FEs else 'Blocks (bilinear material)'}")

    # ==========================================================================
    # Create Structure
    # ==========================================================================
    if FEs:
        # Hybrid structure: FE columns + block beam
        St = Hybrid()

        # Create Timoshenko material and geometry for columns
        fe_mat = TimoshenkoMaterial(E, NU, rho=0)
        fe_geom = GeometryBeam(h=H_C, b=B)

        # Add FE columns (left and right)
        fe_left = Timoshenko([N1, N2], fe_mat, fe_geom, lin_geom=Lin_Geom)
        fe_right = Timoshenko([N4, N3], fe_mat, fe_geom, lin_geom=Lin_Geom)
        St.add_fe(fe_left)
        St.add_fe(fe_right)

        # Add block beam with bilinear material
        beam_mat = Bilinear_Mat(E, NU, FY, A)
        Beam = BeamBlock(N2, N3, BLOCKS, H_B, rho=100., b=B, material=beam_mat)
        for block in Beam.list_blocks:
            St.list_blocks.append(block)

        # Make nodes (hybrid structure)
        St.make_nodes()

        # Create contact faces for beam only
        St.make_cfs(Lin_Geom, nb_cps=CPS)

        # Get control node ID
        control_node = St.get_node_id(N2)

    else:
        # Full block structure: all members as blocks
        St = Structure_Block()

        # Bilinear material for all members
        bilin_mat = Bilinear_Mat(E, NU, FY, A)

        # Add left column (blocks)
        Left_Col = BeamBlock(N1, N2, BLOCKS, H_C, rho=100., b=B, material=bilin_mat)
        for block in Left_Col.list_blocks:
            St.list_blocks.append(block)

        # Add right column (blocks) - note: N3->N4 direction
        Right_Col = BeamBlock(N4, N3, BLOCKS, H_C, rho=100., b=B, material=bilin_mat)
        for block in Right_Col.list_blocks:
            St.list_blocks.append(block)

        # Add beam (blocks)
        Beam = BeamBlock(N2, N3, BLOCKS, H_B, rho=100., b=B, material=bilin_mat)
        for block in Beam.list_blocks:
            St.list_blocks.append(block)

        # Make nodes
        St.make_nodes()

        # Create contact faces
        St.make_cfs(Lin_Geom, nb_cps=CPS)

        # Control node is the last block of left column
        control_node = BLOCKS - 1

    # ==========================================================================
    # Boundary Conditions and Loading
    # ==========================================================================
    F = 100e3  # Reference force [N]

    # Apply horizontal load at top left
    St.load_node(N2, [0], F)

    # Apply vertical loads (gravity) at top nodes - fixed loads
    St.load_node(N2, [1], -F, fixed=True)
    St.load_node(N3, [1], -F, fixed=True)

    # Fix supports (pin connections at base)
    St.fix_node(N1, [0, 1])
    St.fix_node(N4, [0, 1])

    # Plot initial structure
    St.plot(show_contact_faces=False, title=f"Bilinear Frame - Initial ({text_fes[1:]})")

    # ==========================================================================
    # Displacement Control Analysis
    # ==========================================================================
    max_disp = 65e-3  # Maximum displacement [m]
    n_steps = 65      # Number of steps

    print(f"\nRunning pushover analysis...")
    print(f"  Control node: {control_node}")
    print(f"  Control DOF: 0 (horizontal)")
    print(f"  Max displacement: {max_disp * 1000} mm")
    print(f"  Number of steps: {n_steps}")

    St = StaticNonLinear.solve_dispcontrol(
        St, n_steps, max_disp, control_node, 0,
        dir_name=save_path,
        filename=filename,
        tol=1.0
    )

    # ==========================================================================
    # Plot Results
    # ==========================================================================
    St.plot(show_deformed=True, deformation_scale=10, show_contact_faces=False,
            title=f"Bilinear Frame - Deformed ({text_fes[1:]})")
    plt.savefig(os.path.join(save_path, f'{filename}_deformed.png'), dpi=150)

    # Save structure
    St.save_structure(os.path.join(save_path, filename))

    # ==========================================================================
    # Results Summary
    # ==========================================================================
    print("\n" + "="*60)
    print(f"Pushover Analysis Results - Bilinear Frame ({text_fes[1:]})")
    print("="*60)
    print(f"Frame dimensions: 3m x 3m")
    print(f"Column depth: {H_C:.4f}m")
    print(f"Beam depth: {H_B}m")
    print(f"Material: Bilinear (E={E/1e9}GPa, fy={FY/1e6}MPa)")
    print(f"Geometry: {text_lin_geom}")
    print(f"Max displacement: {max_disp * 1000}mm")
    print(f"Results saved to: {save_path}")
    print("="*60)

    plt.show()

    return St


if __name__ == "__main__":
    St = main()
