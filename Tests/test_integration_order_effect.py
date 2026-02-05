"""
Test: Effect of Integration Order on Stress Continuity at Mortar Interfaces

This test compares stress oscillations at hybrid interfaces using different
integration orders (2 vs 3) to determine if insufficient integration accuracy
is the cause of stress discontinuities with quadratic elements.

Theory:
- Mortar coupling requires numerical integration of shape function products
- For quadratic elements (T6, Q9), the integrand N_i * N_j is quartic (degree 4)
- 2-point Gauss rule: exact for polynomials up to degree 3 (cubic)
- 3-point Gauss rule: exact for polynomials up to degree 5 (quintic)

Expected outcome:
- If integration order is the issue: order=3 should show fewer oscillations
- If it's an inherent limitation: both orders will show similar patterns
"""
import sys
from pathlib import Path

# Path setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.FEM.Mesh import Mesh
from Core.Structures.Structure_Hybrid import Hybrid
from Core.Structures.Structure_FEM import Structure_FEM
from Core.Objects.DFEM.Block import Block_2D


def create_hybrid_structure(element_order: int, integration_order: int, mesh_size: float = 0.1):
    """Create a simple hybrid column with block on bottom, FEM on top."""

    # Material and geometry
    E = 30e9  # Pa (concrete)
    nu = 0.2
    thickness = 0.1  # m
    material = PlaneStress(E=E, nu=nu)
    geometry = Geometry2D(t=thickness)

    # Dimensions
    block_height = 1.0  # m
    fem_height = 1.0  # m
    width = 0.5  # m

    # Create hybrid structure
    St = Hybrid()

    # Block (bottom)
    vertices = [
        [0.0, 0.0],
        [width, 0.0],
        [width, block_height],
        [0.0, block_height]
    ]
    block = Block_2D(vertices=vertices, material=material, t=thickness)
    St.add_block(block)

    # FEM mesh (top) - extend slightly beyond interface for proper coupling
    overlap = mesh_size * 0.1  # Small overlap for interface detection
    mesh = Mesh.generate(
        width=width,
        height=fem_height,
        mesh_size=mesh_size,
        element_type="triangle",
        order=element_order,
        origin=(0.0, block_height - overlap)
    )

    # Create FEM portion
    St_fem = Structure_FEM.from_mesh(mesh, material, geometry, verbose=False)

    # Add FEM elements to hybrid
    for fe in St_fem.list_fes:
        St.add_fe(fe)

    # Build nodes
    St.make_nodes()

    # Enable mortar coupling with specified integration order
    St.enable_block_fem_coupling(
        method='mortar',
        integration_order=integration_order,
        interface_tolerance=1e-4,
        interface_orientation='horizontal'
    )

    return St


def apply_loading_and_solve(St):
    """Apply uniform compression and solve."""

    # Find top nodes (FEM)
    n_blocks = len(St.list_blocks)
    max_y = max(node[1] for node in St.list_nodes[n_blocks:])

    top_nodes = []
    for i, node in enumerate(St.list_nodes):
        if i >= n_blocks and abs(node[1] - max_y) < 1e-6:
            top_nodes.append(i)

    # Find bottom nodes (block corners)
    block = St.list_blocks[0]
    min_y = min(v[1] for v in block.vertices)
    bottom_nodes = []
    for i, node in enumerate(St.list_nodes):
        if i < n_blocks and abs(node[1] - min_y) < 1e-6:
            bottom_nodes.append(i)

    # Fix bottom (block base)
    for node_id in bottom_nodes:
        St.fix_node(node_id, 'xy')

    # Also fix rotation of block
    St.fix_node(0, 'z')  # Block centroid rotation

    # Apply uniform downward load at top
    total_force = -100000  # N (100 kN compression)
    force_per_node = total_force / len(top_nodes)

    for node_id in top_nodes:
        St.add_nodal_load(node_id, np.array([0.0, force_per_node]))

    # Solve
    St = Static.solve_linear_saddle_point(St)

    return St


def analyze_interface_stresses(St, element_order):
    """Analyze stress distribution at the interface."""

    n_blocks = len(St.list_blocks)
    block = St.list_blocks[0]

    # Find interface y-coordinate
    interface_y = max(v[1] for v in block.vertices)

    # Find FEM elements near the interface
    interface_elements = []
    for i, fe in enumerate(St.list_fes):
        element_nodes = fe.nodes
        element_y_coords = [n[1] for n in element_nodes]
        if min(element_y_coords) <= interface_y <= max(element_y_coords):
            interface_elements.append(i)

    if not interface_elements:
        return None, None, None

    # Compute stresses at interface
    interface_stresses = []

    for elem_id in interface_elements:
        fe = St.list_fes[elem_id]

        # Get element DOFs
        elem_dofs = []
        for local_idx in range(len(fe.nodes)):
            global_node_id = fe.get_global_node_id(local_idx)
            if global_node_id is not None:
                dofs = St.get_dofs_from_node(global_node_id)
                elem_dofs.extend(dofs[:2])  # Only ux, uy

        if len(elem_dofs) == 0:
            continue

        # Get displacements
        u_elem = St.u[elem_dofs]

        # Compute stresses at element nodes
        try:
            stresses = fe.compute_stress_at_nodes(u_elem)
            interface_stresses.append(stresses)
        except Exception:
            continue

    if not interface_stresses:
        return None, None, None

    # Collect sigma_yy values (vertical stress)
    sigma_yy_values = []
    for stress_array in interface_stresses:
        for node_stress in stress_array:
            if len(node_stress) >= 2:
                sigma_yy_values.append(node_stress[1])  # sigma_yy

    if not sigma_yy_values:
        return None, None, None

    sigma_yy_values = np.array(sigma_yy_values)

    # Compute statistics
    mean_stress = np.mean(sigma_yy_values)
    std_stress = np.std(sigma_yy_values)
    coeff_var = abs(std_stress / mean_stress * 100) if abs(mean_stress) > 1e-10 else 0.0

    # Count sign changes (oscillation indicator)
    sign_changes = np.sum(np.diff(np.sign(sigma_yy_values)) != 0)
    oscillation_ratio = sign_changes / max(len(sigma_yy_values) - 1, 1) * 100

    return coeff_var, oscillation_ratio, sigma_yy_values


def main():
    print("=" * 70)
    print("INTEGRATION ORDER EFFECT ON MORTAR STRESS CONTINUITY")
    print("=" * 70)
    print()
    print("Testing hypothesis: Higher integration order reduces stress oscillations")
    print("for quadratic elements at mortar interfaces.")
    print()

    results = {}

    for elem_order in [1, 2]:  # Linear (T3) and Quadratic (T6)
        elem_name = "T3 (Linear)" if elem_order == 1 else "T6 (Quadratic)"
        print(f"\n{'=' * 70}")
        print(f"Element Type: {elem_name}")
        print(f"{'=' * 70}")

        results[elem_order] = {}

        for int_order in [2, 3]:  # Test both integration orders
            print(f"\n--- Integration Order: {int_order} ---")

            try:
                # Create and solve
                St = create_hybrid_structure(
                    element_order=elem_order,
                    integration_order=int_order,
                    mesh_size=0.08
                )
                St = apply_loading_and_solve(St)

                # Analyze
                coeff_var, osc_ratio, stresses = analyze_interface_stresses(St, elem_order)

                if coeff_var is not None:
                    print(f"  Coefficient of Variation: {coeff_var:.1f}%")
                    print(f"  Oscillation Ratio: {osc_ratio:.1f}%")
                    print(f"  Stress Range: [{np.min(stresses):.2e}, {np.max(stresses):.2e}] Pa")

                    results[elem_order][int_order] = {
                        'coeff_var': coeff_var,
                        'osc_ratio': osc_ratio,
                        'stress_range': (np.min(stresses), np.max(stresses))
                    }
                else:
                    print("  Could not compute interface stresses")

            except Exception as e:
                print(f"  Error: {str(e)}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Integration Order Effect")
    print("=" * 70)
    print()
    print(f"{'Element':<12} {'Int. Order':<12} {'CoV (%)':<12} {'Oscillation (%)':<15}")
    print("-" * 50)

    for elem_order in [1, 2]:
        elem_name = "T3" if elem_order == 1 else "T6"
        for int_order in [2, 3]:
            if int_order in results.get(elem_order, {}):
                r = results[elem_order][int_order]
                print(f"{elem_name:<12} {int_order:<12} {r['coeff_var']:<12.1f} {r['osc_ratio']:<15.1f}")

    # Analysis
    print()
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if 2 in results and 3 in results[2] and 2 in results[2]:
        t6_order2 = results[2].get(2, {})
        t6_order3 = results[2].get(3, {})

        if t6_order2 and t6_order3:
            cov_improvement = t6_order2['coeff_var'] - t6_order3['coeff_var']
            osc_improvement = t6_order2['osc_ratio'] - t6_order3['osc_ratio']

            print(f"\nT6 Element Comparison (Order 2 -> 3):")
            print(f"  CoV change: {cov_improvement:+.1f}% (positive = improvement)")
            print(f"  Oscillation change: {osc_improvement:+.1f}% (positive = improvement)")

            if cov_improvement > 10 or osc_improvement > 10:
                print("\n  CONCLUSION: Higher integration order HELPS significantly.")
                print("  RECOMMENDATION: Use integration_order=3 for quadratic elements.")
            elif cov_improvement > 0 or osc_improvement > 0:
                print("\n  CONCLUSION: Higher integration order provides MINOR improvement.")
                print("  The oscillations may have other contributing factors.")
            else:
                print("\n  CONCLUSION: Integration order is NOT the primary cause.")
                print("  Oscillations likely due to element shape function limitations")
                print("  or Lagrange multiplier space choice (LBB/inf-sup condition).")


if __name__ == "__main__":
    main()
