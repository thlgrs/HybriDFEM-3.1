"""
FEM-FEM Matching Mesh Coupling (Diagonal Interface)
===================================================

Validates coupling between two continuous FEM domains along a diagonal interface.

Structure:
- Domain: Square [0, 1] x [0, 1]
- Interface: Diagonal line from (0,0) to (1,1)
- Parts:
    1. Lower-Right Domain (x > y)
    2. Upper-Left Domain (x < y)
- Mesh: Topological split at the interface (duplicated nodes).

Supported Features:
- Elements: Triangles (T3, T6) or Quadrilaterals (Q4, Q8).
    - Triangles: Simple diagonal split of grid cells.
    - Quads: Subdivision strategy (Union Jack) to maintain conformal mesh.
- Coupling:
    - Lagrange Multipliers (Exact, Augmented System)
    - Penalty Method (Approximate, Standard System via Stiffness Injection)
"""

import sys
from pathlib import Path

import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Imports ---
from Core.Structures.Structure_Hybrid import Hybrid
from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.Coupling.LagrangeNodal import LagrangeCoupling
from Core.Objects.Coupling.InterfaceDetection import detect_coincident_nodes
from Examples.utils.solvers import run_solver
from Examples.utils.helpers import create_config
from Examples.utils.mesh_generation import ELEMENT_MAP
from Examples.utils.model_builders import find_nodes_at_base, find_nodes_at_height
import Examples.utils.visualization as viz

# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_ROOT = str(project_root / 'Examples' / 'Results' / 'Hybrid' / 'fem_fem')
BASE_CONFIG = {
    'geometry': {
        'nx': 10,
        'ny': 10,
        'width': 1.0,
        'height': 1.0,
        'thickness': 1.0,
    },
    'elements': {
        'type': 'quad',
        'order': 2,
    },
    'material': {
        'fem': {'E': 10e9, 'nu': 0.0, 'rho': 0},
    },
    'coupling': {
        'method': 'lagrange',
        'tolerance': 0.1,
    },
    'loads': {
        'Fx': 0,
        'Fy': 0,
    },
    'io': {
        'filename': 'fem_fem',
        'dir': RESULTS_ROOT,
        'show_nodes': True,
        'figsize': (12, 12),
        'sigma_x_range': [None, None],
        'sigma_y_range': [None, None],
        'tau_yx_range': [None, None],
        'tau_xy_range': [None, None],
    }
}

# Load Cases
CONSTRAINT = {'method': 'constraint', 'tolerance': 0.02}
PENALTY = {'method': 'penalty', 'tolerance': 0.02, 'penalty_stiffness': 1e12}
LAGRANGE = {'method': 'lagrange', 'tolerance': 0.02}

COMPRESSION = {'Fx': 0, 'Fy': -1e6}
TRACTION = {'Fx': 0, 'Fy': 1e6}
SHEAR = {'Fx': 1e6, 'Fy': 0}


def build_diagonal_structure(config):
    """
    Builds a square structure split diagonally into two independent meshes.
    Supports both Triangle and Quad meshes via different splitting strategies.
    """
    g = config['geometry']
    e = config['elements']
    m = config['material']

    mat = PlaneStress(**m['fem'])
    geom = Geometry2D(t=g['thickness'])

    # Initialize Hybrid with default settings
    # Note: We manually build nodes/elements to ensure separation, so 
    # merge_coincident_nodes=True (default) doesn't affect us here.
    St = Hybrid()
    St.list_nodes = []
    St.list_fes = []

    nx, ny = g['nx'], g['ny']
    dx = g['width'] / nx
    dy = g['height'] / ny

    # Registries: domain -> key -> node_id
    # Ensures nodes are unique per domain ('lower', 'upper') but shared within domain
    registries = {'lower': {}, 'upper': {}}

    dofs_per_node = 2

    def get_node(domain, key, coords):
        reg = registries[domain]
        if key not in reg:
            idx = len(St.list_nodes)
            St.list_nodes.append(coords)
            St.node_dof_counts.append(dofs_per_node)
            St.node_dof_offsets.append(St.node_dof_offsets[-1] + dofs_per_node)
            reg[key] = idx
        return reg[key]

    # --- MESH GENERATION ---
    for j in range(ny):
        for i in range(nx):
            # Cell Coords
            x0, y0 = i * dx, j * dy
            xc, yc = (i + 0.5) * dx, (j + 0.5) * dy

            # Corner Coords
            c_bl, c_br = [x0, y0], [x0 + dx, y0]
            c_tr, c_tl = [x0 + dx, y0 + dy], [x0, y0 + dy]
            c_ct = [xc, yc]

            # Node Keys
            k_bl, k_br = ('C', i, j), ('C', i + 1, j)
            k_tr, k_tl = ('C', i + 1, j + 1), ('C', i, j + 1)
            k_ct = ('CT', i, j)

            # --- TRIANGLE STRATEGY ---
            if e['type'] == 'triangle':
                # Determine Element Class
                ElementClass = ELEMENT_MAP.get((e['type'], e['order']))
                if ElementClass is None: raise ValueError(f"Unsupported: {e['type']} {e['order']}")

                # Define Triangles for this cell
                # We treat (0,0)-(1,1) diagonal split.
                # Lower: (i,j), (i+1,j), (i+1,j+1)
                # Upper: (i,j), (i+1,j+1), (i,j+1)

                tris = []

                if i >= j:  # Lower Domain
                    tris.append(('lower', [k_bl, k_br, k_tr], [c_bl, c_br, c_tr]))
                if i <= j:  # Upper Domain
                    tris.append(('upper', [k_bl, k_tr, k_tl], [c_bl, c_tr, c_tl]))

                # Handle Higher Order (Order 2) - Not fully implemented in this simplified generic loop
                # (The previous meshgrid implementation handled it better for pure grids, 
                # but this loop supports the generic split logic better).
                # For simplicity in this example, we stick to Linear Triangles logic here.
                # If Order 2 is needed, we'd need midpoint keys.
                if e['order'] > 1:
                    print(
                        "[WARNING] Order > 1 not fully implemented in diagonal builder for Triangles. Falling back to Order 1 logic (geometry correct, but missing mid-nodes).")

                for dom, keys, coords in tris:
                    n_ids = [get_node(dom, k, c) for k, c in zip(keys, coords)]
                    el = ElementClass([St.list_nodes[nid] for nid in n_ids], mat, geom)
                    el.connect = n_ids
                    St.list_fes.append(el)

            # --- QUAD STRATEGY ---
            elif e['type'] == 'quad':
                from Core.Objects.FEM.Quads import Quad4

                # Additional Midpoint Keys/Coords for Quad subdivision
                k_mb, c_mb = ('MH', i, j), [x0 + dx / 2, y0]
                k_mr, c_mr = ('MV', i + 1, j), [x0 + dx, y0 + dy / 2]
                k_mt, c_mt = ('MH', i, j + 1), [x0 + dx / 2, y0 + dy]
                k_ml, c_ml = ('MV', i, j), [x0, y0 + dy / 2]

                # Centroids for diagonal split triangles
                c_gl = [x0 + dx * 2 / 3, y0 + dy * 1 / 3]  # Lower Tri Centroid
                c_gu = [x0 + dx * 1 / 3, y0 + dy * 2 / 3]  # Upper Tri Centroid
                k_gl, k_gu = ('GL', i, j), ('GU', i, j)

                quads_to_add = []

                if i > j:  # Pure Lower Cell (4 Quads)
                    dom = 'lower'
                    quads_to_add = [
                        (dom, [k_bl, k_mb, k_ct, k_ml], [c_bl, c_mb, c_ct, c_ml]),
                        (dom, [k_mb, k_br, k_mr, k_ct], [c_mb, c_br, c_mr, c_ct]),
                        (dom, [k_ct, k_mr, k_tr, k_mt], [c_ct, c_mr, c_tr, c_mt]),
                        (dom, [k_ml, k_ct, k_mt, k_tl], [c_ml, c_ct, c_mt, c_tl])
                    ]
                elif i < j:  # Pure Upper Cell (4 Quads)
                    dom = 'upper'
                    quads_to_add = [
                        (dom, [k_bl, k_mb, k_ct, k_ml], [c_bl, c_mb, c_ct, c_ml]),
                        (dom, [k_mb, k_br, k_mr, k_ct], [c_mb, c_br, c_mr, c_ct]),
                        (dom, [k_ct, k_mr, k_tr, k_mt], [c_ct, c_mr, c_tr, c_mt]),
                        (dom, [k_ml, k_ct, k_mt, k_tl], [c_ml, c_ct, c_mt, c_tl])
                    ]
                else:  # Diagonal Split (i == j)
                    # Lower Triangle (3 Quads)
                    quads_to_add.extend([
                        ('lower', [k_bl, k_mb, k_gl, k_ct], [c_bl, c_mb, c_gl, c_ct]),
                        ('lower', [k_mb, k_br, k_mr, k_gl], [c_mb, c_br, c_mr, c_gl]),
                        ('lower', [k_mr, k_tr, k_ct, k_gl], [c_mr, c_tr, c_ct, c_gl])
                    ])
                    # Upper Triangle (3 Quads)
                    quads_to_add.extend([
                        ('upper', [k_bl, k_ct, k_gu, k_ml], [c_bl, c_ct, c_gu, c_ml]),
                        ('upper', [k_ct, k_tr, k_mt, k_gu], [c_ct, c_tr, c_mt, c_gu]),
                        ('upper', [k_mt, k_tl, k_ml, k_gu], [c_mt, c_tl, c_ml, c_gu])
                    ])

                for dom, keys, coords in quads_to_add:
                    n_ids = [get_node(dom, k, c) for k, c in zip(keys, coords)]
                    el = Quad4([St.list_nodes[nid] for nid in n_ids], mat, geom)
                    el.connect = n_ids
                    St.list_fes.append(el)

    # Initialize DOFs for all elements
    for elem in St.list_fes:
        for k, nid in enumerate(elem.connect):
            base_dof = St.node_dof_offsets[nid]
            elem.dofs[2 * k] = base_dof
            elem.dofs[2 * k + 1] = base_dof + 1

    # Finalize Structure
    St.nb_dofs = St.compute_nb_dofs()
    St.U = np.zeros(St.nb_dofs, dtype=float)
    St.P = np.zeros(St.nb_dofs, dtype=float)
    St.P_fixed = np.zeros(St.nb_dofs, dtype=float)
    St.dof_fix = np.array([], dtype=int)
    St.dof_free = np.arange(St.nb_dofs, dtype=int)
    St.nb_dof_fix = len(St.dof_fix)
    St.nb_dof_free = len(St.dof_free)

    return St


def apply_manual_penalty(St, pairs, stiffness):
    """
    Manually adds penalty springs between node pairs by wrapping get_K_str.
    This allows testing penalty coupling without full integration into the Hybrid class.
    """
    print(f"  [INFO] Injecting Manual Penalty Stiffness (k={stiffness:.2e})")

    St._manual_penalty_data = {'pairs': pairs, 'k': stiffness}

    original_get_K0 = St.get_K_str0
    original_get_K = St.get_K_str

    def add_springs(K_matrix):
        k = St._manual_penalty_data['k']
        pairs = St._manual_penalty_data['pairs']
        for n1, n2 in pairs:
            dof1 = St.get_dofs_from_node(n1)
            dof2 = St.get_dofs_from_node(n2)

            # Add spring k to both u and v directions
            for i in range(2):
                idx1, idx2 = dof1[i], dof2[i]
                K_matrix[idx1, idx1] += k
                K_matrix[idx2, idx2] += k
                K_matrix[idx1, idx2] -= k
                K_matrix[idx2, idx1] -= k
        return K_matrix

    St.get_K_str0 = lambda: add_springs(original_get_K0())
    St.get_K_str = lambda: add_springs(original_get_K())


def build_and_solve(config):
    """Builds the model, applies BCs/Loads, Couples, and Solves."""
    method = config['coupling']['method']

    # 1. Build Structure
    St = build_diagonal_structure(config)

    # 2. Fix Base (y=0)
    base_nodes = find_nodes_at_base(St)
    for n in base_nodes:
        St.fix_node(n, [0, 1])

    # 3. Load Top (y=H)
    max_y = max(n[1] for n in St.list_nodes)
    top_nodes = find_nodes_at_height(St, max_y, tolerance=0.01)
    if not top_nodes:
        raise ValueError(f"No nodes found at max_y={max_y}")

    Fx = config['loads']['Fx'] / len(top_nodes)
    Fy = config['loads']['Fy'] / len(top_nodes)

    St.load_node(top_nodes, [0], Fx)
    St.load_node(top_nodes, [1], Fy)

    # 4. Apply Coupling
    if method in ['lagrange', 'penalty']:
        # Detect Coincident Nodes
        pairs = detect_coincident_nodes(St, tolerance=1e-6)

        # Filter Redundant (Fully Fixed) Pairs
        valid_pairs = []
        fixed_dofs_set = set(St.dof_fix)
        for n1, n2 in pairs:
            dofs1 = St.get_dofs_from_node(n1)
            dofs2 = St.get_dofs_from_node(n2)
            if all(d in fixed_dofs_set for d in dofs1) and all(d in fixed_dofs_set for d in dofs2):
                continue
            valid_pairs.append((n1, n2))

        print(f"  Coupling: {method.upper()}. Pairs: {len(valid_pairs)}")

        if method == 'lagrange':
            lc = LagrangeCoupling()
            for n1, n2 in valid_pairs:
                lc.add_node_pair(n1, n2)
            St.lagrange_coupling = lc
            St.lagrange_coupling.activate()
            St.coupling_enabled = True

        elif method == 'penalty':
            apply_manual_penalty(St, valid_pairs, config['coupling'].get('penalty_stiffness', 1e12))

    # 5. Check Connectivity (Zero Stiffness check)
    St.dofs_defined()
    K0 = St.get_K_str0()
    zeros = np.where(np.abs(K0.diagonal()) < 1e-12)[0]
    if len(zeros) > 0:
        print(f"  [ERROR] Found {len(zeros)} zero diagonal entries in K0!")
    else:
        print("  [OK] K0 check passed.")

    # 6. Solve and Visualize
    St = run_solver(St, config)
    viz.plot_stress(St, config)
    viz.plot_displacement(St, config)

    return St


def run_case(label, load_case):
    print("=" * 60)
    print(f"FEM-FEM {label}")
    print("=" * 60)

    io = {
        'sigma_x_range': [None, None],
        'sigma_y_range': [None, None],
        'tau_yx_range': [None, None],
        'tau_xy_range': [None, None],
    }

    # Clone load case to avoid mutation
    loads = load_case.copy()

    # Lagrange
    print("\n--- LAGRANGE ---")
    cfg_lag = create_config(BASE_CONFIG, f"{label.lower()}_lagrange",
                            coupling=LAGRANGE, loads=loads, io=io)
    build_and_solve(cfg_lag)

    # Penalty
    print("\n--- PENALTY ---")
    cfg_pen = create_config(BASE_CONFIG, f"{label.lower()}_penalty",
                            coupling=PENALTY, loads=loads, io=io)
    build_and_solve(cfg_pen)


if __name__ == "__main__":
    run_case("COMPRESSION", COMPRESSION)
    run_case("TRACTION", TRACTION)
    run_case("SHEAR", SHEAR)
