"""
FEM-FEM Mortar Coupling with Diagonal Interface
================================================
Demonstrates coupling two non-matching FEM meshes along a diagonal interface
using the Mortar method with Core classes.

Scenario:
- Square domain [0,1] x [0,1] split diagonally (y = x)
- Lower-Right Domain: Fine mesh
- Upper-Left Domain: Coarse mesh
- Coupling: Mortar method using Core.MortarCoupling

"""

import os
import sys
from pathlib import Path

import numpy as np

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Core Imports ---
from Core.Structures.Structure_Hybrid import Hybrid
from Core.Structures.Structure_FEM import Structure_FEM
from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.Coupling.Mortar import MortarCoupling
from Core.Objects.Coupling.InterfaceDetectionFem import detect_fem_fem_interfaces
from Core.Objects.FEM.Mesh import Mesh

# --- Utils Imports ---
from Examples.utils.visualization import visualize
from Examples.utils.solvers import run_solver
from Examples.utils.helpers import create_config

# =============================================================================
# CONFIGURATION
# =============================================================================
RESULTS_ROOT = str(project_root / 'Examples' / 'Results' / 'Hybrid' / 'fem_fem')

BASE_CONFIG = {
    'geometry': {
        'domain_size': 1.0,
        'element_size_up': 0.07,
        'element_size_down': 0.14,
        'thickness': 1.0,
    },
    'elements': {
        'type': 'quad',
        'order': 1
    },
    'material': {'E': 10e9, 'nu': 0.0, 'rho': 0},
    'coupling': {
        'method': 'mortar',
        'tolerance': 5e-3,  # Interface detection tolerance
        'integration_order': 2,
    },
    'loads': {
        'Fx': 0,
        'Fy': 0,
    },
    'io': {
        'filename': '',
        'dir': RESULTS_ROOT,
        'scale': 50.0,
        'figsize': (12, 12),
        'show_nodes': False,
    }
}


# =============================================================================
# MODEL GENERATION
# =============================================================================

def generate_meshes(config):
    """
    Generate two triangular meshes using Core.Objects.FEM.Mesh (Gmsh).

    Creates:
    - Lower-right triangle: (0,0) -> (1,0) -> (1,1)
    - Upper-left triangle: (0,0) -> (1,1) -> (0,1)
    """
    g = config['geometry']
    L = g['domain_size']
    element = config['elements']

    tol = 1e-3

    # Lower-Right
    points_lower = [(0, 0), (L, 0), (L, L - 0.1 * L - tol), (0, 0.1 * L - tol)]
    edge_groups_lower = {
        "bottom": [0],
        "right": [1],
        "diagonal": [2],
        "left": [3]
    }

    mesh_lower = Mesh(
        points=points_lower,
        element_type=element['type'],
        element_size=g['element_size_down'],
        order=element['order'],
        name="lower_mesh",
        edge_groups=edge_groups_lower,
    )

    # Upper-Left
    points_upper = [(0, 0.1 * L + tol), (L, L - 0.1 * L + tol), (L, L), (0, L)]
    edge_groups_upper = {
        "diagonal": [0],
        "right": [1],
        "top": [2],
        "left": [3]
    }

    mesh_upper = Mesh(
        points=points_upper,
        element_type=element['type'],
        element_size=g['element_size_up'],
        order=element['order'],
        name="upper_mesh",
        edge_groups=edge_groups_upper,
    )

    print("  Generating meshes with Gmsh...")
    mesh_lower.generate_mesh()
    mesh_upper.generate_mesh()

    print(f"    Lower mesh: {len(mesh_lower.nodes())} nodes, {len(mesh_lower.elements())} elements")
    print(f"    Upper mesh: {len(mesh_upper.nodes())} nodes, {len(mesh_upper.elements())} elements")

    return mesh_lower, mesh_upper


def setup_mortar_coupling(St, config):
    """
    Set up mortar coupling using Core.Objects.Coupling classes.
    """
    cp = config['coupling']

    # Detect FEM-FEM interfaces using Core function
    interfaces = detect_fem_fem_interfaces(
        St,
        tolerance=cp['tolerance']
    )

    if not interfaces:
        raise ValueError("No interfaces detected! Check mesh alignment.")

    print(f"  Detected {len(interfaces)} raw interface segments")

    # For each interface, detect slave edges
    for interface in interfaces:
        interface.detect_slave_edges(St, tolerance=cp['tolerance'])

    # Filter interfaces that have slave edges
    valid_interfaces = [iface for iface in interfaces if iface.slave_edges]
    print(f"  Valid interfaces with slave edges: {len(valid_interfaces)}")

    if not valid_interfaces:
        raise ValueError("No valid interfaces with slave edges found!")

    # Create MortarCoupling using Core class
    coupling = MortarCoupling(
        integration_order=cp['integration_order'],
        interface_tolerance=cp['tolerance'],
        interface_orientation=None,
    )

    # Assign interfaces
    coupling.interfaces = valid_interfaces
    coupling.active = True

    # Build constraint matrix
    coupling.build_constraint_matrix(St)

    # Attach to structure
    St.mortar_coupling = coupling

    return coupling


def create_model(config):
    """
    Create the complete hybrid structure model.
    Generates meshes, builds structure, and sets up coupling.
    """
    print("\n[1] Generating Model...")

    # 1. Generate meshes
    mesh_lower, mesh_upper = generate_meshes(config)

    # 2. Build combined structure
    m = config['material']
    g = config['geometry']

    mat = PlaneStress(**m)
    geom = Geometry2D(t=g['thickness'])

    St = Hybrid(merge_coincident_nodes=True)

    # Use Structure_FEM.from_mesh for Gmsh meshes
    St_lower = Structure_FEM.from_mesh(mesh_lower, mat, geom)
    St_upper = Structure_FEM.from_mesh(mesh_upper, mat, geom)

    # Add FE elements from both meshes to the Hybrid structure
    for fe in St_lower.list_fes:
        St.add_fe(fe)
    for fe in St_upper.list_fes:
        St.add_fe(fe)

    # Build node list from elements
    St.make_nodes()

    print(f"  Total nodes: {len(St.list_nodes)}")
    print(f"  Total elements: {len(St.list_fes)}")
    print(f"  Total DOFs: {St.nb_dofs}")

    # 3. Setup mortar coupling
    setup_mortar_coupling(St, config)

    return St


# =============================================================================
# BOUNDARY CONDITIONS & ANALYSIS
# =============================================================================

def apply_conditions(St, config):
    """Apply boundary conditions: fix base, load top."""
    print("\n[2] Applying Boundary Conditions...")

    coords = np.array(St.list_nodes)
    L = config['geometry']['domain_size']

    # Fix base nodes (y = 0, lower mesh only)
    base_tolerance = 1e-6
    base_nodes = np.where(np.abs(coords[:, 1]) < base_tolerance)[0]
    St.fix_node(base_nodes.tolist(), [0, 1])
    print(f"  Fixed {len(base_nodes)} base nodes (y=0)")

    # Load top nodes (y = L, upper mesh only)
    top_tolerance = 1e-6
    top_nodes = np.where(np.abs(coords[:, 1] - L) < top_tolerance)[0]

    control_node = None
    if len(top_nodes) > 0:
        loads = config.get('loads', {})
        fx = loads.get('Fx', 0.0)
        fy = loads.get('Fy', 0.0)

        if fx != 0:
            force_per_node_x = fx / len(top_nodes)
            St.load_node(top_nodes.tolist(), [0], force_per_node_x)

        if fy != 0:
            force_per_node_y = fy / len(top_nodes)
            St.load_node(top_nodes.tolist(), [1], force_per_node_y)

        print(f"  Loaded {len(top_nodes)} top nodes (y={L}) with Fx={fx:.1e}, Fy={fy:.1e}")
        # Use the middle top node as control node
        control_node = top_nodes[len(top_nodes) // 2]
    else:
        print("  [WARNING] No top nodes found!")

    return St, control_node


def analyze_results(St, config, control_node):
    """Analyze and display results. Save to markdown."""
    io_conf = config['io']
    g_conf = config['geometry']
    m_conf = config['material']
    c_conf = config['coupling']

    lines = []

    def log(text=""):
        print(text)
        lines.append(text)

    log("\n" + "=" * 60)
    log("   RESULTS ANALYSIS")
    log("=" * 60)

    # Check constraint residual (Specific to Mortar)
    if hasattr(St, 'mortar_coupling') and St.mortar_coupling and St.mortar_coupling.constraint_matrix_G is not None:
        residual = np.linalg.norm(St.mortar_coupling.constraint_matrix_G @ St.U)
        log(f"  Constraint residual: {residual:.2e}")
        if residual < 1e-8:
            log("  [PASS] Constraints satisfied to machine precision")
        elif residual < 1e-4:
            log("  [OK] Constraints reasonably satisfied")
        else:
            log("  [WARNING] Large constraint residual!")

    # Displacement statistics
    u_max = np.max(np.abs(St.U[0::2]))  # Assuming 2D
    v_max = np.max(np.abs(St.U[1::2]))

    log(f"\n**Max Displacements:**")
    log(f"  Max |u_x|: {u_max:.6e} m")
    log(f"  Max |u_y|: {v_max:.6e} m")

    if control_node is not None:
        dofs = St.get_dofs_from_node(control_node)
        log(f"\n**Control Node {control_node}:**")
        log(f"  - ux: {St.U[dofs[0]]:.6e} m")
        log(f"  - uy: {St.U[dofs[1]]:.6e} m")

    # --- Save Markdown ---
    os.makedirs(io_conf['dir'], exist_ok=True)
    md_path = os.path.join(io_conf['dir'], io_conf['filename'] + ".md")

    md_content = [f"# Analysis Results: {io_conf['filename']}", ""]
    md_content.extend(lines[3:])

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))

    print(f"\n[OK] Results saved to: {md_path}")


def run_config(config):
    """Execute full analysis pipeline for a configuration."""
    name = config['io']['filename']
    print(f"\n{'=' * 60}")
    print(f"   Running: {name}")
    print(f"{'=' * 60}")
    St = create_model(config)
    St, control_node = apply_conditions(St, config)
    St = run_solver(St, config)
    analyze_results(St, config, control_node)
    visualize(St, config)
    return St


def run_configs(configs):
    print(f"\nRunning {len(configs)} configuration(s):")
    for config in configs:
        print(f"  - {config['io']['filename']}")
    print()
    for config in configs:
        run_config(config)
    print(f"\nCompleted {len(configs)} analyses")


def run_loadcase(configs, load):
    new_configs = []
    for config in configs:
        old_name = config['io']['filename']
        new_name = f"{load['name']}_{old_name}"
        new_configs.append(create_config(config, new_name, loads=load))
    run_configs(new_configs)


def run_loadcases(configs, loads):
    for load in loads:
        run_loadcase(configs, load)


# =============================================================================
# CONFIGURATION VARIANTS
# =============================================================================

# Define load cases
COMPRESSION = {'name': 'compression', 'Fx': 0, 'Fy': -1e6}
TRACTION = {'name': 'traction', 'Fx': 0, 'Fy': 1e6}
SHEAR = {'name': 'shear', 'Fx': 1e6, 'Fy': 0}

# --- Triangle3 (T3) - Linear Triangles ---
T3_MORTAR = create_config(BASE_CONFIG, 't3_mortar', coupling={'method': 'mortar'},
                          elements={'type': 'triangle', 'order': 1})

# --- Triangle6 (T6) - Quadratic Triangles ---
T6_MORTAR = create_config(BASE_CONFIG, 't6_mortar', coupling={'method': 'mortar'},
                          elements={'type': 'triangle', 'order': 2})

# --- Quad4 (Q4) - Bilinear Quads ---
Q4_MORTAR = create_config(BASE_CONFIG, 'q4_mortar', coupling={'method': 'mortar'},
                          elements={'type': 'quad', 'order': 1})

# --- Quad8 (Q8) - Serendipity Quads ---
Q8_MORTAR = create_config(BASE_CONFIG, 'q8_mortar', coupling={'method': 'mortar'},
                          elements={'type': 'quad', 'order': 2})

# --- Configuration Groups ---
MORTAR_CONFIGS = [T3_MORTAR, T6_MORTAR, Q4_MORTAR, Q8_MORTAR]
ALL_LOADS = [COMPRESSION, TRACTION, SHEAR]



# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    CONFIGS = MORTAR_CONFIGS
    LOADS = ALL_LOADS
    run_loadcases(CONFIGS, LOADS)
