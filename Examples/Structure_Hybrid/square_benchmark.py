"""
Square Geometry Benchmark - Coupling Methods Comparison
=========================================================

This benchmark compares all four coupling methods (Constraint, Penalty, Lagrange, Mortar)
on a square geometry with alternating horizontal block/FEM slices.

Geometry: 1m x 1m square with 20 alternating slices (block-FEM-block-...)
Benchmark cases:
- Compression: Fixed base, vertical load at top center
- Shear: Fixed base, horizontal load at top-left corner
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

from Examples.utils.visualization import plot_deformed, plot_stress, plot_initial

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Library Imports ---
from Core.Solvers.Static import StaticLinear
from Examples.utils.model_builders import (
    create_hybrid_column_slices, find_nodes_at_height, find_nodes_at_length
)
from Examples.utils.helpers import create_config


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class BenchmarkResult:
    """Stores results from a single benchmark run."""
    method: str
    n_dofs: int
    n_coupled_nodes: int

    # Timing (seconds)
    time_assembly: float = 0.0
    time_solve: float = 0.0
    time_total: float = 0.0

    # Accuracy
    tip_displacement: float = 0.0
    displacement_error: float = 0.0  # Relative to reference
    constraint_residual: float = 0.0

    # Conditioning
    condition_number: float = 0.0

    # Interface forces
    interface_force_available: bool = False
    interface_force_magnitude: float = 0.0

    # System info
    system_dimension: int = 0
    matrix_type: str = ""


@dataclass
class BenchmarkSuite:
    """Collection of results for all methods on a single benchmark."""
    name: str
    description: str
    reference_displacement: float  # Analytical or high-fidelity reference
    results: Dict[str, BenchmarkResult] = field(default_factory=dict)


# =============================================================================
# Configuration Templates
# =============================================================================

RESULTS_ROOT = str(project_root / "Examples" / 'Results' / 'Hybrid' / 'square_benchmark')

# Base SQUARE geometry configuration
SQUARE_BASE = {
    'geometry': {
        'width': 1.0,  # Square: 1m width
        'thickness': 0.25,  # Out-of-plane thickness (m)
        'n_slices': 20,  # Number of alternating slices
        'start_with': 'block',  # First slice type (at bottom)
        'block_slice_height': 0.05,  # Height of block slices (m)
        'fem_slice_height': 0.05,  # Height of FEM slices (m)

        # Mesh Refinement
        'nx': 20,  # Global horizontal refinement
        'ny_block_slice': 2,  # Block slices are 2 blocks high
        'ny_fem_slice': 2,  # FEM slices are 2 elements high
        'coupling_offset': 1e-6,  # Gap between slices
    },
    'elements': {
        'type': 'quad',
        'order': 1,
    },
    'material': {
        'block': {'E': 10e9, 'nu': 0.0, 'rho': 0},
        'fem': {'E': 10e9, 'nu': 0.0, 'rho': 0},
    },
    'contact': {
        'kn': 10e9, 'ks': 10e9, 'LG': True, 'nb_cps': 20
    },
    'coupling': {
        'method': 'constraint',
        'tolerance': 0.001,
        'integration_order': 2,
        'penalty_stiffness': 1e12,
        'interface_orientation': 'horizontal',
    },
    'bc': {
        'type': 'compression',
    },
    'loads': {
        'Fx': 0,
        'Fy': -1e6,  # 1 MN compression
    },
    'io': {
        'filename': 'square_benchmark',
        'dir': RESULTS_ROOT,
        'show_nodes': False,
        'figsize': (8, 8),
        'scale': 50,
    }
}

# Compression benchmark configuration
COMPRESSION_BENCH = {
    **SQUARE_BASE,
    'bc': {'type': 'compression'},
    'loads': {'Fx': 0, 'Fy': -1e6},
    'io': {**SQUARE_BASE['io'], 'filename': 'compression_bench'},
}

# Shear benchmark configuration
SHEAR_BENCH = {
    **SQUARE_BASE,
    'bc': {'type': 'shear'},
    'loads': {'Fx': 1e6, 'Fy': 0},
    'io': {**SQUARE_BASE['io'], 'filename': 'shear_bench'},
}

# =============================================================================
# Coupling Method Configurations
# =============================================================================

COUPLING_CONFIGS = {
    'constraint': {'method': 'constraint', 'tolerance': 0.001},
    'penalty': {'method': 'penalty', 'tolerance': 0.001, 'penalty_stiffness': 1e12},
    'lagrange': {'method': 'lagrange', 'tolerance': 0.001},
    'mortar': {'method': 'mortar', 'tolerance': 0.001, 'integration_order': 2},
}


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_total_height(config):
    """Calculate total height from geometry config."""
    g = config['geometry']
    n_slices = g['n_slices']
    start_block = (g['start_with'] == 'block')

    if start_block:
        n_block = (n_slices + 1) // 2
        n_fem = n_slices // 2
    else:
        n_fem = (n_slices + 1) // 2
        n_block = n_slices // 2

    return (n_block * g['block_slice_height'] +
            n_fem * g['fem_slice_height'] +
            (n_slices - 1) * g.get('coupling_offset', 0))


def apply_boundary_conditions(St, config):
    """Apply boundary conditions based on config['bc']['type']."""
    g = config['geometry']
    l_conf = config['loads']
    bc_type = config['bc']['type']

    total_height = calculate_total_height(config)
    total_width = g['width']

    tol = 1e-3

    # Find node sets
    bottom_set = set(find_nodes_at_height(St, 0.0, tolerance=tol))
    top_set = set(find_nodes_at_height(St, total_height, tolerance=tol))
    left_set = set(find_nodes_at_length(St, 0.0, tolerance=tol))
    right_set = set(find_nodes_at_length(St, total_width, tolerance=tol))
    center_x_set = set(find_nodes_at_length(St, total_width / 2, tolerance=0.1))

    # Fix base nodes
    for node in bottom_set:
        St.fix_node(node_ids=[node], dofs=[0, 1, 2])

    if bc_type == 'shear':
        # Shear: Load at top-left corner
        nodes_to_load = list(set(find_nodes_at_height(St, total_height, tolerance=0.1)).intersection(left_set))
        if not nodes_to_load:
            nodes_to_load = [list(top_set)[0]]
        control_node = list(top_set.intersection(right_set))
        if control_node:
            control_node = control_node[0]
        else:
            control_node = list(top_set)[-1]

        if l_conf.get('Fx', 0) != 0:
            Fx = l_conf['Fx'] / len(nodes_to_load)
            St.load_node(node_ids=nodes_to_load, dofs=[0], force=Fx)

    elif bc_type == 'compression':
        # Compression: Load at top center
        nodes_to_load = list(top_set.intersection(center_x_set))
        if not nodes_to_load:
            nodes_to_load = list(top_set)
        control_node = nodes_to_load[0]

        if l_conf.get('Fy', 0) != 0:
            Fy = l_conf['Fy'] / len(nodes_to_load)
            St.load_node(node_ids=nodes_to_load, dofs=[1], force=Fy)

    else:
        # Default: distribute load over all top nodes
        nodes_to_load = list(top_set)
        control_node = list(top_set)[len(top_set) // 2]

        if l_conf.get('Fx', 0) != 0:
            Fx = l_conf['Fx'] / len(nodes_to_load)
            St.load_node(node_ids=nodes_to_load, dofs=[0], force=Fx)
        if l_conf.get('Fy', 0) != 0:
            Fy = l_conf['Fy'] / len(nodes_to_load)
            St.load_node(node_ids=nodes_to_load, dofs=[1], force=Fy)

    return control_node, len(bottom_set), len(nodes_to_load)


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_single_benchmark(base_config: dict, coupling_method: str,
                         reference_disp: float = None,
                         verbose: bool = True) -> BenchmarkResult:
    """
    Run a single benchmark with specified coupling method.
    """
    method_names = {
        'constraint': 'Constraint',
        'penalty': 'Penalty',
        'lagrange': 'Lagrange',
        'mortar': 'Mortar'
    }

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"  Running: {method_names.get(coupling_method, coupling_method)}")
        print(f"{'=' * 50}")

    result = BenchmarkResult(method=coupling_method, n_dofs=0, n_coupled_nodes=0)

    # --- Create configuration ---
    coupling_cfg = COUPLING_CONFIGS[coupling_method].copy()
    bc_type = base_config['bc']['type']
    config = create_config(base_config, f'bench_{bc_type}_{coupling_method}', coupling=coupling_cfg)

    # --- Create model ---
    t_start = time.perf_counter()

    try:
        St = create_hybrid_column_slices(config)
    except Exception as e:
        print(f"  [ERROR] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return result

    t_model = time.perf_counter()

    # Record DOF info
    result.n_dofs = St.nb_dofs
    if hasattr(St, 'coupled_fem_nodes') and St.coupled_fem_nodes:
        result.n_coupled_nodes = len(St.coupled_fem_nodes)

    # --- Apply boundary conditions ---
    try:
        control_node, n_fixed, n_loaded = apply_boundary_conditions(St, config)
    except Exception as e:
        print(f"  [ERROR] BC application failed: {e}")
        import traceback
        traceback.print_exc()
        return result

    if verbose:
        print(f"  Model: {len(St.list_nodes)} nodes, {St.nb_dofs} DOFs")
        print(f"  Coupled nodes: {result.n_coupled_nodes}")
        print(f"  Fixed base nodes: {n_fixed}, Loaded nodes: {n_loaded}")

    # --- Assembly ---
    try:
        St.get_K_str0()
        St.get_P_r()
    except Exception as e:
        print(f"  [ERROR] Assembly failed: {e}")
        import traceback
        traceback.print_exc()
        return result

    t_assembly = time.perf_counter()
    result.time_assembly = t_assembly - t_model

    # --- Solve ---
    try:
        if coupling_method in ['lagrange', 'mortar']:
            St = StaticLinear.solve_augmented(St)
            result.matrix_type = "Indefinite"
        else:
            St = StaticLinear.solve(St, optimized=True)
            result.matrix_type = "SPD"
    except Exception as e:
        print(f"  [ERROR] Solve failed: {e}")
        import traceback
        traceback.print_exc()
        return result

    t_solve = time.perf_counter()
    result.time_solve = t_solve - t_assembly
    result.time_total = t_solve - t_start

    # --- Extract results ---

    # Expand displacement if needed (constraint coupling)
    U = St.U
    if coupling_method == 'constraint' and hasattr(St, 'coupling_T') and St.coupling_T is not None:
        try:
            U = St.coupling_T @ St.U
        except:
            pass

    # --- Visualization (optional) ---
    try:
        plot_initial(St, config)
        plot_deformed(St, config)
        plot_stress(St, config)
    except:
        pass

    # Control point displacement
    try:
        dofs = St.get_dofs_from_node(control_node)
        bc_type = config['bc']['type']
        # Use appropriate displacement component based on load direction
        if bc_type == 'shear' or config['loads'].get('Fx', 0) != 0:
            result.tip_displacement = abs(U[dofs[0]])
        else:
            result.tip_displacement = abs(U[dofs[1]])
    except:
        result.tip_displacement = np.max(np.abs(U))

    # Displacement error
    if reference_disp is not None and reference_disp > 0:
        result.displacement_error = (result.tip_displacement - reference_disp) / reference_disp

    # Constraint residual
    try:
        if coupling_method == 'constraint' and hasattr(St, 'constraint_coupling'):
            if St.constraint_coupling is not None:
                max_err = St.constraint_coupling.verify_constraints(St, U)
                result.constraint_residual = max_err
        elif coupling_method == 'penalty' and hasattr(St, 'penalty_coupling'):
            if St.penalty_coupling is not None:
                errors, max_err = St.penalty_coupling.compute_constraint_errors(St, U)
                result.constraint_residual = max_err
        elif coupling_method == 'lagrange' and hasattr(St, 'lagrange_coupling'):
            if St.lagrange_coupling is not None:
                C = St.lagrange_coupling.constraint_matrix_C
                if C is not None:
                    n_u = min(C.shape[1], len(U))
                    result.constraint_residual = np.linalg.norm(C @ U[:n_u])
        elif coupling_method == 'mortar' and hasattr(St, 'mortar_coupling'):
            if St.mortar_coupling is not None:
                C = St.mortar_coupling.constraint_matrix_C
                if C is not None:
                    n_u = min(C.shape[1], len(U))
                    result.constraint_residual = np.linalg.norm(C @ U[:n_u])
    except Exception as e:
        if verbose:
            print(f"  [WARN] Constraint residual: {e}")

    # Condition number (only for small systems)
    try:
        if result.n_dofs < 500:
            K = St.K0
            if hasattr(St, 'K_reduced') and St.K_reduced is not None:
                K = St.K_reduced
            if K is not None:
                result.condition_number = np.linalg.cond(K)
    except:
        pass

    # Interface forces
    try:
        if coupling_method == 'lagrange' and hasattr(St, 'lagrange_coupling'):
            if St.lagrange_coupling is not None and hasattr(St.lagrange_coupling, 'multipliers'):
                lam = St.lagrange_coupling.multipliers
                if lam is not None and len(lam) > 0:
                    result.interface_force_available = True
                    result.interface_force_magnitude = np.linalg.norm(lam)
        elif coupling_method == 'mortar' and hasattr(St, 'mortar_coupling'):
            if St.mortar_coupling is not None and hasattr(St.mortar_coupling, 'multipliers'):
                lam = St.mortar_coupling.multipliers
                if lam is not None and len(lam) > 0:
                    result.interface_force_available = True
                    result.interface_force_magnitude = np.linalg.norm(lam)
        elif coupling_method == 'penalty' and hasattr(St, 'penalty_coupling'):
            if St.penalty_coupling is not None:
                forces = St.penalty_coupling.estimate_interface_forces(St, U)
                if forces:
                    result.interface_force_available = True
                    result.interface_force_magnitude = sum(np.linalg.norm(f) for f in forces.values())
    except:
        pass

    # System dimension
    if coupling_method == 'constraint':
        result.system_dimension = result.n_dofs - 2 * result.n_coupled_nodes
    elif coupling_method in ['lagrange', 'mortar']:
        result.system_dimension = result.n_dofs + 2 * result.n_coupled_nodes
    else:
        result.system_dimension = result.n_dofs

    if verbose:
        print(f"  Tip displacement: {result.tip_displacement:.6e} m ({result.tip_displacement * 1000:.4f} mm)")
        if reference_disp:
            print(f"  Relative error: {result.displacement_error * 100:.4f}%")
        print(f"  Constraint residual: {result.constraint_residual:.3e}")
        print(
            f"  Time: {result.time_total * 1000:.1f} ms (assembly: {result.time_assembly * 1000:.1f}, solve: {result.time_solve * 1000:.1f})")
        if result.condition_number > 0:
            print(f"  Condition number: {result.condition_number:.2e}")

    return result


def run_benchmark_suite(base_config, name: str = None, verbose: bool = True) -> BenchmarkSuite:
    """
    Run all coupling methods on a specified benchmark.
    """
    methods = ['constraint', 'penalty', 'lagrange', 'mortar']

    suite_name = name or base_config['io']['filename']

    suite = BenchmarkSuite(
        name=suite_name,
        description=f"BC: {base_config['bc']['type']}, Loads: Fx={base_config['loads'].get('Fx', 0) / 1e3:.0f}kN, Fy={base_config['loads'].get('Fy', 0) / 1e3:.0f}kN",
        reference_displacement=0.0
    )

    # Use constraint solution as reference
    ref_result = run_single_benchmark(base_config, 'constraint', verbose=verbose)
    suite.reference_displacement = ref_result.tip_displacement
    suite.results['constraint'] = ref_result

    for method in methods[1:]:
        result = run_single_benchmark(
            base_config, method,
            reference_disp=suite.reference_displacement,
            verbose=verbose
        )
        suite.results[method] = result

    return suite


# =============================================================================
# Output Formatting
# =============================================================================

def format_latex_table(suite: BenchmarkSuite) -> str:
    """Generate LaTeX table for thesis."""
    methods = ['constraint', 'penalty', 'lagrange', 'mortar']

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"    \centering")
    lines.append(f"    \\caption{{Results for {suite.name}.}}")
    lines.append(r"    \begin{tabularx}{\textwidth}{lXXXX}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Metric} & \textbf{Constraint} & \textbf{Penalty} & \textbf{Lagrange} & \textbf{Mortar} \\")
    lines.append(r"        \midrule")

    # System dimension
    dims = [str(suite.results[m].system_dimension) if m in suite.results else "---" for m in methods]
    lines.append(f"        System dimension & {' & '.join(dims)} \\\\")

    # Displacement
    disps = []
    for m in methods:
        if m in suite.results:
            d = suite.results[m].tip_displacement
            disps.append(f"{d * 1000:.4f}")
        else:
            disps.append("---")
    lines.append(f"        Tip displacement (mm) & {' & '.join(disps)} \\\\")

    # Displacement error
    errs = []
    for m in methods:
        if m in suite.results:
            e = suite.results[m].displacement_error
            errs.append(f"{e * 100:.2f}\\%")
        else:
            errs.append("---")
    lines.append(f"        Relative error & {' & '.join(errs)} \\\\")

    # Constraint residual
    res = []
    for m in methods:
        if m in suite.results:
            r = suite.results[m].constraint_residual
            if r > 0:
                exp = int(np.floor(np.log10(abs(r))))
                mant = r / (10 ** exp)
                res.append(f"${mant:.1f} \\times 10^{{{exp}}}$")
            else:
                res.append("$< 10^{-15}$")
        else:
            res.append("---")
    lines.append(f"        Constraint residual & {' & '.join(res)} \\\\")

    # Total time
    times = []
    for m in methods:
        if m in suite.results:
            t = suite.results[m].time_total * 1000
            times.append(f"{t:.1f}")
        else:
            times.append("---")
    lines.append(f"        Total time (ms) & {' & '.join(times)} \\\\")

    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabularx}")
    lines.append(r"\end{table}")

    return '\n'.join(lines)


def format_markdown_table(suite: BenchmarkSuite) -> str:
    """Generate Markdown table for documentation."""
    methods = ['constraint', 'penalty', 'lagrange', 'mortar']

    lines = []
    lines.append(f"# {suite.name}")
    lines.append(f"")
    lines.append(f"**Configuration:** {suite.description}")
    lines.append(f"")
    lines.append(
        f"**Reference displacement:** {suite.reference_displacement:.6e} m ({suite.reference_displacement * 1000:.4f} mm)")
    lines.append(f"")
    lines.append("| Metric | Constraint | Penalty | Lagrange | Mortar |")
    lines.append("|--------|-----------|---------|----------|--------|")

    metrics = [
        ('System dimension', 'system_dimension', "{}"),
        ('Matrix type', 'matrix_type', "{}"),
        ('Tip displacement (mm)', 'tip_displacement', "{:.4f}", lambda x: x * 1000),
        ('Relative error (%)', 'displacement_error', "{:.4f}", lambda x: x * 100),
        ('Constraint residual', 'constraint_residual', "{:.2e}"),
        ('Assembly time (ms)', 'time_assembly', "{:.1f}", lambda x: x * 1000),
        ('Solve time (ms)', 'time_solve', "{:.1f}", lambda x: x * 1000),
        ('Total time (ms)', 'time_total', "{:.1f}", lambda x: x * 1000),
        ('Condition number', 'condition_number', "{:.2e}"),
    ]

    for item in metrics:
        name = item[0]
        attr = item[1]
        fmt = item[2]
        transform = item[3] if len(item) > 3 else lambda x: x

        values = []
        for m in methods:
            if m in suite.results:
                try:
                    val = getattr(suite.results[m], attr)
                    if val == 0 and attr == 'condition_number':
                        values.append("N/A")
                    else:
                        values.append(fmt.format(transform(val)))
                except:
                    values.append("---")
            else:
                values.append("---")
        lines.append(f"| {name} | {' | '.join(values)} |")

    return '\n'.join(lines)


def print_summary(suites: List[BenchmarkSuite]):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("  SQUARE BENCHMARK SUMMARY")
    print("=" * 70)

    for suite in suites:
        print(f"\n{suite.name}")
        print("-" * len(suite.name))
        print(f"Configuration: {suite.description}")
        print(f"Reference: {suite.reference_displacement:.6e} m ({suite.reference_displacement * 1000:.4f} mm)\n")

        print(f"{'Method':<12} {'Disp (mm)':<12} {'Error (%)':<10} {'Residual':<12} {'Time (ms)':<10}")
        print("-" * 56)

        for method in ['constraint', 'penalty', 'lagrange', 'mortar']:
            if method in suite.results:
                r = suite.results[method]
                err_str = f"{r.displacement_error * 100:.4f}" if r.displacement_error != 0 else "ref"
                print(
                    f"{method:<12} {r.tip_displacement * 1000:<12.4f} {err_str:<10} {r.constraint_residual:<12.2e} {r.time_total * 1000:<10.1f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  SQUARE GEOMETRY BENCHMARK")
    print("  Alternating Block/FEM Horizontal Slices")
    print("=" * 70)

    suites = []

    # Run compression benchmark
    print("\n" + "=" * 70)
    print("  COMPRESSION BENCHMARK")
    print("=" * 70)
    compression_suite = run_benchmark_suite(COMPRESSION_BENCH, name="Square - Compression")
    suites.append(compression_suite)

    # Run shear benchmark
    print("\n" + "=" * 70)
    print("  SHEAR BENCHMARK")
    print("=" * 70)
    shear_suite = run_benchmark_suite(SHEAR_BENCH, name="Square - Shear")
    suites.append(shear_suite)

    # Print summary
    print_summary(suites)

    # Save output
    output_dir = Path(RESULTS_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    for suite in suites:
        safe_name = suite.name.lower().replace(' ', '_').replace('-', '_')

        # Markdown
        md_content = format_markdown_table(suite)
        md_path = output_dir / f"{safe_name}.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"\n[OK] Markdown saved to: {md_path}")

        # LaTeX
        tex_content = format_latex_table(suite)
        tex_path = output_dir / f"{safe_name}.tex"
        with open(tex_path, 'w') as f:
            f.write(tex_content)
        print(f"[OK] LaTeX saved to: {tex_path}")


if __name__ == "__main__":
    main()
