"""
Unified Example Runner
======================

Provides a standardized execution pipeline for all example types:
- FEM, Block, and Hybrid structures
- Linear and nonlinear analyses
- Timing, reporting, and visualization

Usage:
    from Examples.utils.runner import run_example, run_examples

    St, report = run_example(config, create_model, apply_conditions)
"""

import time
from typing import Callable, Dict, List, Optional, Tuple, Any

from Examples.utils.reporting import AnalysisReport, plot_history
from Examples.utils.solvers import run_solver
from Examples.utils.visualization import visualize


def run_example(
    config: Dict,
    create_model: Callable,
    apply_conditions: Callable,
    analyze_callback: Optional[Callable] = None,
    skip_visualization: bool = False,
    skip_report: bool = False,
    verbose: bool = True
) -> Tuple[Any, Optional[AnalysisReport]]:
    """
    Run a complete analysis example with standardized pipeline.

    Pipeline:
    1. Create model from config
    2. Apply boundary conditions and loads
    3. Solve the system
    4. Generate report (with timing)
    5. Create visualizations

    Args:
        config: Configuration dictionary with geometry, material, loads, io, etc.
        create_model: Function(config) -> Structure that builds the model
        apply_conditions: Function(St, config) -> (St, control_node) that applies BCs
        analyze_callback: Optional Function(St, config, control_node) for custom analysis
        skip_visualization: If True, skip generating plots
        skip_report: If True, skip generating markdown report
        verbose: If True, print progress messages

    Returns:
        Tuple of (solved_structure, report) where report is AnalysisReport or None
    """
    name = config['io'].get('filename', 'analysis')
    phase_times = {}

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"   Running: {name}")
        print(f"{'=' * 60}")

    # Track total time
    total_start = time.time()

    # Phase 1: Model creation
    t0 = time.time()
    St = create_model(config)
    phase_times['Model Creation'] = time.time() - t0

    # Phase 2: Apply conditions
    t0 = time.time()
    St, control_node = apply_conditions(St, config)
    phase_times['Boundary Conditions'] = time.time() - t0

    # Phase 3: Solve
    t0 = time.time()
    St = run_solver(St, config, control_node)
    phase_times['Solver'] = time.time() - t0

    # Phase 4: Custom analysis (optional)
    if analyze_callback:
        t0 = time.time()
        analyze_callback(St, config, control_node)
        phase_times['Custom Analysis'] = time.time() - t0

    # Phase 5: Report generation
    report = None
    if not skip_report:
        t0 = time.time()
        report = AnalysisReport(St, config, start_time=total_start)
        report.end_time = time.time()

        # Get loads for equilibrium check
        l = config.get('loads', config.get('loading', {}))
        applied_Fx = l.get('Fx', 0)
        applied_Fy = l.get('Fy', 0)

        report.full_report(
            control_node=control_node,
            applied_Fx=applied_Fx,
            applied_Fy=applied_Fy,
            phase_times=phase_times
        )
        report.save()
        phase_times['Reporting'] = time.time() - t0

    # Phase 6: Visualization
    if not skip_visualization:
        t0 = time.time()
        visualize(St, config)
        phase_times['Visualization'] = time.time() - t0

        # Plot history for nonlinear
        if 'control' in config:
            plot_history(St, config, control_node)

    total_time = time.time() - total_start
    if verbose:
        print(f"\n[OK] Completed in {total_time:.2f}s")

    return St, report


def run_examples(
    configs: List[Dict],
    create_model: Callable,
    apply_conditions: Callable,
    analyze_callback: Optional[Callable] = None,
    skip_visualization: bool = False,
    skip_report: bool = False,
    verbose: bool = True
) -> List[Tuple[Any, Optional[AnalysisReport]]]:
    """
    Run multiple examples with the same model builder and BC functions.

    Args:
        configs: List of configuration dictionaries
        create_model: Function(config) -> Structure
        apply_conditions: Function(St, config) -> (St, control_node)
        analyze_callback: Optional custom analysis function
        skip_visualization: If True, skip generating plots
        skip_report: If True, skip generating markdown reports
        verbose: If True, print progress messages

    Returns:
        List of (structure, report) tuples
    """
    n_configs = len(configs)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"   Running {n_configs} configuration(s)")
        print(f"{'=' * 60}")
        for i, config in enumerate(configs, 1):
            print(f"  {i}. {config['io'].get('filename', f'config_{i}')}")
        print()

    results = []
    for i, config in enumerate(configs, 1):
        if verbose and n_configs > 1:
            print(f"\n[{i}/{n_configs}] ", end='')

        result = run_example(
            config, create_model, apply_conditions,
            analyze_callback=analyze_callback,
            skip_visualization=skip_visualization,
            skip_report=skip_report,
            verbose=verbose
        )
        results.append(result)

    if verbose and n_configs > 1:
        print(f"\n{'=' * 60}")
        print(f"   Completed {n_configs} analyses")
        print(f"{'=' * 60}")

    return results


class ExampleRunner:
    """
    Object-oriented runner for more complex example workflows.

    Allows customizing each phase of the pipeline.
    """

    def __init__(self, config: Dict):
        """Initialize with configuration."""
        self.config = config
        self.St = None
        self.control_node = None
        self.report = None
        self.phase_times = {}
        self.start_time = None

    def build_model(self, create_model: Callable) -> 'ExampleRunner':
        """Build the model using provided function."""
        self.start_time = time.time()
        t0 = time.time()
        self.St = create_model(self.config)
        self.phase_times['Model Creation'] = time.time() - t0
        return self

    def apply_bcs(self, apply_conditions: Callable) -> 'ExampleRunner':
        """Apply boundary conditions using provided function."""
        t0 = time.time()
        self.St, self.control_node = apply_conditions(self.St, self.config)
        self.phase_times['Boundary Conditions'] = time.time() - t0
        return self

    def solve(self) -> 'ExampleRunner':
        """Run the solver."""
        t0 = time.time()
        self.St = run_solver(self.St, self.config, self.control_node)
        self.phase_times['Solver'] = time.time() - t0
        return self

    def generate_report(self) -> 'ExampleRunner':
        """Generate analysis report."""
        t0 = time.time()
        end_time = time.time()

        self.report = AnalysisReport(self.St, self.config, start_time=self.start_time)
        self.report.end_time = end_time

        l = self.config.get('loads', self.config.get('loading', {}))
        self.report.full_report(
            control_node=self.control_node,
            applied_Fx=l.get('Fx', 0),
            applied_Fy=l.get('Fy', 0),
            phase_times=self.phase_times
        )
        self.report.save()

        self.phase_times['Reporting'] = time.time() - t0
        return self

    def visualize(self) -> 'ExampleRunner':
        """Generate visualizations."""
        t0 = time.time()
        visualize(self.St, self.config)

        if 'control' in self.config:
            plot_history(self.St, self.config, self.control_node)

        self.phase_times['Visualization'] = time.time() - t0
        return self

    def run_all(self, create_model: Callable, apply_conditions: Callable) -> 'ExampleRunner':
        """Run the complete pipeline."""
        name = self.config['io'].get('filename', 'analysis')
        print(f"\n{'=' * 60}")
        print(f"   Running: {name}")
        print(f"{'=' * 60}")

        self.build_model(create_model)
        self.apply_bcs(apply_conditions)
        self.solve()
        self.generate_report()
        self.visualize()

        total_time = time.time() - self.start_time
        print(f"\n[OK] Completed in {total_time:.2f}s")

        return self
