"""
Examples Utilities Package
==========================

Centralized utilities for running HybriDFEM examples:

- helpers: Configuration management, path utilities
- model_builders: High-level structure builders
- mesh_generation: Low-level vectorized mesh creation
- boundary_conditions: Common BC patterns
- solvers: Unified solver interface
- visualization: Standardized plotting
- reporting: Comprehensive analysis reports
- runner: Unified execution pipeline
- compare_results: Multi-analysis comparison
- export: CSV export for pandas analysis

Quick Start:
    from Examples.utils import run_example, create_config
    from Examples.utils.model_builders import create_fem_column
    from Examples.utils.boundary_conditions import apply_cantilever_bc

    config = create_config(BASE_CONFIG, 'my_analysis', ...)
    St, report = run_example(config, create_fem_column, apply_cantilever_bc)

Export to CSV:
    from Examples.utils.export import export_state
    paths = export_state(St, config)
"""

# Core helpers
from Examples.utils.helpers import (
    create_config,
    detect_structure_type,
    detect_analysis_type,
    detect_elem_type,
    RESULTS_ROOT,
    ELEMENT_MAP
)

# Runner utilities
from Examples.utils.runner import (
    run_example,
    run_examples,
    ExampleRunner
)

# Boundary conditions
from Examples.utils.boundary_conditions import (
    apply_cantilever_bc,
    apply_compression_bc,
    apply_shear_bc,
    apply_tip_load_bc,
    apply_conditions_from_config,
    find_node_sets,
    get_bc_function
)

# Reporting
from Examples.utils.reporting import (
    AnalysisReport,
    plot_history
)

# Visualization
from Examples.utils.visualization import (
    visualize,
    plot_initial,
    plot_deformed,
    plot_stress,
    plot_displacement
)

# Solvers
from Examples.utils.solvers import run_solver

# Export utilities
from Examples.utils.export import (
    export_state,
    export_nodes,
    export_displacements,
    export_stresses,
    export_reactions,
    export_elements,
    load_displacements,
    load_stresses,
    # Linear system Ku = P
    export_system,
    export_stiffness_matrix,
    export_load_vector,
    export_displacement_vector,
    load_stiffness_matrix,
    load_load_vector,
    load_system
)

# Model builders (selective export)
from Examples.utils.model_builders import (
    create_block_column,
    create_fem_column,
    create_hybrid_column,
    create_hybrid_beam_slices,
    create_hybrid_column_slices,
    setup_contact,
    find_nodes_at_base,
    find_nodes_at_height,
    find_nodes_at_length
)

__all__ = [
    # Config
    'create_config',
    'detect_structure_type',
    'detect_analysis_type',
    'detect_elem_type',
    'RESULTS_ROOT',
    'ELEMENT_MAP',
    # Runner
    'run_example',
    'run_examples',
    'ExampleRunner',
    # Boundary conditions
    'apply_cantilever_bc',
    'apply_compression_bc',
    'apply_shear_bc',
    'apply_tip_load_bc',
    'apply_conditions_from_config',
    'find_node_sets',
    'get_bc_function',
    # Reporting
    'AnalysisReport',
    'plot_history',
    # Visualization
    'visualize',
    'plot_initial',
    'plot_deformed',
    'plot_stress',
    'plot_displacement',
    # Solvers
    'run_solver',
    # Export
    'export_state',
    'export_nodes',
    'export_displacements',
    'export_stresses',
    'export_reactions',
    'export_elements',
    'load_displacements',
    'load_stresses',
    # Linear system
    'export_system',
    'export_stiffness_matrix',
    'export_load_vector',
    'export_displacement_vector',
    'load_stiffness_matrix',
    'load_load_vector',
    'load_system',
    # Model builders
    'create_block_column',
    'create_fem_column',
    'create_hybrid_column',
    'create_hybrid_beam_slices',
    'create_hybrid_column_slices',
    'setup_contact',
    'find_nodes_at_base',
    'find_nodes_at_height',
    'find_nodes_at_length',
]
