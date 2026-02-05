"""
HybriDFEM - Hybrid Finite Element Method Framework

A Python framework for hybrid structural analysis combining:
- Discrete Finite Element Method (DFEM): Rigid block assemblies with contact mechanics
- Continuous Finite Element Method (FEM): Triangular/quadrilateral plane stress/strain elements
- Hybrid coupling: Four methods for block-FEM interaction

Main Components
---------------
Structures : Structure classes for analysis
    - Structure_2D: Abstract base class with variable DOF support
    - Structure_FEM: Pure continuous FEM structures
    - StructureBlock: Pure discrete block assemblies
    - Hybrid: Combined block-FEM structures with coupling
    - Block generators: BeamBlock, ArchBlock, WallBlock, etc.

Solvers : Analysis algorithms
    - StaticLinear: Linear static analysis (aliased as Static)
    - StaticNonLinear: Nonlinear static analysis (force/displacement control)
    - Dynamic: Time-history dynamic analysis (Central Difference Method)
    - Modal: Eigenvalue analysis for natural frequencies

Visualization : Post-processing
    - Visualizer: Deformed shape and stress visualization
    - PlotStyle: Plot styling configuration
    - Plotter: Simple plotting utilities

Objects : Element and material models (import from Core.Objects)
    - FEM: Continuous finite elements (Triangle3, Triangle6, Quad4, Timoshenko)
    - DFEM: Rigid blocks with contact mechanics
    - ConstitutiveLaw: Constitutive models (PlaneStress, PlaneStrain, etc.)
    - Coupling: Block-FEM coupling methods (constraint, penalty, Lagrange, mortar)

Quick Start
-----------
>>> from Core import Hybrid, Static, Visualizer
>>> from Core.Objects import Triangle3, Block_2D
>>> from Core.Objects.ConstitutiveLaw import PlaneStress
>>>
>>> # Create hybrid structure
>>> structure = Hybrid()
>>> # Add blocks and elements...
>>> structure.make_nodes()
>>> structure.enable_block_fem_coupling(method='constraint')
>>>
>>> # Run analysis
>>> structure = Static.solve_linear(structure)
>>>
>>> # Visualize
>>> viz = Visualizer(structure)
>>> viz.plot_deformed_shape(scale=10)

Version: 3.0
Author: HybriDFEM Development Team
"""

# Version information
__version__ = '3.0.0'
__author__ = 'UCLouvain HybriDFEM'

# Objects package (available but typically imported from subpackages)
from Core import Objects

# Solver classes
from Core.Solvers import (
    # Main solvers
    Static,
    StaticLinear,
    StaticNonLinear,
    Dynamic,
    Modal,
    Solver,
    # Visualization
    Visualizer,
    PlotStyle,
    Plotter,
    # Exceptions
    ConvergenceError,
    SingularSystemError,
)

# Structure classes (most commonly used)
from Core.Structures import (
    # Base and main structures
    Structure_2D,
    Structure_FEM,
    Structure_Block,
    Hybrid,
    # Block generators
    BeamBlock,
    TaperedBeamBlock,
    ArchBlock,
    WallBlock,
    VoronoiBlock,
)

# Define public API
__all__ = [
    # Version info
    '__version__',
    '__author__',

    # Structures (primary API)
    'Structure_2D',
    'Structure_FEM',
    'Structure_Block',
    'Hybrid',

    # Block generators
    'BeamBlock',
    'TaperedBeamBlock',
    'ArchBlock',
    'WallBlock',
    'VoronoiBlock',

    # Solvers (primary API)
    'Static',
    'StaticLinear',
    'StaticNonLinear',
    'Dynamic',
    'Modal',
    'Solver',

    # Visualization
    'Visualizer',
    'PlotStyle',
    'Plotter',

    # Exceptions
    'ConvergenceError',
    'SingularSystemError',

    # Objects (subpackage)
    'Objects',
]
