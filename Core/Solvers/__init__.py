"""
HybriDFEM Solvers

Analysis algorithms for structural mechanics problems.

This package provides solvers for:
- Static equilibrium (linear and nonlinear)
- Dynamic time-history response
- Modal analysis (eigenvalues and eigenvectors)

Solvers
-------
StaticLinear : Linear static equilibrium analysis
    - solve_linear(): Direct linear analysis
    - solve_linear_saddle_point(): Saddle point system for Lagrange/Mortar coupling

StaticNonLinear : Nonlinear static equilibrium analysis
    - solve_forcecontrol(): Incremental force control with Newton-Raphson
    - solve_dispcontrol(): Displacement control for softening behavior

Dynamic : Time-history analysis
    - CDM(): Central Difference Method (explicit)
    - Support for prescribed dynamic excitation

Modal : Eigenvalue analysis
    - solve_modal(): Natural frequencies and mode shapes
    - Support for subset or full spectrum

Solver : Base class
    - save(): Pickle structure to file
    - load(): Load structure from file

Visualizer : Post-processing visualization
    - plot_deformed_shape(): Visualize deformed structure
    - plot_stress_contour(): Stress field visualization

Plotter : Simple plotting utilities
    - Basic matplotlib-based plots

Exceptions
----------
ConvergenceError : Raised when solver fails to converge
SingularSystemError : Raised when system matrix is singular

Usage
-----
>>> from Core.Solvers import StaticLinear, Modal
>>>
>>> # Linear static analysis
>>> structure = StaticLinear.solve_linear(structure)
>>>
>>> # Modal analysis
>>> structure = Modal.solve_modal(structure, modes=10)
"""

from .Dynamic import Dynamic
from .Modal import Modal
from .Plotter import Plotter
from .Solver import Solver
from .Static import (
    ConvergenceError,
    SingularSystemError,
    SolverConstants,
    StaticBase,
    StaticLinear,
    StaticNonLinear,
)
from .Visualizer import PlotStyle, Visualizer

# Backward compatibility alias
Static = StaticLinear

__all__ = [
    # Solver classes
    'StaticLinear',
    'StaticNonLinear',
    'StaticBase',
    'Static',  # Alias for StaticLinear (backward compatibility)
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

    # Constants
    'SolverConstants',
]
