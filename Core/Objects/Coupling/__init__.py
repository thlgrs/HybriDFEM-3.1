"""
Coupling Module for HybriDFEM

This module provides coupling strategies between rigid blocks (DFEM) and
continuous FEM elements in hybrid structures.

Coupling Methods
----------------
ConstraintCoupling : DOF elimination (fastest, exact)
    - Eliminates coupled FEM DOFs using transformation matrix
    - Requires matching meshes (exact nodal coincidence)

PenaltyCoupling : Stiffness springs (simple, approximate)
    - Adds penalty springs between coupled nodes
    - Requires matching meshes, needs parameter tuning

LagrangeCoupling : Lagrange multipliers (exact, force output)
    - Enforces constraints via Lagrange multipliers at nodes
    - Requires matching meshes, provides interface forces

MortarCoupling : Distributed multipliers (exact, non-matching meshes)
    - Enforces constraints via distributed multipliers along interfaces
    - Works with non-matching meshes, uses numerical integration

Base Classes
------------
CouplingBase : Abstract base class for all coupling methods
NoCoupling : Null coupling for baseline comparison (no-op)

Utilities
---------
ConstraintMatrix : Utilities for constraint matrix computations
compute_rigid_body_constraint : Compute rigid body constraint for a node

Integration
-----------
IntegrationPoint : Integration point with position and weight
MortarInterface : Interface definition for mortar coupling
InterfacePoint : Point on a mortar interface

Legacy
------
CouplingInterface : Deprecated base class (use CouplingBase instead)

Usage
-----
>>> from Core.Objects.Coupling import (
...     PenaltyCoupling, Condensation, LagrangeCoupling, MortarCoupling
... )
>>>
>>> # Penalty coupling (approximate, easy to implement)
>>> coupling = PenaltyCoupling(penalty_factor=1000.0, auto_scale=True)
>>>
>>> # Constraint coupling (exact, DOF reduction)
>>> coupling = Condensation(verbose=True)
>>>
>>> # Lagrange coupling (exact, direct interface forces)
>>> coupling = LagrangeCoupling(verbose=True)
>>>
>>> # Mortar coupling (exact, works with non-matching meshes)
>>> coupling = MortarCoupling(integration_order=2, interface_tolerance=1e-4)

Author: HybriDFEM Development Team
Date: January 2025
"""

# Base classes
from .BaseCoupling import BaseCoupling, NoCoupling
# Coupling method implementations
from .Condensation import Condensation
# Utility classes and functions
from .ConstraintMatrix import ConstraintMatrix, compute_rigid_body_constraint
from .IntegrationRule import IntegrationPoint
from .LagrangeNodal import LagrangeCoupling
from .Mortar import MortarCoupling
from .MortarInterface import InterfacePoint, MortarInterface
from .Penalty import PenaltyCoupling

__all__ = [
    # Base classes
    'BaseCoupling',
    'NoCoupling',

    # Coupling methods (main API)
    'Condensation',
    'PenaltyCoupling',
    'LagrangeCoupling',
    'MortarCoupling',

    # Utilities
    'ConstraintMatrix',
    'compute_rigid_body_constraint',

    # Integration and interface classes
    'IntegrationPoint',
    'MortarInterface',
    'InterfacePoint',

]

__version__ = '0.2.0'
