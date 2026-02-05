"""
Continuous Finite Element (FEM) Module

This module provides continuous finite elements for structural analysis:
- Triangular elements for 2D plane stress/strain analysis
- Quadrilateral elements for 2D plane stress/strain analysis
- Beam elements for frame structures
- Mesh utilities for element generation

Elements
--------
FE : Abstract base class for all finite elements

Element2D : 2D solid elements base class
    - Triangle3: 3-node linear triangular element (CST)
    - Triangle6: 6-node quadratic triangular element (LST)
    - Quad4: 4-node bilinear quadrilateral element
    - Quad8: 8-node serendipity quadrilateral element
    - Quad9: 9-node Lagrangian quadrilateral element

Timoshenko : Beam elements
    - 2-node beam with shear deformation
    - Suitable for frame structures

Geometry Classes
----------------
Geometry2D : 2D element geometry
    - t: Thickness for plane stress/strain elements

GeometryBeam : Beam geometry
    - A: Cross-sectional area
    - I: Second moment of area

QuadRule : Quadrature rule definition
    - Integration points and weights

Mesh Utilities
--------------
Mesh : Mesh generation and manipulation
    - Import from gmsh format
    - Node and element management

Coupling Utilities
------------------
CoupledSystem : Test harness for isolated coupling validation
GeneralizedCoupledSystem : Extended coupling system

Usage
-----
>>> from Core.Objects.FEM import Triangle3, Quad4, Geometry2D
>>> from Core.Objects.ConstitutiveLaw import PlaneStress
>>>
>>> mat = PlaneStress(E=200e9, nu=0.3, rho=7850)
>>> geom = Geometry2D(t=0.01)  # 10mm thickness
>>>
>>> tri = Triangle3(nodes=[(0,0), (1,0), (0,1)], mat=mat, geom=geom)
>>> quad = Quad4(nodes=[(0,0), (1,0), (1,1), (0,1)], mat=mat, geom=geom)
"""

from .BaseFE import BaseFE
from .Element2D import Element2D, Geometry2D, QuadRule
from .Mesh import Mesh
from .Quads import Quad4, Quad8, Quad9
from .Timoshenko import GeometryBeam, Timoshenko
from .Triangles import Triangle3, Triangle6

__all__ = [
    # Base classes
    'BaseFE',
    'Element2D',
    'QuadRule',

    # Triangular elements
    'Triangle3',
    'Triangle6',

    # Quadrilateral elements
    'Quad4',
    'Quad8',
    'Quad9',

    # Beam elements
    'Timoshenko',

    # Geometry classes
    'Geometry2D',
    'GeometryBeam',

    # Mesh utilities
    'Mesh',

]
