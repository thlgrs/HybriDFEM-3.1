"""
HybriDFEM Objects

Finite elements, blocks, materials, and coupling methods.

This package contains the building blocks for structural models:
- FEM: Continuous finite elements
- DFEM: Discrete (rigid block) elements
- ConstitutiveLaw: Constitutive models
- Coupling: Block-FEM coupling strategies

Subpackages
-----------
FEM : Continuous finite elements
    - Triangle3: 3-node linear triangular element (CST)
    - Triangle6: 6-node quadratic triangular element (LST)
    - Quad4: 4-node bilinear quadrilateral element
    - Quad8: 8-node serendipity quadrilateral element
    - Quad9: 9-node Lagrangian quadrilateral element
    - Timoshenko: 2-node beam element with shear deformation
    - Geometry2D: Geometric properties (thickness)
    - Mesh: Mesh import and manipulation utilities

DFEM : Discrete finite elements (rigid blocks)
    - Block_2D: Rigid polygonal block
    - Contact: Contact laws (elastic, no-tension, Coulomb friction)
    - CF_2D: Contact face interface between blocks
    - CP_2D: Point-to-point contact pair
    - Spring_2D: Spring element
    - Surface: Surface element

ConstitutiveLaw : Constitutive models
    - ConstitutiveLaw: 1D base class
    - PlaneStress: 2D plane stress
    - PlaneStrain: 2D plane strain
    - TimoshenkoMaterial: Beam material
    - (Plus plastic, damage, and nonlinear models)

Coupling : Block-FEM coupling methods
    - ConstraintCoupling: DOF elimination (fastest, exact)
    - PenaltyCoupling: Stiffness springs (simple, approximate)
    - LagrangeCoupling: Lagrange multipliers (exact, force output)
    - MortarCoupling: Distributed multipliers (non-matching meshes)

Usage
-----
>>> from Core.Objects import Triangle3, Quad4, Block_2D
>>> from Core.Objects.ConstitutiveLaw import PlaneStress
>>> from Core.Objects.FEM import Geometry2D
>>>
>>> # Create material
>>> mat = PlaneStress(E=200e9, nu=0.3, rho=7850)
>>> geom = Geometry2D(t=0.01)
>>>
>>> # Create elements
>>> tri = Triangle3(nodes=[(0,0), (1,0), (0,1)], mat=mat, geom=geom)
>>> quad = Quad4(nodes=[(0,0), (1,0), (1,1), (0,1)], mat=mat, geom=geom)
"""

from Core.Objects import ConstitutiveLaw
# Import subpackages
from Core.Objects import Coupling
from Core.Objects import DFEM
from Core.Objects import FEM
# Materials (commonly used)
from Core.Objects.ConstitutiveLaw import (
    Material as MaterialBase,
    PlaneStress,
    PlaneStrain,
    TimoshenkoMaterial,
    Bilinear_Mat,
    Plastic_Mat,
    SimplePlaneStressBilinear,
)
# Coupling methods
from Core.Objects.Coupling import (
    BaseCoupling,
    Condensation,
    PenaltyCoupling,
    LagrangeCoupling,
    MortarCoupling,
    NoCoupling,
)
# DFEM elements
from Core.Objects.DFEM import (
    Block_2D,
    Contact,
    NoTension_EP,
    NoTension_CD,
    Coulomb,
    Bilinear,
    CF_2D,
    CP_2D,
    Spring_2D,
    Surface,
)
# FEM elements (commonly used)
from Core.Objects.FEM import (
    # Base classes
    BaseFE,
    Element2D,
    # Triangular elements
    Triangle3,
    Triangle6,
    # Quadrilateral elements
    Quad4,
    Quad8,
    Quad9,
    # Beam elements
    Timoshenko,
    # Geometry classes
    Geometry2D,
    GeometryBeam,
    # Mesh utilities
    Mesh,
)

__all__ = [
    # Subpackages
    'FEM',
    'DFEM',
    'ConstitutiveLaw',
    'Coupling',

    # FEM base classes
    'BaseFE',
    'Element2D',

    # FEM elements
    'Triangle3',
    'Triangle6',
    'Quad4',
    'Quad8',
    'Quad9',
    'Timoshenko',
    'Geometry2D',
    'GeometryBeam',
    'Mesh',

    # DFEM elements
    'Block_2D',
    'Contact',
    'NoTension_EP',
    'NoTension_CD',
    'Coulomb',
    'Bilinear',
    'CF_2D',
    'CP_2D',
    'Spring_2D',
    'Surface',

    # Materials
    'MaterialBase',
    'PlaneStress',
    'PlaneStrain',
    'TimoshenkoMaterial',
    'Bilinear_Mat',
    'Plastic_Mat',
    'SimplePlaneStressBilinear',

    # Coupling
    'BaseCoupling',
    'Condensation',
    'PenaltyCoupling',
    'LagrangeCoupling',
    'MortarCoupling',
    'NoCoupling',
]
