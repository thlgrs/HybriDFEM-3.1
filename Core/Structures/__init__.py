"""
HybriDFEM-3 Structure Classes

This module provides structure classes for hybrid finite element analysis.

Structure Classes
-----------------
Structure_2D : Abstract base class
    - Common interface for all 2D structures
    - Variable DOF management per node
    - KD-tree optimization for node searches

Structure_FEM : Pure continuous FEM structures
    - Triangular/quadrilateral plane stress/strain elements
    - 2 DOF per node [ux, uy]

StructureBlock : Pure discrete block assemblies
    - Rigid blocks with contact mechanics
    - 3 DOF per node [ux, uy, rz]

Hybrid : Combined block-FEM structures
    - Multiple inheritance from StructureBlock and Structure_FEM
    - Four coupling methods: constraint, penalty, Lagrange, mortar

Block Generators (preset geometries)
------------------------------------
BeamBlock : Uniform beam with rectangular blocks
TaperedBeamBlock : Tapered beam with varying depth
ArchBlock : Circular arch with wedge-shaped voussoirs
WallBlock : Rectangular wall with block pattern
VoronoiBlock : Voronoi tessellation of irregular blocks

Usage
-----
>>> from Core.Structures import Hybrid, Structure_FEM, Structure_Block
>>> from Core.Structures import BeamBlock, ArchBlock
>>>
>>> # Pure FEM structure
>>> fem = Structure_FEM()
>>>
>>> # Pure block structure
>>> block = Structure_Block()
>>>
>>> # Hybrid structure
>>> hybrid = Hybrid()
>>>
>>> # Preset block geometry
>>> beam = BeamBlock(length=10.0, height=0.5, n_blocks=20)
>>> arch = ArchBlock(radius=5.0, angle=90, n_voussoirs=15)
"""

from .Structure_2D import Structure_2D
from .Structure_Block import (
    ArchBlock,
    BeamBlock,
    Structure_Block,
    TaperedBeamBlock,
    VoronoiBlock,
    WallBlock,
)
from .Structure_FEM import Structure_FEM
from .Structure_Hybrid import Hybrid

__all__ = [
    # Base class
    'Structure_2D',

    # Main structure classes
    'Structure_FEM',
    'Structure_Block',
    'Hybrid',

    # Block generators (preset geometries)
    'BeamBlock',
    'TaperedBeamBlock',
    'ArchBlock',
    'WallBlock',
    'VoronoiBlock',
]
