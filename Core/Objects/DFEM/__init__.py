"""
Discrete Finite Element Method (DFEM) Module

This module provides rigid block elements with contact mechanics for
discontinuous structural analysis.

Elements
--------
Block_2D : Rigid polygonal block
    - 3 DOFs per block: [ux, uy, rotation_z]
    - Geometric properties computed from vertices
    - Reference point for rigid body kinematics

Contact Laws
------------
Contact : Base elastic contact
NoTension_EP : Elastic no-tension contact
NoTension_CD : No-tension with contact deletion
Coulomb : Mohr-Coulomb plasticity with friction
Bilinear : Bilinear elastic-plastic contact

Contact Interfaces
------------------
ContactFace (CF_2D) : Manages multiple contact pairs along block edges
ContactPair (CP_2D) : Individual point-to-point contact

Other
-----
Spring : Spring element for connections
Surface : Surface element for interfaces

Usage
-----
>>> from Core.Objects.DFEM import Block_2D, Coulomb
>>>
>>> # Create rectangular block
>>> vertices = np.array([[0,0], [1,0], [1,1], [0,1]])
>>> block = Block_2D(vertices, b=1.0)  # b = out-of-plane width
>>>
>>> # Create contact law
>>> contact = Coulomb(kn=1e9, ks=1e8, mu=0.6, c=0, psi=0)
"""

from Core.Objects.ConstitutiveLaw.Contact import (
    Contact,
    NoTension_EP,
    NoTension_CD,
    Coulomb,
    Bilinear
)
from Core.Objects.ConstitutiveLaw.Spring import Spring_2D

from .Block import Block_2D
from .ContactFace import CF_2D
from .ContactPair import CP_2D
from .Surface import Surface

__all__ = [
    # Block element
    'Block_2D',

    # Contact laws
    'Contact',
    'NoTension_EP',
    'NoTension_CD',
    'Coulomb',
    'Bilinear',

    # Contact management
    'CF_2D',
    'CP_2D',

    # Other elements
    'Spring_2D',
    'Surface',
]
