"""
ConstitutiveLaw Constitutive Models

This module provides material models for structural analysis including
elastic, plastic, and nonlinear constitutive laws.

Base & 1D Materials
-------------------
Material : Base class for 1D/interface materials
ConstitutiveLaw : Alias for Material
Bilinear_Mat : Elastic-plastic with linear hardening
Plastic_Mat : Perfect plasticity
Mixed_Hardening_Mat : Combined kinematic/isotropic hardening
NoTension_Mat : Material that fails in tension

2D Continuum Materials (Plane Stress/Strain)
--------------------------------------------
PlaneStress : 2D plane stress formulation
PlaneStrain : 2D plane strain formulation
SimplePlaneStressBilinear : Simplified bilinear 2D plasticity

Beam Materials
--------------
TimoshenkoMaterial : Material for Timoshenko beam elements

Contact & Interface Laws
------------------------
Contact : Base linear elastic contact law
Bilinear : Bilinear contact law
NoTension_EP : No-tension contact with plastic strain tracking
NoTension_CD : No-tension contact with deletion
Coulomb : Mohr-Coulomb friction law

Spring Elements
---------------
Spring_2D : 2D spring element manager

Specialized Models
------------------
concrete_EC : Eurocode concrete model
steel_EC : Eurocode steel model
steel_tensionchord : Steel with tension stiffening
concrete_tensionchord : Concrete with tension stiffening
popovics_concrete : Popovics nonlinear concrete
KSP_concrete : KSP nonlinear concrete
concrete_EC_softening : Concrete model with softening
"""

from .Material import (
    Material,
    PlaneStress,
    PlaneStrain,
    TimoshenkoMaterial,
    Bilinear_Mat,
    Plastic_Mat,
    Mixed_Hardening_Mat,
    concrete_EC,
    steel_EC,
    steel_tensionchord,
    concrete_tensionchord,
    popovics_concrete,
    KSP_concrete,
    NoTension_Mat,
    Plastic_Stiffness_Deg,
    concrete_EC_softening
)

ConstitutiveLaw = Material

from .Contact import (
    Contact,
    Bilinear,
    NoTension_EP,
    NoTension_CD,
    Coulomb
)

from .Spring import Spring_2D
from .SimplePlaneStressBilinear import SimplePlaneStressBilinear

__all__ = [
    # Base materials
    'Material',
    'ConstitutiveLaw',

    # 2D materials
    'PlaneStress',
    'PlaneStrain',
    'SimplePlaneStressBilinear',

    # Beam materials
    'TimoshenkoMaterial',

    # Nonlinear 1D materials
    'Bilinear_Mat',
    'Plastic_Mat',
    'Mixed_Hardening_Mat',
    'NoTension_Mat',
    'Plastic_Stiffness_Deg',

    # Contact laws
    'Contact',
    'Bilinear',
    'NoTension_EP',
    'NoTension_CD',
    'Coulomb',

    # Spring elements
    'Spring_2D',

    # Specialized models
    'concrete_EC',
    'steel_EC',
    'steel_tensionchord',
    'concrete_tensionchord',
    'popovics_concrete',
    'KSP_concrete',
    'concrete_EC_softening',
]