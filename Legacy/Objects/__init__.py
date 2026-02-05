"""
Legacy Objects Module (DEPRECATED)

WARNING: This module contains the old object implementations and is
DEPRECATED. Use the 'Core.Objects' package instead.

Old Classes (Legacy)
--------------------
Structure : Old structure class
    - Use Core.Structures.Structure_FEM or Structure_Block instead

Block_2D : Old block implementation
    - Use Core.Objects.DFEM.Block_2D instead

Contact : Old contact laws
    - Use Core.Objects.ConstitutiveLaw.Contact, Coulomb, etc. instead

ConstitutiveLaw : Old material models
    - Use Core.Objects.ConstitutiveLaw.PlaneStress, PlaneStrain, etc. instead

Migration Examples
------------------
Old (Legacy):
    from Legacy.Objects.Structure import Structure
    from Legacy.Objects.Block import Block_2D

New (Core):
    from Core.Structures import Structure_Block
    from Core.Objects.DFEM import Block_2D

Status: Deprecated - Use 'Core.Objects' instead
"""

import warnings

warnings.warn(
    "Legacy.Objects is deprecated. Use 'Core.Objects' instead.",
    DeprecationWarning,
    stacklevel=2
)

__version__ = '2.0.0-legacy'
__status__ = 'deprecated'

# Note: Old classes can still be imported if needed,
# but users should migrate to Core package
__all__ = []
