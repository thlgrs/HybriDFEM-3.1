"""
Legacy HybriDFEM Implementation (DEPRECATED)

⚠️ WARNING: This package contains the old HybriDFEM implementation and is
DEPRECATED. It is kept for reference and backward compatibility only.

**DO NOT USE FOR NEW PROJECTS!**

For new projects, use the 'Core' package instead:
    from Core import Hybrid, Static
    from Core.Objects import Triangle3, Block_2D

What's Here
-----------
This Legacy package contains:
- Legacy/Objects/: Old implementation of Structure, Block, Contact, etc.
- Legacy/Examples/: Historical example scripts (many may not work with current code)

Migration Guide
---------------
Old (Legacy):
    from Legacy.Objects.Structure import Structure
    from Legacy.Objects.Block import Block_2D

New (Core):
    from Core.Structures import Structure_FEM, Structure_Block, Hybrid
    from Core.Objects.DFEM import Block_2D

Why Deprecated?
---------------
1. **Better Architecture**: Core has cleaner separation of concerns
2. **Variable DOF Support**: Core supports mixed DOF structures (blocks + FEM)
3. **More Features**: Hybrid coupling, mortar method, etc.
4. **Better Documentation**: Comprehensive docstrings and examples
5. **Active Development**: All new features go into Core

Backward Compatibility
----------------------
This package is kept to:
- Allow old scripts to run (with warnings)
- Serve as reference for migration
- Maintain historical examples

When imported, a deprecation warning will be issued.

Version: 2.0 (Legacy/Frozen)
Status: Deprecated - Use 'Core' package instead
"""

import warnings

# Issue deprecation warning when Legacy is imported
warnings.warn(
    "\n"
    "╔════════════════════════════════════════════════════════════════╗\n"
    "║  ⚠️  DEPRECATION WARNING: Legacy Package                       ║\n"
    "╠════════════════════════════════════════════════════════════════╣\n"
    "║  You are importing from the deprecated 'Legacy' package.       ║\n"
    "║  This code is kept for reference only.                         ║\n"
    "║                                                                ║\n"
    "║  For new projects, use the 'Core' package instead:             ║\n"
    "║    from Core import Hybrid, Static                             ║\n"
    "║    from Core.Objects import Triangle3, Block_2D                ║\n"
    "║                                                                ║\n"
    "║  See docs/MIGRATION_GUIDE.md for migration instructions.       ║\n"
    "╚════════════════════════════════════════════════════════════════╝\n",
    DeprecationWarning,
    stacklevel=2
)

__version__ = '2.0.0-legacy'
__status__ = 'deprecated'

# Import subpackages (but still warn users)
from Legacy import Objects

__all__ = [
    '__version__',
    '__status__',
    'Objects',
]
