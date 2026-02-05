"""
Legacy Examples (DEPRECATED)

⚠️ WARNING: These examples use the old Legacy implementation and may not work
with the current codebase. They are kept for historical reference only.

**For working examples, see the 'Examples/' directory at the project root!**

What's Here
-----------
This directory contains historical examples organized by topic:

- Reinforced Concrete/: RC beam examples (tension chord, localization, etc.)
- Linear_Dynamic/: Dynamic analysis examples (cantilever, crane, etc.)
- Modal Analysis/: Modal analysis examples
- Nonlinear_Dynamic/: Nonlinear dynamic examples
- F&TL Walls/: Ferris & Tin-Loi wall examples
- IABSE_2024/, COMPDYN_2023/, etc.: Conference paper examples

Status
------
Many of these examples:
❌ May use deprecated APIs
❌ May not run with current code
❌ Are not actively maintained

Recommended Alternatives
------------------------
For working, up-to-date examples, see:

📁 Examples/Structure_FEM/
   - example_beam.py: Cantilever beam with triangular elements

📁 Examples/Structure_Hybrid/
   - example_beam_on_columns.py: Hybrid structure with constraint coupling
   - example_beam_on_columns_penalty.py: Hybrid with penalty coupling
   - example_beam_on_columns_lagrange.py: Hybrid with Lagrange multipliers
   - example_mortar_coupling.py: Hybrid with mortar method (non-matching meshes)
   - compare_coupling_methods.py: Compare all 4 coupling methods

📁 Examples/Structure_Block/
   - example_column.py: Stacked block column

These examples use the modern 'Core' package and are actively maintained!

Purpose
-------
Legacy examples are preserved for:
1. Historical reference
2. Migration guidance
3. Verification of results
4. Academic record keeping

Note: If you need to run these examples, you may need to:
- Update imports to use Core package
- Adjust API calls to match current interface
- Refer to migration guide: docs/MIGRATION_GUIDE.md

Status: Deprecated - Use 'Examples/' directory instead
"""

import warnings

warnings.warn(
    "Legacy examples are deprecated and may not work. "
    "Use examples in the 'Examples/' directory instead (project root).",
    DeprecationWarning,
    stacklevel=2
)

__version__ = '2.0.0-legacy'
__status__ = 'deprecated-examples'

# No imports - these are scripts, not a package
__all__ = []
