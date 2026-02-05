"""
GUI Dialogs

This package contains dialog windows for user input in the HybriDFEM GUI.

Dialogs
-------
AddBlockDialog : Create rectangular blocks with dimensions
AddMaterialDialog : Define material properties (E, nu, rho)
AddNodalLoadDialog : Apply force vectors to nodes
AddSupportDialog : Fix DOFs at nodes (boundary conditions)
"""

from .AddBlockDialog import AddBlockDialog
from .AddMaterialDialog import AddMaterialDialog
from .AddNodalLoadDialog import AddNodalLoadDialog
from .AddSupportDialog import AddSupportDialog

__all__ = [
    'AddBlockDialog',
    'AddMaterialDialog',
    'AddNodalLoadDialog',
    'AddSupportDialog',
]
