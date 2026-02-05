"""
HybriDFEM Graphical User Interface

PyQt6-based GUI for interactive structural modeling and analysis.

Main Components
---------------
MainWindow : Main application window
    - Central viewport for visualization
    - Tabbed control panels
    - Project state management

ProjectState : Centralized state manager
    - Structure data
    - Materials library
    - Analysis results
    - Inter-panel communication via Qt signals

ViewportWidget : Matplotlib-based visualization
    - Interactive structure display
    - Node selection for loads/supports
    - Deformed shape visualization

Control Panels
--------------
GeometryPanel : Geometry definition
    - Rhino import
    - Manual block creation
    - Geometry information display

MaterialPanel : ConstitutiveLaw management
    - ConstitutiveLaw library
    - Add/edit/delete materials
    - ConstitutiveLaw assignment

AnalysisPanel : Analysis setup and execution
    - Boundary conditions
    - Load application
    - Solver configuration
    - Run analysis

ResultsPanel : Post-processing
    - Displacement results
    - Deformation visualization
    - Export capabilities

Dialogs
-------
AddBlockDialog : Create rectangular blocks
AddMaterialDialog : Define material properties
AddNodalLoadDialog : Apply nodal forces
AddSupportDialog : Fix DOFs at nodes

Utilities
---------
launch_gui : GUI launcher script
rhino_integration : Rhino geometry import

Usage
-----
To launch the GUI:

    python GUI/Utils/launch_gui.py

Or programmatically:

    >>> from PyQt6.QtWidgets import QApplication
    >>> from GUI import HybridFEMMainWindow
    >>>
    >>> app = QApplication(sys.argv)
    >>> window = HybridFEMMainWindow()
    >>> window.show()
    >>> sys.exit(app.exec())
"""

__version__ = '1.0.0'

# Dialogs (available for advanced use)
from .Dialogs.AddBlockDialog import AddBlockDialog
from .Dialogs.AddMaterialDialog import AddMaterialDialog
from .Dialogs.AddNodalLoadDialog import AddNodalLoadDialog
from .Dialogs.AddSupportDialog import AddSupportDialog
# Main window
from .MainWindow import HybridFEMMainWindow
from .Panels.AnalysisPanel import AnalysisPanel
# Panels (available for advanced use)
from .Panels.GeometryPanel import GeometryPanel
from .Panels.MaterialPanel import MaterialPanel
from .Panels.ResultsPanel import ResultsPanel
# Core components
from .ProjectState import ProjectState
from .ViewportWidget import ViewportWidget

__all__ = [
    # Version
    '__version__',

    # Main window (primary entry point)
    'HybridFEMMainWindow',

    # Core components
    'ProjectState',
    'ViewportWidget',

    # Panels
    'GeometryPanel',
    'MaterialPanel',
    'AnalysisPanel',
    'ResultsPanel',

    # Dialogs
    'AddBlockDialog',
    'AddMaterialDialog',
    'AddNodalLoadDialog',
    'AddSupportDialog',
]
