# GUI/MainWindow.py
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QDockWidget, QMessageBox, QTabWidget, QFileDialog)

from GUI.Panels.AnalysisPanel import AnalysisPanel
from GUI.Panels.GeometryPanel import GeometryPanel
from GUI.Panels.MaterialPanel import MaterialPanel
from GUI.Panels.ResultsPanel import ResultsPanel
from GUI.ProjectState import ProjectState
from GUI.ViewportWidget import ViewportWidget


class HybridFEMMainWindow(QMainWindow):
    """Main application window for HybridFEM GUI."""

    def __init__(self):
        super().__init__()

        # Create centralized state manager
        self.project_state = ProjectState()

        # Initialize UI components
        self.init_ui()

        # Connect global signals
        self.project_state.log_message.connect(self.log_message_slot)

    def init_ui(self):
        self.setWindowTitle("HybridFEM")
        self.setGeometry(100, 50, 1200, 750)

        # Set window icon
        from pathlib import Path
        icon_path = Path(__file__).parent / "Resources" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Central viewport for visualization
        self.viewport = ViewportWidget(self.project_state, self)
        self.setCentralWidget(self.viewport)

        # Left dock widget for control panels
        self.control_dock = QDockWidget("Control Panel", self)
        self.control_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        # Tabbed panels for different functionalities
        self.panel_tabs = QTabWidget()

        self.geometry_panel = GeometryPanel(self.project_state, self)
        self.panel_tabs.addTab(self.geometry_panel, "Geometry")

        self.material_panel = MaterialPanel(self.project_state, self)
        self.panel_tabs.addTab(self.material_panel, "Materials & BCs")

        self.analysis_panel = AnalysisPanel(self.project_state, self)
        self.panel_tabs.addTab(self.analysis_panel, "Analysis")

        self.results_panel = ResultsPanel(self.project_state, self)
        self.panel_tabs.addTab(self.results_panel, "Results")

        self.control_dock.setWidget(self.panel_tabs)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.control_dock)

        # Bottom dock widget for console log
        self.log_dock = QDockWidget("Console Log", self)
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)

        self.console_log = QTextEdit()
        self.console_log.setReadOnly(True)
        self.log_dock.setWidget(self.console_log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

        self.setup_menu_bar()
        self.log_message_slot("HybridFEM initialized")

    def setup_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        new_action = file_menu.addAction("New Project")
        new_action.triggered.connect(self.project_state.new_project)

        file_menu.addSeparator()

        open_action = file_menu.addAction("Open Project...")
        open_action.triggered.connect(self.open_project)
        open_action.setShortcut("Ctrl+O")

        save_action = file_menu.addAction("Save Project")
        save_action.triggered.connect(self.save_project)
        save_action.setShortcut("Ctrl+S")

        save_as_action = file_menu.addAction("Save Project As...")
        save_as_action.triggered.connect(self.save_project_as)
        save_as_action.setShortcut("Ctrl+Shift+S")

        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        help_menu = menubar.addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

    def log_message_slot(self, message):
        """Display message in console log and auto-scroll to bottom."""
        self.console_log.append(f"> {message}")
        self.console_log.verticalScrollBar().setValue(
            self.console_log.verticalScrollBar().maximum()
        )

    def show_about(self):
        QMessageBox.about(
            self, "About HybridFEM",
            "HybridFEM - Structural Analysis Tool\n\n"
            "Architecture GUI v2."
        )

    def open_project(self):
        """Open project from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "HybridFEM Project (*.hfem);;JSON Files (*.json);;All Files (*.*)"
        )

        if file_path:
            success = self.project_state.load_project(file_path)
            if success:
                QMessageBox.information(
                    self,
                    "Project Loaded",
                    f"Project loaded successfully from:\n{file_path}\n\n"
                    "Note: Geometry must be recreated manually."
                )
            else:
                QMessageBox.critical(
                    self,
                    "Load Failed",
                    f"Failed to load project from:\n{file_path}\n\n"
                    "Check console for error details."
                )

    def save_project(self):
        """Save project to current file or prompt for location."""
        if self.project_state.project_file_path:
            # Save to existing file
            success = self.project_state.save_project(self.project_state.project_file_path)
            if not success:
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    "Failed to save project. Check console for error details."
                )
        else:
            # No current file, prompt for save location
            self.save_project_as()

    def save_project_as(self):
        """Save project to a new file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            "",
            "HybridFEM Project (*.hfem);;JSON Files (*.json);;All Files (*.*)"
        )

        if file_path:
            # Add extension if not present
            if not file_path.endswith(('.hfem', '.json')):
                file_path += '.hfem'

            success = self.project_state.save_project(file_path)
            if success:
                QMessageBox.information(
                    self,
                    "Project Saved",
                    f"Project saved successfully to:\n{file_path}"
                )
            else:
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    "Failed to save project. Check console for error details."
                )


def main():
    app = QApplication(sys.argv)

    # Set application-wide icon (taskbar, dock, etc.)
    from pathlib import Path
    icon_path = Path(__file__).parent / "Resources" / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = HybridFEMMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()