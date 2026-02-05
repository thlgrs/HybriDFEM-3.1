"""
Analysis panel for solver configuration and execution.

This module provides the analysis control panel for setting up boundary conditions,
loads, and running various analysis types (static, modal, dynamic).
"""

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QMessageBox, QGroupBox, QFormLayout, QComboBox,
                             QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar, QLabel)

from GUI.Dialogs.CouplingConfigDialog import CouplingConfigDialog

# Import backend solvers
try:
    from Core.Solvers import StaticLinear, StaticNonLinear, Modal, Dynamic
    SOLVERS_AVAILABLE = True
except ImportError:
    print("Warning: Unable to import 'Core' solvers.")
    SOLVERS_AVAILABLE = False

    # Create dummy solvers for testing
    class StaticLinear:
        @staticmethod
        def solve(structure, **kwargs):
            print("Calling dummy 'solve' solver")
            import numpy as np
            structure.U = np.random.rand(structure.nb_dofs) * 0.01
            return structure

        @staticmethod
        def solve_augmented(structure, **kwargs):
            print("Calling dummy 'solve_augmented' solver")
            import numpy as np
            structure.U = np.random.rand(structure.nb_dofs) * 0.01
            return structure

    class StaticNonLinear:
        @staticmethod
        def solve_forcecontrol(structure, **kwargs):
            print("Calling dummy 'solve_forcecontrol' solver")
            import numpy as np
            structure.U = np.random.rand(structure.nb_dofs) * 0.1
            return structure

        @staticmethod
        def solve_dispcontrol(structure, **kwargs):
            print("Calling dummy 'solve_dispcontrol' solver")
            import numpy as np
            structure.U = np.random.rand(structure.nb_dofs) * 0.1
            return structure

    class Modal:
        @staticmethod
        def solve(structure, **kwargs):
            print("Calling dummy 'solve_modal' solver")
            import numpy as np
            structure.eig_vals = np.random.rand(kwargs.get('n_modes', 10))
            return structure

    class Dynamic:
        pass


class AnalysisWorker(QThread):
    """
    Worker thread to run analysis without freezing the GUI.
    Emits the solved structure on success or error message on failure.
    """
    finished = pyqtSignal(object)  # Emitted with solved structure on success
    error = pyqtSignal(str)  # Emitted with error message on failure
    log = pyqtSignal(str)  # Emitted for log messages

    def __init__(self, analysis_func, *args, **kwargs):
        super().__init__()
        self.analysis_func = analysis_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        """Execute the analysis function in the thread."""
        try:
            self.log.emit("Starting analysis...")

            # Call solver function (e.g., Static.solve_linear)
            # which must return the solved structure
            solved_structure = self.analysis_func(*self.args, **self.kwargs)

            self.log.emit("Analysis completed successfully!")
            self.finished.emit(solved_structure)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Analysis failed: {e}"
            self.log.emit(f"[ERROR] {error_msg}")
            self.error.emit(error_msg)


class AnalysisPanel(QWidget):
    """Panel for configuring and executing analysis."""

    def __init__(self, project_state, parent=None):
        super().__init__(parent)
        self.project_state = project_state
        self.main_window = parent
        self.analysis_worker = None  # Keep reference to thread
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Analysis Type
        type_group = QGroupBox("Analysis Type")
        type_layout = QVBoxLayout()

        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            "Linear Static",
            "Nonlinear Static (Force Control)",
            "Nonlinear Static (Displacement Control)",
            "Modal Analysis",
            # "Dynamic Linear",  # TODO: add later
            # "Dynamic Nonlinear",  # TODO: add later
        ])
        self.analysis_type.currentTextChanged.connect(self.update_analysis_options)
        type_layout.addWidget(self.analysis_type)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Analysis Parameters
        self.params_group = QGroupBox("Analysis Parameters")
        self.params_layout = QFormLayout()

        # Nonlinear Static Parameters
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 10000)
        self.steps_input.setValue(100)
        self.params_layout.addRow("Number of Steps:", self.steps_input)

        self.tolerance_input = QDoubleSpinBox()
        self.tolerance_input.setRange(1e-10, 1.0)
        self.tolerance_input.setValue(1e-6)
        self.tolerance_input.setDecimals(10)
        self.params_layout.addRow("Tolerance:", self.tolerance_input)

        # Modal Parameters
        self.modes_input = QSpinBox()
        self.modes_input.setRange(1, 100)
        self.modes_input.setValue(10)
        self.params_layout.addRow("Number of Modes:", self.modes_input)

        # Displacement Control Parameters
        self.control_node_input = QSpinBox()
        self.control_node_input.setRange(0, 100000)
        self.control_node_input.setValue(0)
        self.control_node_input.setToolTip("Node ID for displacement control")
        self.params_layout.addRow("Control Node ID:", self.control_node_input)

        self.control_dof_input = QComboBox()
        self.control_dof_input.addItems(["ux (0)", "uy (1)", "rz (2)"])
        self.control_dof_input.setCurrentIndex(1)  # Default to uy
        self.control_dof_input.setToolTip("DOF direction for displacement control")
        self.params_layout.addRow("Control DOF:", self.control_dof_input)

        self.target_disp_input = QDoubleSpinBox()
        self.target_disp_input.setRange(-1.0, 1.0)
        self.target_disp_input.setValue(-0.01)
        self.target_disp_input.setDecimals(6)
        self.target_disp_input.setSingleStep(0.001)
        self.target_disp_input.setToolTip("Target displacement in meters (positive or negative)")
        self.params_layout.addRow("Target Disp. [m]:", self.target_disp_input)

        self.max_iter_input = QSpinBox()
        self.max_iter_input.setRange(1, 1000)
        self.max_iter_input.setValue(25)
        self.max_iter_input.setToolTip("Maximum Newton-Raphson iterations per step")
        self.params_layout.addRow("Max Iterations:", self.max_iter_input)

        # Geometry Options
        self.linear_geom = QCheckBox("Linear Geometry")
        self.linear_geom.setChecked(True)
        self.params_layout.addRow("", self.linear_geom)

        self.params_group.setLayout(self.params_layout)
        layout.addWidget(self.params_group)

        # Hybrid Coupling Configuration
        coupling_group = QGroupBox("Hybrid Coupling (for Block+FEM structures)")
        coupling_layout = QVBoxLayout()

        # Coupling status display
        self.coupling_status = QLabel("Not configured")
        self.coupling_status.setStyleSheet("color: gray;")
        coupling_layout.addWidget(self.coupling_status)

        # Configure button
        self.btn_configure_coupling = QPushButton("Configure Coupling...")
        self.btn_configure_coupling.clicked.connect(self.open_coupling_dialog)
        coupling_layout.addWidget(self.btn_configure_coupling)

        coupling_group.setLayout(coupling_layout)
        layout.addWidget(coupling_group)

        # Connect to structure changed signal to update coupling status
        self.project_state.structure_changed.connect(self.update_coupling_status)

        # Run Analysis Button
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_run.clicked.connect(self.run_analysis)
        layout.addWidget(self.btn_run)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate mode (busy)
        layout.addWidget(self.progress_bar)

        layout.addStretch()
        self.setLayout(layout)

        self.update_analysis_options()

    def update_analysis_options(self):
        """Show/hide parameters based on analysis type."""
        analysis = self.analysis_type.currentText()

        # Hide all specific parameters
        self.steps_input.parent().setVisible(False)
        self.tolerance_input.parent().setVisible(False)
        self.modes_input.parent().setVisible(False)
        self.control_node_input.parent().setVisible(False)
        self.control_dof_input.parent().setVisible(False)
        self.target_disp_input.parent().setVisible(False)
        self.max_iter_input.parent().setVisible(False)

        # Show relevant parameters
        if analysis == "Linear Static":
            self.linear_geom.parent().setVisible(True)

        elif analysis == "Nonlinear Static (Force Control)":
            self.steps_input.parent().setVisible(True)
            self.tolerance_input.parent().setVisible(True)
            self.linear_geom.parent().setVisible(True)

        elif analysis == "Nonlinear Static (Displacement Control)":
            self.steps_input.parent().setVisible(True)
            self.tolerance_input.parent().setVisible(True)
            self.control_node_input.parent().setVisible(True)
            self.control_dof_input.parent().setVisible(True)
            self.target_disp_input.parent().setVisible(True)
            self.max_iter_input.parent().setVisible(True)
            self.linear_geom.parent().setVisible(True)

        elif "Modal" in analysis:
            self.modes_input.parent().setVisible(True)
            self.linear_geom.parent().setVisible(True)  # Modal analysis is linear

    def open_coupling_dialog(self):
        """Open dialog to configure hybrid coupling."""
        structure = self.project_state.structure
        dialog = CouplingConfigDialog(structure=structure, parent=self.main_window)

        if dialog.exec():
            data = dialog.get_data()
            method = data['method']
            params = data['params']

            # Configure coupling in project state
            success = self.project_state.configure_coupling(method, params)

            if success:
                self.update_coupling_status()
                QMessageBox.information(
                    self,
                    "Coupling Configured",
                    f"Hybrid coupling configured: {method.upper()}\n\n"
                    f"Coupling will be applied when analysis is run."
                )

    def update_coupling_status(self):
        """Update the coupling status label based on structure and configuration."""
        structure = self.project_state.structure

        # Check if structure is hybrid
        is_hybrid = (structure and
                    hasattr(structure, 'list_blocks') and
                    hasattr(structure, 'list_fes') and
                    len(getattr(structure, 'list_blocks', [])) > 0 and
                    len(getattr(structure, 'list_fes', [])) > 0)

        if not is_hybrid:
            self.coupling_status.setText("Not applicable (structure not hybrid)")
            self.coupling_status.setStyleSheet("color: gray;")
            self.btn_configure_coupling.setEnabled(False)
            return

        # Structure is hybrid, check if coupling configured
        self.btn_configure_coupling.setEnabled(True)

        if self.project_state.coupling_enabled:
            method = self.project_state.coupling_method.upper()
            params_str = ", ".join([f"{k}={v}" for k, v in self.project_state.coupling_params.items()])
            status_text = f"Configured: {method} ({params_str})"
            self.coupling_status.setText(status_text)
            self.coupling_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.coupling_status.setText("Not configured (will use default: CONSTRAINT)")
            self.coupling_status.setStyleSheet("color: orange;")

    def run_analysis(self):
        # 1. Check if structure exists
        structure = self.project_state.structure
        if not structure:
            QMessageBox.warning(self, "Warning",
                                "No structure loaded. Please import or create geometry first.")
            return

        # 2. Prepare structure (apply boundary conditions, etc.)
        try:
            # Ensure nodes are generated (just in case)
            if not structure.nb_dofs:
                structure.make_nodes()

            # Apply GUI options to structure
            structure.set_lin_geom(self.linear_geom.isChecked())

            # Re-apply boundary conditions (in case structure was regenerated)
            self.project_state.log_message.emit("Preparing structure for analysis...")
            for node_id, dofs in self.project_state.supports.items():
                structure.fix_node(node_id, dofs)
            for node_id, load_list in self.project_state.loads.items():
                for dof_index, value in load_list:
                    structure.load_node(node_id, [dof_index], value)

            # SAFETY CHECK: Ensure block structures have contact interfaces
            if hasattr(structure, 'list_blocks') and hasattr(structure, 'detect_interfaces'):
                if hasattr(structure, 'list_cfs') and len(structure.list_cfs) == 0:
                    self.project_state.log_message.emit(
                        "Warning: No contact interfaces detected. Running detect_interfaces()...")
                    structure.detect_interfaces()
                    self.project_state.log_message.emit(f"Detected {len(structure.list_cfs)} contact faces.")

            # Apply hybrid coupling if configured
            self.project_state.apply_coupling_to_structure()

            # Apply contact configuration if configured
            self.project_state.apply_contact_to_structure()

            self.project_state.log_message.emit(f"Structure ready: {structure.nb_dofs} DOFs.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Structure preparation failed: {e}")
            self.project_state.log_message.emit(f"[ERROR] Structure Preparation: {e}")
            return

        # 3. Disable UI and show progress
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)

        # 4. Select solver and arguments
        analysis_type = self.analysis_type.currentText()
        solver_function = None
        args = [structure]
        kwargs = {}

        # Check if we need saddle point solver (Lagrange or Mortar coupling)
        use_saddle_point = (self.project_state.coupling_enabled and
                           self.project_state.coupling_method in ['lagrange', 'mortar'])

        if analysis_type == "Linear Static":
            if use_saddle_point:
                solver_function = StaticLinear.solve_augmented
                self.project_state.log_message.emit(
                    f"Using augmented solver for {self.project_state.coupling_method.upper()} coupling")
            else:
                solver_function = StaticLinear.solve
                kwargs["optimized"] = True

        elif analysis_type == "Nonlinear Static (Force Control)":
            # Note: Nonlinear + saddle point not typically combined
            # For now, use standard nonlinear solver
            if use_saddle_point:
                self.project_state.log_message.emit(
                    "[WARNING] Nonlinear analysis with Lagrange/Mortar coupling not fully supported. "
                    "Using standard solver.")
            solver_function = StaticNonLinear.solve_forcecontrol
            kwargs["n_steps"] = self.steps_input.value()
            kwargs["tol"] = self.tolerance_input.value()

        elif analysis_type == "Nonlinear Static (Displacement Control)":
            # Displacement control analysis
            if use_saddle_point:
                self.project_state.log_message.emit(
                    "[WARNING] Nonlinear analysis with Lagrange/Mortar coupling not fully supported. "
                    "Using standard solver.")

            # Extract control DOF from dropdown (format: "ux (0)", "uy (1)", "rz (2)")
            dof_text = self.control_dof_input.currentText()
            control_dof = int(dof_text.split("(")[1].split(")")[0])  # Extract number from "(X)"

            solver_function = StaticNonLinear.solve_dispcontrol
            kwargs["n_steps"] = self.steps_input.value()
            kwargs["target_disp"] = self.target_disp_input.value()
            kwargs["control_node"] = self.control_node_input.value()
            kwargs["control_dof"] = control_dof
            kwargs["tol"] = self.tolerance_input.value()
            kwargs["max_iter"] = self.max_iter_input.value()

            self.project_state.log_message.emit(
                f"Displacement Control: node={kwargs['control_node']}, DOF={control_dof}, "
                f"target={kwargs['target_disp']*1000:.2f}mm in {kwargs['n_steps']} steps"
            )

        elif analysis_type == "Modal Analysis":
            if use_saddle_point:
                self.project_state.log_message.emit(
                    "[WARNING] Modal analysis with Lagrange/Mortar coupling not fully supported. "
                    "Using standard solver.")
            solver_function = Modal.solve
            kwargs["n_modes"] = self.modes_input.value()

        else:
            QMessageBox.warning(self, "Warning", f"Analysis '{analysis_type}' not implemented.")
            self.on_analysis_done()  # Re-enable UI
            return

        # 5. Create and start worker
        self.analysis_worker = AnalysisWorker(solver_function, *args, **kwargs)

        # 6. Connect worker signals
        self.analysis_worker.log.connect(self.project_state.log_message)
        self.analysis_worker.error.connect(self.on_analysis_error)

        # Most important connection:
        # When worker finishes, call ProjectState.set_analysis_results
        self.analysis_worker.finished.connect(self.project_state.set_analysis_results)

        # When thread is done (success or failure), re-enable UI
        self.analysis_worker.finished.connect(self.on_analysis_done)
        self.analysis_worker.error.connect(self.on_analysis_done)

        self.analysis_worker.start()  # Start analysis

    def on_analysis_done(self):
        """Slot to re-enable UI after analysis."""
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.analysis_worker = None  # Release reference

    def on_analysis_error(self, error_msg):
        """Slot called when analysis fails."""
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed: {error_msg}")
