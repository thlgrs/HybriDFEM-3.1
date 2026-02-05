"""
Dialog for configuring hybrid coupling between blocks and FEM elements.

This module provides a dialog for selecting and configuring the coupling method
for hybrid structures (blocks + FEM). Supports 4 methods:
- Constraint (DOF elimination)
- Penalty (stiffness springs)
- Lagrange (nodal multipliers)
- Mortar (distributed multipliers)
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QDialogButtonBox, QDoubleSpinBox, QLabel, QComboBox,
                             QGroupBox, QRadioButton, QButtonGroup, QTextEdit, QSpinBox)
from PyQt6.QtCore import Qt


class CouplingConfigDialog(QDialog):
    """Dialog for configuring hybrid structure coupling."""

    def __init__(self, structure=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Hybrid Coupling")
        self.structure = structure
        self.resize(600, 550)

        # Check if structure is hybrid
        if not structure or not hasattr(structure, 'list_blocks') or not hasattr(structure, 'list_fes'):
            self.is_hybrid = False
        else:
            self.is_hybrid = len(structure.list_blocks) > 0 and len(structure.list_fes) > 0

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Warning if not hybrid
        if not self.is_hybrid:
            warning = QLabel("⚠️ Structure is not hybrid (requires both blocks and FEM elements).\n"
                           "Coupling configuration will not be applied.")
            warning.setStyleSheet("color: orange; font-weight: bold;")
            warning.setWordWrap(True)
            layout.addWidget(warning)

        # Method Selection
        method_group = QGroupBox("Coupling Method")
        method_layout = QVBoxLayout()

        self.method_buttons = QButtonGroup()

        self.constraint_radio = QRadioButton("Constraint (DOF Elimination)")
        self.penalty_radio = QRadioButton("Penalty (Stiffness Springs)")
        self.lagrange_radio = QRadioButton("Lagrange (Nodal Multipliers)")
        self.mortar_radio = QRadioButton("Mortar (Distributed Multipliers)")

        self.method_buttons.addButton(self.constraint_radio, 0)
        self.method_buttons.addButton(self.penalty_radio, 1)
        self.method_buttons.addButton(self.lagrange_radio, 2)
        self.method_buttons.addButton(self.mortar_radio, 3)

        self.constraint_radio.setChecked(True)  # Default

        method_layout.addWidget(self.constraint_radio)
        method_layout.addWidget(self.penalty_radio)
        method_layout.addWidget(self.lagrange_radio)
        method_layout.addWidget(self.mortar_radio)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Method Description
        self.method_description = QTextEdit()
        self.method_description.setReadOnly(True)
        self.method_description.setMaximumHeight(120)
        layout.addWidget(self.method_description)

        # Constraint Parameters
        self.constraint_group = QGroupBox("Constraint Parameters")
        constraint_layout = QFormLayout()

        self.constraint_tolerance = QDoubleSpinBox()
        self.constraint_tolerance.setRange(1e-12, 1e-3)
        self.constraint_tolerance.setValue(1e-9)
        self.constraint_tolerance.setDecimals(12)
        self.constraint_tolerance.setSuffix(" m")
        constraint_layout.addRow(QLabel("Node Detection Tolerance:"), self.constraint_tolerance)

        self.constraint_group.setLayout(constraint_layout)
        layout.addWidget(self.constraint_group)

        # Penalty Parameters
        self.penalty_group = QGroupBox("Penalty Parameters")
        penalty_layout = QFormLayout()

        self.penalty_mode = QComboBox()
        self.penalty_mode.addItems(['Auto', 'Manual'])
        self.penalty_mode.currentTextChanged.connect(self.on_penalty_mode_changed)
        penalty_layout.addRow(QLabel("Penalty Factor:"), self.penalty_mode)

        self.penalty_value = QDoubleSpinBox()
        self.penalty_value.setRange(1e6, 1e15)
        self.penalty_value.setValue(1e12)
        self.penalty_value.setDecimals(0)
        self.penalty_value.setSingleStep(1e11)
        self.penalty_value.setEnabled(False)  # Initially disabled (Auto mode)
        penalty_layout.addRow(QLabel("Manual Value:"), self.penalty_value)

        self.penalty_group.setLayout(penalty_layout)
        layout.addWidget(self.penalty_group)

        # Lagrange Parameters
        self.lagrange_group = QGroupBox("Lagrange Parameters")
        lagrange_layout = QFormLayout()

        self.lagrange_tolerance = QDoubleSpinBox()
        self.lagrange_tolerance.setRange(1e-12, 1e-3)
        self.lagrange_tolerance.setValue(1e-9)
        self.lagrange_tolerance.setDecimals(12)
        self.lagrange_tolerance.setSuffix(" m")
        lagrange_layout.addRow(QLabel("Node Detection Tolerance:"), self.lagrange_tolerance)

        self.lagrange_group.setLayout(lagrange_layout)
        layout.addWidget(self.lagrange_group)

        # Mortar Parameters
        self.mortar_group = QGroupBox("Mortar Parameters")
        mortar_layout = QFormLayout()

        self.mortar_order = QSpinBox()
        self.mortar_order.setRange(1, 3)
        self.mortar_order.setValue(2)
        mortar_layout.addRow(QLabel("Integration Order:"), self.mortar_order)

        self.mortar_tolerance = QDoubleSpinBox()
        self.mortar_tolerance.setRange(1e-6, 1.0)
        self.mortar_tolerance.setValue(0.01)
        self.mortar_tolerance.setDecimals(4)
        self.mortar_tolerance.setSuffix(" m")
        mortar_layout.addRow(QLabel("Interface Detection Tolerance:"), self.mortar_tolerance)

        self.mortar_group.setLayout(mortar_layout)
        layout.addWidget(self.mortar_group)

        # Status/Preview
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(80)
        layout.addWidget(QLabel("Coupling Status:"))
        layout.addWidget(self.status_text)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                      QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Connect signals
        self.constraint_radio.toggled.connect(self.update_ui)
        self.penalty_radio.toggled.connect(self.update_ui)
        self.lagrange_radio.toggled.connect(self.update_ui)
        self.mortar_radio.toggled.connect(self.update_ui)

        # Initialize UI
        self.update_ui()
        self.update_status()

    def on_penalty_mode_changed(self, mode):
        """Enable/disable manual penalty value input."""
        self.penalty_value.setEnabled(mode == 'Manual')

    def update_ui(self):
        """Show/hide parameter groups based on selected method."""
        # Hide all groups first
        self.constraint_group.setVisible(False)
        self.penalty_group.setVisible(False)
        self.lagrange_group.setVisible(False)
        self.mortar_group.setVisible(False)

        # Show relevant group
        if self.constraint_radio.isChecked():
            self.constraint_group.setVisible(True)
            self.update_method_description('constraint')
        elif self.penalty_radio.isChecked():
            self.penalty_group.setVisible(True)
            self.update_method_description('penalty')
        elif self.lagrange_radio.isChecked():
            self.lagrange_group.setVisible(True)
            self.update_method_description('lagrange')
        elif self.mortar_radio.isChecked():
            self.mortar_group.setVisible(True)
            self.update_method_description('mortar')

    def update_method_description(self, method):
        """Update the method description text."""
        descriptions = {
            'constraint': (
                "<b>Constraint (DOF Elimination)</b><br>"
                "• <b>Fastest</b> method (smallest system)<br>"
                "• <b>Exact</b> constraint enforcement<br>"
                "• Requires <b>matching meshes</b> (FEM nodes at block reference points)<br>"
                "• No parameter tuning needed<br>"
                "• Uses standard linear solver<br>"
                "• <b>Best for production</b> when meshes align"
            ),
            'penalty': (
                "<b>Penalty (Stiffness Springs)</b><br>"
                "• Simple implementation<br>"
                "• <b>Approximate</b> constraint enforcement (violation ∝ 1/penalty)<br>"
                "• Requires <b>matching meshes</b><br>"
                "• Requires parameter tuning (penalty factor)<br>"
                "• Uses standard linear solver<br>"
                "• <b>Best for prototyping</b> and soft constraints"
            ),
            'lagrange': (
                "<b>Lagrange (Nodal Multipliers)</b><br>"
                "• <b>Exact</b> constraint enforcement<br>"
                "• Direct access to <b>interface forces</b> (multipliers)<br>"
                "• Requires <b>matching meshes</b><br>"
                "• No parameter tuning needed<br>"
                "• Uses saddle point solver (LDLT)<br>"
                "• <b>Best for interface analysis</b> (need forces)"
            ),
            'mortar': (
                "<b>Mortar (Distributed Multipliers)</b><br>"
                "• <b>Exact</b> constraint enforcement (weak form)<br>"
                "• Works with <b>non-matching meshes</b> (unique!)<br>"
                "• Distributed interface forces (physically accurate)<br>"
                "• Requires FEM mesh to span block interfaces<br>"
                "• Uses saddle point solver (LDLT)<br>"
                "• <b>Best for complex geometries</b> and multi-scale modeling"
            )
        }
        self.method_description.setHtml(descriptions.get(method, ""))

    def update_status(self):
        """Update the coupling status preview."""
        if not self.is_hybrid or not self.structure:
            self.status_text.setText("Not a hybrid structure. Coupling not applicable.")
            return

        n_blocks = len(self.structure.list_blocks) if hasattr(self.structure, 'list_blocks') else 0
        n_fes = len(self.structure.list_fes) if hasattr(self.structure, 'list_fes') else 0
        n_nodes = len(self.structure.list_nodes) if hasattr(self.structure, 'list_nodes') else 0

        status = f"Structure has {n_blocks} blocks and {n_fes} FEM elements ({n_nodes} nodes).\n"

        # Try to estimate coupled nodes (simplified)
        if n_nodes > 0:
            status += f"Coupling will be applied when analysis is run."
        else:
            status += "Call make_nodes() before applying coupling."

        self.status_text.setText(status)

    def get_data(self):
        """
        Return coupling configuration as dict.
        """
        if self.constraint_radio.isChecked():
            method = 'constraint'
            params = {
                'tolerance': self.constraint_tolerance.value()
            }
        elif self.penalty_radio.isChecked():
            method = 'penalty'
            penalty_mode = self.penalty_mode.currentText().lower()
            params = {
                'penalty': penalty_mode if penalty_mode == 'auto' else self.penalty_value.value()
            }
        elif self.lagrange_radio.isChecked():
            method = 'lagrange'
            params = {
                'tolerance': self.lagrange_tolerance.value()
            }
        elif self.mortar_radio.isChecked():
            method = 'mortar'
            params = {
                'integration_order': self.mortar_order.value(),
                'interface_tolerance': self.mortar_tolerance.value()
            }
        else:
            method = 'constraint'
            params = {'tolerance': 1e-9}

        return {
            'method': method,
            'params': params
        }
