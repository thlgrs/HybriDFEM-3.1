"""
Dialog for applying support boundary conditions.

This module provides a dialog for fixing DOFs (x, y, rotation)
at selected nodes to define structural supports.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout,
                             QDialogButtonBox, QCheckBox, QLabel)


class AddSupportDialog(QDialog):
    """Dialog for applying support boundary conditions to a node."""

    def __init__(self, node_id, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Add Support to Node {node_id}")

        self.node_id = node_id

        # Create checkboxes
        self.fix_ux = QCheckBox("Fix Translation X (ux)")
        self.fix_uy = QCheckBox("Fix Translation Y (uy)")
        self.fix_rz = QCheckBox("Fix Rotation Z (rz)")

        # Set ux and uy checked by default (simple support)
        self.fix_ux.setChecked(True)
        self.fix_uy.setChecked(True)

        # OK / Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                      QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Layout
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        form_layout.addRow(QLabel(f"Selected node: {self.node_id}"))
        form_layout.addRow(self.fix_ux)
        form_layout.addRow(self.fix_uy)
        form_layout.addRow(self.fix_rz)

        layout.addLayout(form_layout)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def get_data(self):
        """Retrieve list of DOFs to fix."""
        dofs_to_fix = []
        if self.fix_ux.isChecked():
            dofs_to_fix.append(0)  # 0 = ux
        if self.fix_uy.isChecked():
            dofs_to_fix.append(1)  # 1 = uy
        if self.fix_rz.isChecked():
            dofs_to_fix.append(2)  # 2 = rz

        return {
            "node_id": self.node_id,
            "dofs": dofs_to_fix
        }
