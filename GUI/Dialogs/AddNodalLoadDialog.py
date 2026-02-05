"""
Dialog for applying nodal loads.

This module provides a dialog for applying force vectors
to selected nodes in the structure.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout,
                             QDialogButtonBox, QDoubleSpinBox, QLabel)


class AddNodalLoadDialog(QDialog):
    """Dialog for applying nodal loads."""

    def __init__(self, node_id, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Add Load to Node {node_id}")

        self.node_id = node_id

        # Input widgets
        self.load_fx = QDoubleSpinBox()
        self.load_fx.setRange(-1e9, 1e9)
        self.load_fx.setValue(0.0)
        self.load_fx.setSuffix(" N")

        self.load_fy = QDoubleSpinBox()
        self.load_fy.setRange(-1e9, 1e9)
        self.load_fy.setValue(-1000.0)  # Default: 1kN downward
        self.load_fy.setSuffix(" N")

        self.load_mz = QDoubleSpinBox()
        self.load_mz.setRange(-1e9, 1e9)
        self.load_mz.setValue(0.0)
        self.load_mz.setSuffix(" Nm")

        # OK / Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                      QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Layout
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        form_layout.addRow(QLabel(f"Selected node: {self.node_id}"))
        form_layout.addRow(QLabel("Force in X (Fx):"), self.load_fx)
        form_layout.addRow(QLabel("Force in Y (Fy):"), self.load_fy)
        form_layout.addRow(QLabel("Moment in Z (Mz):"), self.load_mz)

        layout.addLayout(form_layout)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def get_data(self):
        """Retrieve loads to apply."""
        loads = []
        if fx := self.load_fx.value():  # Python 3.8+ walrus operator
            loads.append((0, fx))  # DOF 0 (ux)
        if fy := self.load_fy.value():
            loads.append((1, fy))  # DOF 1 (uy)
        if mz := self.load_mz.value():
            loads.append((2, mz))  # DOF 2 (rz)

        return {
            "node_id": self.node_id,
            "loads": loads  # List of tuples (dof_index, value)
        }
