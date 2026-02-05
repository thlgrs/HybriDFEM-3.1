"""
Dialog for adding rigid blocks to the structure.

This module provides a dialog for creating blocks by specifying
reference point, dimensions, and optional material properties.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout,
                             QDialogButtonBox, QDoubleSpinBox, QLabel, QComboBox)


class AddBlockDialog(QDialog):
    """Dialog for inputting parameters of a new rectangular block."""

    def __init__(self, materials_dict=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Rectangular Block")
        self.materials_dict = materials_dict or {}

        # Create input widgets
        self.xc_input = QDoubleSpinBox()
        self.xc_input.setRange(-1000.0, 1000.0)
        self.xc_input.setValue(0.0)

        self.yc_input = QDoubleSpinBox()
        self.yc_input.setRange(-1000.0, 1000.0)
        self.yc_input.setValue(0.0)

        self.length_input = QDoubleSpinBox()
        self.length_input.setRange(0.01, 1000.0)
        self.length_input.setValue(1.0)

        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(0.01, 1000.0)
        self.height_input.setValue(0.5)

        # ConstitutiveLaw selection dropdown (filter to show only Basic materials for blocks)
        self.material_combo = QComboBox()
        if self.materials_dict:
            # Show all materials but prefer Basic formulation
            for mat_name, mat_data in self.materials_dict.items():
                if isinstance(mat_data, dict):
                    formulation = mat_data.get('formulation', 'Basic')
                    if formulation == 'Basic':
                        display_name = f"{mat_name} (Block)"
                    elif formulation == 'PlaneStress':
                        thickness = mat_data.get('thickness', 0.01)
                        display_name = f"{mat_name} (PlaneStress, t={thickness*1000:.1f}mm) ⚠️"
                    elif formulation == 'PlaneStrain':
                        thickness = mat_data.get('thickness', 0.01)
                        display_name = f"{mat_name} (PlaneStrain, t={thickness*1000:.1f}mm) ⚠️"
                    else:
                        display_name = mat_name
                else:
                    # Legacy format
                    display_name = mat_name
                self.material_combo.addItem(display_name, mat_name)  # Store original name as data

            # Set default material as default selection
            default_idx = self.material_combo.findText("Concrete C30/37 (Default) (Block)")
            if default_idx >= 0:
                self.material_combo.setCurrentIndex(default_idx)
        else:
            self.material_combo.addItem("No materials available")

        # Create OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                      QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Layout
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        form_layout.addRow(QLabel("Center (xc):"), self.xc_input)
        form_layout.addRow(QLabel("Center (yc):"), self.yc_input)
        form_layout.addRow(QLabel("Length (l):"), self.length_input)
        form_layout.addRow(QLabel("Height (h):"), self.height_input)
        form_layout.addRow(QLabel("ConstitutiveLaw:"), self.material_combo)

        layout.addLayout(form_layout)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def get_data(self):
        """
        Helper method to retrieve data after user clicks OK.
        """
        # Get original material name from userData (not display text)
        current_idx = self.material_combo.currentIndex()
        if current_idx >= 0:
            material_name = self.material_combo.itemData(current_idx)
            if material_name is None:
                # Fallback to text if no data stored (legacy or "No materials available")
                material_name = self.material_combo.currentText()
        else:
            material_name = None

        return {
            "xc": self.xc_input.value(),
            "yc": self.yc_input.value(),
            "length": self.length_input.value(),
            "height": self.height_input.value(),
            "material_name": material_name
        }
