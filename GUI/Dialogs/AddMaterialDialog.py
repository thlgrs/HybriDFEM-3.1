"""
Dialog for defining material properties.

This module provides a dialog for creating materials by specifying
name, elastic modulus, Poisson's ratio, and density.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout,
                             QDialogButtonBox, QDoubleSpinBox, QLineEdit, QLabel, QComboBox)


class AddMaterialDialog(QDialog):
    """Dialog for inputting properties of a new material."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New ConstitutiveLaw")

        # Create input widgets
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("E.g.: Steel S235, Concrete C30/37")

        self.E_input = QDoubleSpinBox()
        self.E_input.setRange(1e6, 1e13)  # From 1 MPa to 10,000 GPa
        self.E_input.setValue(210e9)  # Default: steel
        self.E_input.setSuffix(" Pa")
        self.E_input.setDecimals(0)
        self.E_input.setSingleStep(1e9)

        self.nu_input = QDoubleSpinBox()
        self.nu_input.setRange(0.0, 0.5)
        self.nu_input.setValue(0.3)
        self.nu_input.setSingleStep(0.01)

        self.rho_input = QDoubleSpinBox()
        self.rho_input.setRange(1, 20000)  # From 1 to 20,000 kg/m³
        self.rho_input.setValue(7850)  # Default: steel
        self.rho_input.setSuffix(" kg/m³")

        # ConstitutiveLaw formulation type
        self.formulation_combo = QComboBox()
        self.formulation_combo.addItems(['Basic (Block)', 'PlaneStress (FEM)', 'PlaneStrain (FEM)'])
        self.formulation_combo.setCurrentText('Basic (Block)')
        self.formulation_combo.currentTextChanged.connect(self.on_formulation_changed)

        # Thickness (for FEM elements only)
        self.thickness_input = QDoubleSpinBox()
        self.thickness_input.setRange(0.001, 10.0)  # From 1mm to 10m
        self.thickness_input.setValue(0.01)  # Default: 10mm
        self.thickness_input.setSuffix(" m")
        self.thickness_input.setDecimals(4)
        self.thickness_input.setSingleStep(0.001)

        # Create OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                      QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Layout
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        form_layout.addRow(QLabel("ConstitutiveLaw Name:"), self.name_input)
        form_layout.addRow(QLabel("Formulation:"), self.formulation_combo)
        form_layout.addRow(QLabel("Young's Modulus (E):"), self.E_input)
        form_layout.addRow(QLabel("Poisson's Ratio (ν):"), self.nu_input)
        form_layout.addRow(QLabel("Density (ρ):"), self.rho_input)

        # Store thickness row for show/hide
        self.thickness_label = QLabel("Thickness (t):")
        self.thickness_row = form_layout.rowCount()
        form_layout.addRow(self.thickness_label, self.thickness_input)

        layout.addLayout(form_layout)
        layout.addWidget(button_box)
        self.setLayout(layout)

        # Initialize visibility based on default formulation
        self.on_formulation_changed(self.formulation_combo.currentText())

    def on_formulation_changed(self, text):
        """Show/hide thickness input based on formulation type."""
        is_fem = 'FEM' in text
        self.thickness_label.setVisible(is_fem)
        self.thickness_input.setVisible(is_fem)

    def get_data(self):
        """
        Helper method to retrieve data after user clicks OK.
        Returns dict with: name, E, nu, rho, formulation, thickness
        """
        formulation_text = self.formulation_combo.currentText()
        # Extract formulation type from combo text
        if 'PlaneStress' in formulation_text:
            formulation = 'PlaneStress'
        elif 'PlaneStrain' in formulation_text:
            formulation = 'PlaneStrain'
        else:
            formulation = 'Basic'

        return {
            "name": self.name_input.text().strip(),
            "E": self.E_input.value(),
            "nu": self.nu_input.value(),
            "rho": self.rho_input.value(),
            "formulation": formulation,
            "thickness": self.thickness_input.value()
        }
