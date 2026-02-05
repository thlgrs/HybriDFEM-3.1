"""
Dialog for generating structured FEM meshes.

This module provides a dialog for creating structured triangular meshes
by specifying a rectangular domain and division counts.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QSpinBox,
                             QLabel, QComboBox, QGroupBox)


class MeshGeneratorDialog(QDialog):
    """Dialog for generating structured triangular FEM meshes."""

    def __init__(self, materials_dict=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Structured Mesh")
        self.materials_dict = materials_dict or {}
        self.resize(500, 400)

        # Domain bounds
        self.x0_input = QDoubleSpinBox()
        self.x0_input.setRange(-1000.0, 1000.0)
        self.x0_input.setValue(0.0)
        self.x0_input.setSuffix(" m")

        self.y0_input = QDoubleSpinBox()
        self.y0_input.setRange(-1000.0, 1000.0)
        self.y0_input.setValue(0.0)
        self.y0_input.setSuffix(" m")

        self.x1_input = QDoubleSpinBox()
        self.x1_input.setRange(-1000.0, 1000.0)
        self.x1_input.setValue(3.0)
        self.x1_input.setSuffix(" m")

        self.y1_input = QDoubleSpinBox()
        self.y1_input.setRange(-1000.0, 1000.0)
        self.y1_input.setValue(1.0)
        self.y1_input.setSuffix(" m")

        # Mesh divisions
        self.nx_input = QSpinBox()
        self.nx_input.setRange(1, 1000)
        self.nx_input.setValue(10)

        self.ny_input = QSpinBox()
        self.ny_input.setRange(1, 1000)
        self.ny_input.setValue(2)

        # Element type
        self.element_type_combo = QComboBox()
        self.element_type_combo.addItems([
            'Triangle3 (Linear)',
            'Triangle6 (Quadratic)',
            'Quad4 (Bilinear)',
            'Quad8 (Serendipity)'
        ])
        self.element_type_combo.setCurrentText('Triangle3 (Linear)')
        self.element_type_combo.setToolTip(
            "Triangle3: 3-node linear triangle (fastest)\n"
            "Triangle6: 6-node quadratic triangle (better accuracy)\n"
            "Quad4: 4-node bilinear quadrilateral\n"
            "Quad8: 8-node serendipity quadrilateral (highest accuracy)"
        )

        # ConstitutiveLaw selection
        self.material_combo = QComboBox()
        if self.materials_dict:
            # Filter to show only FEM-compatible materials
            fem_materials = []
            for name, mat_data in self.materials_dict.items():
                if isinstance(mat_data, dict):
                    if mat_data['formulation'] in ['PlaneStress', 'PlaneStrain']:
                        fem_materials.append(name)
                # If no FEM materials, show all
            if not fem_materials:
                self.material_combo.addItems(list(self.materials_dict.keys()))
                self.material_combo.setEnabled(False)
                self.no_fem_mat_label = QLabel("⚠️ No FEM materials available. Please add PlaneStress or PlaneStrain material first.")
                self.no_fem_mat_label.setStyleSheet("color: orange;")
                self.no_fem_mat_label.setWordWrap(True)
            else:
                self.material_combo.addItems(fem_materials)
                self.no_fem_mat_label = None
        else:
            self.material_combo.addItem("No materials available")
            self.material_combo.setEnabled(False)
            self.no_fem_mat_label = QLabel("⚠️ No materials in library.")
            self.no_fem_mat_label.setStyleSheet("color: orange;")

        # Create OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                      QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Layout
        layout = QVBoxLayout()

        # Domain group
        domain_group = QGroupBox("Rectangular Domain")
        domain_layout = QFormLayout()
        domain_layout.addRow(QLabel("X Min (x₀):"), self.x0_input)
        domain_layout.addRow(QLabel("Y Min (y₀):"), self.y0_input)
        domain_layout.addRow(QLabel("X Max (x₁):"), self.x1_input)
        domain_layout.addRow(QLabel("Y Max (y₁):"), self.y1_input)
        domain_group.setLayout(domain_layout)
        layout.addWidget(domain_group)

        # Mesh group
        mesh_group = QGroupBox("Mesh Parameters")
        mesh_layout = QFormLayout()
        mesh_layout.addRow(QLabel("X Divisions (nx):"), self.nx_input)
        mesh_layout.addRow(QLabel("Y Divisions (ny):"), self.ny_input)
        mesh_layout.addRow(QLabel("Element Type:"), self.element_type_combo)
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)

        # ConstitutiveLaw group
        material_group = QGroupBox("ConstitutiveLaw")
        material_layout = QVBoxLayout()
        material_layout.addWidget(QLabel("Select ConstitutiveLaw:"))
        material_layout.addWidget(self.material_combo)
        if self.no_fem_mat_label:
            material_layout.addWidget(self.no_fem_mat_label)
        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        # Info label
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: gray; font-size: 9pt;")
        self.info_label.setWordWrap(True)
        self.update_info_label()
        layout.addWidget(self.info_label)

        # Connect signals to update info
        self.nx_input.valueChanged.connect(self.update_info_label)
        self.ny_input.valueChanged.connect(self.update_info_label)
        self.element_type_combo.currentTextChanged.connect(self.update_info_label)

        layout.addWidget(button_box)
        self.setLayout(layout)

    def update_info_label(self):
        """Update the informational label with mesh statistics."""
        nx = self.nx_input.value()
        ny = self.ny_input.value()
        elem_type = self.element_type_combo.currentText()

        # Calculate mesh statistics
        n_cells = nx * ny

        if 'Quad' in elem_type:
            # Quadrilateral elements: 1 element per cell
            n_elements = n_cells
            elem_shape = "quadrilateral"

            if 'Quad4' in elem_type:
                n_nodes = (nx + 1) * (ny + 1)  # Linear quad: corner nodes only
            else:  # Quad8
                # Serendipity element: corners + mid-side nodes
                n_nodes = (nx + 1) * (ny + 1) + nx * (ny + 1) + (nx + 1) * ny
        else:
            # Triangular elements: 2 elements per cell
            n_elements = n_cells * 2
            elem_shape = "triangular"

            if 'Triangle3' in elem_type:
                n_nodes = (nx + 1) * (ny + 1)  # Linear triangle: corner nodes only
            else:  # Triangle6
                # Quadratic triangle: corners + mid-side nodes + some interior
                n_nodes = (nx + 1) * (ny + 1) + nx * (ny + 1) + (nx + 1) * ny + nx * ny

        n_dofs = n_nodes * 2  # 2 DOF per node (ux, uy)

        info_text = (f"Mesh will generate:\n"
                    f"  • {n_elements} {elem_shape} elements\n"
                    f"  • ~{n_nodes} nodes\n"
                    f"  • ~{n_dofs} DOFs")

        self.info_label.setText(info_text)

    def get_data(self):
        """
        Helper method to retrieve data after user clicks OK.
        Returns dict with domain bounds, divisions, element type, and material.
        """
        elem_type_text = self.element_type_combo.currentText()

        # Extract element type from dropdown text
        if 'Triangle3' in elem_type_text:
            elem_type = 'Triangle3'
        elif 'Triangle6' in elem_type_text:
            elem_type = 'Triangle6'
        elif 'Quad4' in elem_type_text:
            elem_type = 'Quad4'
        elif 'Quad8' in elem_type_text:
            elem_type = 'Quad8'
        else:
            elem_type = 'Triangle3'  # Default fallback

        return {
            "x0": self.x0_input.value(),
            "y0": self.y0_input.value(),
            "x1": self.x1_input.value(),
            "y1": self.y1_input.value(),
            "nx": self.nx_input.value(),
            "ny": self.ny_input.value(),
            "element_type": elem_type,
            "material_name": self.material_combo.currentText() if self.material_combo.isEnabled() else None
        }
