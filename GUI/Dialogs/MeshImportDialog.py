"""
Dialog for importing external mesh files into HybridFEM.

Supports Gmsh (.msh), Triangle (.node/.ele), and other formats via meshio.
"""

from pathlib import Path

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLabel, QComboBox, QPushButton, QGroupBox,
                             QLineEdit, QFileDialog, QDialogButtonBox,
                             QMessageBox, QRadioButton, QDoubleSpinBox)


class MeshImportDialog(QDialog):
    """Dialog for importing mesh files from external sources."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import External Mesh")
        self.setModal(True)
        self.resize(600, 500)
        self.mesh_file_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Introduction
        intro_label = QLabel(
            "<b>Import External Mesh Files</b><br>"
            "Import meshes created with Gmsh, Triangle, or other mesh generators.<br>"
            "Supported formats: .msh (Gmsh), .vtk, .vtu, .mesh, .off, and more via meshio."
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        # File selection group
        file_group = QGroupBox("Mesh File")
        file_layout = QVBoxLayout()

        file_select_layout = QHBoxLayout()
        self.file_path_input = QLineEdit()
        self.file_path_input.setPlaceholderText("Select mesh file...")
        self.file_path_input.setReadOnly(True)
        file_select_layout.addWidget(self.file_path_input)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_mesh_file)
        file_select_layout.addWidget(self.btn_browse)

        file_layout.addLayout(file_select_layout)

        # File format hint
        format_label = QLabel(
            "<i>Tip: Gmsh .msh files are recommended. "
            "For other formats, meshio must be installed (pip install meshio).</i>"
        )
        format_label.setWordWrap(True)
        file_layout.addWidget(format_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Mesh interpretation group
        interp_group = QGroupBox("Mesh Interpretation")
        interp_layout = QFormLayout()

        # Element type selection (auto-detect or override)
        self.element_type_combo = QComboBox()
        self.element_type_combo.addItems([
            "Auto-detect from file",
            "Force Triangle3 (linear triangles)",
            "Force Triangle6 (quadratic triangles)",
            "Force Quad4 (linear quads)",
            "Force Quad8 (quadratic quads)"
        ])
        self.element_type_combo.setCurrentIndex(0)
        self.element_type_combo.setToolTip(
            "Auto-detect reads element types from file.\n"
            "Force options convert all elements to selected type."
        )
        interp_layout.addRow("Element Type:", self.element_type_combo)

        # Thickness (for 2D plane stress/strain elements)
        self.thickness_input = QDoubleSpinBox()
        self.thickness_input.setRange(0.001, 10.0)
        self.thickness_input.setValue(0.01)  # 10mm default
        self.thickness_input.setDecimals(4)
        self.thickness_input.setSingleStep(0.001)
        self.thickness_input.setSuffix(" m")
        self.thickness_input.setToolTip("Thickness for 2D plane stress/strain elements")
        interp_layout.addRow("Element Thickness:", self.thickness_input)

        interp_group.setLayout(interp_layout)
        layout.addWidget(interp_group)

        # ConstitutiveLaw assignment group
        material_group = QGroupBox("ConstitutiveLaw Assignment")
        material_layout = QFormLayout()

        # ConstitutiveLaw selection (will be populated from project state)
        self.material_combo = QComboBox()
        self.material_combo.addItem("(Use default from project)")
        self.material_combo.setToolTip("ConstitutiveLaw properties for imported elements")
        material_layout.addRow("ConstitutiveLaw:", self.material_combo)

        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        # Boundary conditions group
        bc_group = QGroupBox("Boundary Conditions (Optional)")
        bc_layout = QVBoxLayout()

        bc_info_label = QLabel(
            "Boundary conditions from physical groups in the mesh file (if available).\n"
            "After import, you can add additional BCs manually in the Materials & BCs panel."
        )
        bc_info_label.setWordWrap(True)
        bc_layout.addWidget(bc_info_label)

        self.import_bc_check = QRadioButton("Import boundary groups from mesh file (if available)")
        self.import_bc_check.setChecked(True)
        bc_layout.addWidget(self.import_bc_check)

        self.skip_bc_check = QRadioButton("Skip BCs (add manually later)")
        bc_layout.addWidget(self.skip_bc_check)

        bc_group.setLayout(bc_layout)
        layout.addWidget(bc_group)

        # Preview/Info section
        info_group = QGroupBox("Mesh Information")
        info_layout = QVBoxLayout()

        self.info_label = QLabel("No mesh file selected.")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)

        self.btn_preview = QPushButton("Preview Mesh")
        self.btn_preview.clicked.connect(self.preview_mesh)
        self.btn_preview.setEnabled(False)
        info_layout.addWidget(self.btn_preview)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def browse_mesh_file(self):
        """Browse for mesh file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Mesh File",
            "",
            "Gmsh Files (*.msh);;VTK Files (*.vtk *.vtu);;All Files (*.*)"
        )

        if file_path:
            self.mesh_file_path = file_path
            self.file_path_input.setText(file_path)
            self.btn_preview.setEnabled(True)
            self.analyze_mesh_file()

    def analyze_mesh_file(self):
        """Analyze the mesh file and display information."""
        if not self.mesh_file_path:
            return

        try:
            import meshio
            mesh = meshio.read(self.mesh_file_path)

            # Extract mesh information
            num_nodes = len(mesh.points)
            num_cells = sum(len(cells.data) for cells in mesh.cells)

            # Get cell types
            cell_types = [cells.type for cells in mesh.cells]
            cell_type_str = ", ".join(set(cell_types))

            # Get physical groups if available
            physical_groups = []
            if hasattr(mesh, 'field_data') and mesh.field_data:
                physical_groups = list(mesh.field_data.keys())

            info_text = f"<b>Mesh File:</b> {Path(self.mesh_file_path).name}<br>"
            info_text += f"<b>Nodes:</b> {num_nodes}<br>"
            info_text += f"<b>Elements:</b> {num_cells}<br>"
            info_text += f"<b>Element Types:</b> {cell_type_str}<br>"

            if physical_groups:
                info_text += f"<b>Physical Groups:</b> {', '.join(physical_groups)}<br>"
                info_text += "<i>Boundary conditions can be extracted from these groups.</i>"
            else:
                info_text += "<i>No physical groups found.</i>"

            self.info_label.setText(info_text)

        except ImportError:
            self.info_label.setText(
                "<b style='color:red;'>Error:</b> meshio library not installed.<br>"
                "Install with: <code>pip install meshio</code>"
            )

        except Exception as e:
            self.info_label.setText(
                f"<b style='color:red;'>Error reading mesh:</b> {str(e)}"
            )

    def preview_mesh(self):
        """Preview the mesh using matplotlib."""
        if not self.mesh_file_path:
            QMessageBox.warning(self, "No File", "Please select a mesh file first.")
            return

        try:
            from Core.Objects.FEM.Mesh import Mesh
            import matplotlib.pyplot as plt

            # Create Mesh object from file
            mesh = Mesh(mesh_file=self.mesh_file_path)
            mesh.read_mesh()

            # Plot the mesh
            mesh.plot(title=f"Preview: {Path(self.mesh_file_path).name}")

        except ImportError as e:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Failed to import required libraries:\n{e}\n\n"
                "Make sure meshio, gmsh, and matplotlib are installed."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Preview Error",
                f"Failed to preview mesh:\n{e}"
            )

    def validate_and_accept(self):
        """Validate inputs before accepting."""
        if not self.mesh_file_path:
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please select a mesh file to import."
            )
            return

        if not Path(self.mesh_file_path).exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"Mesh file not found:\n{self.mesh_file_path}"
            )
            return

        self.accept()

    def set_available_materials(self, material_names):
        """Set available materials in the combo box."""
        self.material_combo.clear()
        self.material_combo.addItem("(Use default from project)")
        self.material_combo.addItems(material_names)

    def get_data(self):
        """
        Get the mesh import configuration.

        Returns
        -------
        dict
            Configuration dictionary with mesh import settings
        """
        element_type_map = {
            0: "auto",
            1: "Triangle3",
            2: "Triangle6",
            3: "Quad4",
            4: "Quad8"
        }

        return {
            'file_path': self.mesh_file_path,
            'element_type': element_type_map[self.element_type_combo.currentIndex()],
            'thickness': self.thickness_input.value(),
            'material_name': self.material_combo.currentText() if self.material_combo.currentIndex() > 0 else None,
            'import_bc': self.import_bc_check.isChecked()
        }
