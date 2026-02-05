"""
Geometry panel for structure definition and block management.

This module provides the geometry control panel for importing structures
from Rhino, adding blocks, and managing structural geometry.
"""

import os

from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QTextEdit,
                             QGroupBox, QLineEdit, QComboBox, QLabel)

from GUI.Dialogs.AddBlockDialog import AddBlockDialog
from GUI.Dialogs.MeshGeneratorDialog import MeshGeneratorDialog
from GUI.Dialogs.ContactConfigDialog import ContactConfigDialog
from GUI.Dialogs.MeshImportDialog import MeshImportDialog


class GeometryPanel(QWidget):
    """Panel for geometry definition including Rhino import and block management."""

    def __init__(self, project_state, parent=None):
        super().__init__(parent)
        self.project_state = project_state
        self.main_window = parent
        self.init_ui()

        self.project_state.structure_changed.connect(self.update_geometry_info)
        self.update_geometry_info()

    def init_ui(self):
        layout = QVBoxLayout()

        # Structure Type Selection
        type_group = QGroupBox("Structure Type")
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        type_layout.addWidget(type_label)

        self.structure_type_combo = QComboBox()
        self.structure_type_combo.addItems(['Hybrid', 'Block', 'FEM'])
        self.structure_type_combo.setCurrentText('Hybrid')  # Default
        self.structure_type_combo.currentTextChanged.connect(self.on_structure_type_changed)
        type_layout.addWidget(self.structure_type_combo)

        type_info = QLabel("(Hybrid: Blocks+FEM, Block: Rigid blocks only, FEM: Continuous elements only)")
        type_info.setWordWrap(True)
        type_info.setStyleSheet("color: gray; font-size: 9pt;")
        type_layout.addWidget(type_info)
        type_layout.addStretch()

        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Rhino Import Section
        rhino_group = QGroupBox("Rhino Import")
        rhino_layout = QVBoxLayout()
        import_layout = QHBoxLayout()
        self.rhino_path_edit = QLineEdit()
        self.rhino_path_edit.setPlaceholderText("Select Rhino file or geometry...")
        import_layout.addWidget(self.rhino_path_edit)
        self.btn_browse_rhino = QPushButton("Browse")
        self.btn_browse_rhino.clicked.connect(self.browse_rhino_file)
        import_layout.addWidget(self.btn_browse_rhino)
        self.btn_import_rhino = QPushButton("Import from Rhino")
        self.btn_import_rhino.clicked.connect(self.import_from_rhino)
        import_layout.addWidget(self.btn_import_rhino)
        rhino_layout.addLayout(import_layout)
        rhino_group.setLayout(rhino_layout)
        layout.addWidget(rhino_group)

        # Manual Creation Section
        manual_group = QGroupBox("Manual Creation")
        manual_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()

        self.btn_add_block = QPushButton("Add Block")
        self.btn_add_block.clicked.connect(self.open_add_block_dialog)
        btn_layout.addWidget(self.btn_add_block)

        self.btn_add_beam = QPushButton("Add Beam (FEM)")
        self.btn_add_beam.setEnabled(False)  # TODO: implement beam dialog
        btn_layout.addWidget(self.btn_add_beam)

        self.btn_add_fem = QPushButton("Generate Mesh (FEM)")
        self.btn_add_fem.clicked.connect(self.open_mesh_generator_dialog)
        btn_layout.addWidget(self.btn_add_fem)

        manual_layout.addLayout(btn_layout)

        # Second row for import button
        import_btn_layout = QHBoxLayout()

        self.btn_import_mesh = QPushButton("Import External Mesh...")
        self.btn_import_mesh.clicked.connect(self.open_mesh_import_dialog)
        self.btn_import_mesh.setToolTip("Import mesh from Gmsh, Triangle, or other formats")
        import_btn_layout.addWidget(self.btn_import_mesh)
        import_btn_layout.addStretch()

        manual_layout.addLayout(import_btn_layout)

        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)

        # Contact Configuration (for Block structures)
        contact_group = QGroupBox("Contact Configuration (Block Structures)")
        contact_layout = QVBoxLayout()

        self.contact_status_label = QLabel("Not configured")
        self.contact_status_label.setStyleSheet("color: gray;")
        contact_layout.addWidget(self.contact_status_label)

        self.btn_configure_contact = QPushButton("Configure Contacts...")
        self.btn_configure_contact.clicked.connect(self.open_contact_config_dialog)
        self.btn_configure_contact.setEnabled(False)  # Enable when blocks exist
        contact_layout.addWidget(self.btn_configure_contact)

        contact_group.setLayout(contact_layout)
        layout.addWidget(contact_group)

        # Connect to structure changed signal to update contact button
        self.project_state.structure_changed.connect(self.update_contact_button_state)

        # Geometry Info
        info_group = QGroupBox("Geometry Information")
        info_layout = QVBoxLayout()
        self.geometry_info = QTextEdit()
        self.geometry_info.setReadOnly(True)
        self.geometry_info.setMaximumHeight(150)
        info_layout.addWidget(self.geometry_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        layout.addStretch()
        self.setLayout(layout)

    def on_structure_type_changed(self, text):
        """
        Slot triggered when user changes structure type selection.
        Converts UI text to lowercase for backend.
        """
        stype = text.lower()  # Convert 'Hybrid' -> 'hybrid', etc.
        self.project_state.set_structure_type(stype)

    def browse_rhino_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Rhino File", "",
            "Rhino Files (*.3dm);;All Files (*.*)"
        )
        if file_path:
            self.rhino_path_edit.setText(file_path)

    def import_from_rhino(self):
        file_path = self.rhino_path_edit.text()
        if not file_path:
            self.project_state.log_message.emit("No file selected. Loading dummy example.")
            self.project_state.load_structure_from_rhino("dummy_path_example")
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Warning", "File does not exist")
            return

        self.project_state.load_structure_from_rhino(file_path)

    def open_add_block_dialog(self):
        """Open dialog to add a new block."""
        # Pass material library to dialog
        dialog = AddBlockDialog(
            materials_dict=self.project_state.materials,
            parent=self.main_window
        )

        if dialog.exec():
            data = dialog.get_data()
            self.project_state.create_new_block(
                xc=data["xc"],
                yc=data["yc"],
                length=data["length"],
                height=data["height"],
                material_name=data["material_name"]
            )

    def open_mesh_generator_dialog(self):
        """Open dialog to generate structured FEM mesh."""
        # Pass material library to dialog
        dialog = MeshGeneratorDialog(
            materials_dict=self.project_state.materials,
            parent=self.main_window
        )

        if dialog.exec():
            data = dialog.get_data()
            if not data["material_name"]:
                QMessageBox.warning(self, "Warning",
                                  "No FEM material selected. Please add a PlaneStress or PlaneStrain material first.")
                return

            self.project_state.create_structured_mesh(
                x0=data["x0"],
                y0=data["y0"],
                x1=data["x1"],
                y1=data["y1"],
                nx=data["nx"],
                ny=data["ny"],
                element_type=data["element_type"],
                material_name=data["material_name"]
            )

    def open_mesh_import_dialog(self):
        """Open dialog to import external mesh file."""
        dialog = MeshImportDialog(parent=self.main_window)

        # Populate material list
        material_names = list(self.project_state.materials.keys())
        dialog.set_available_materials(material_names)

        if dialog.exec():
            data = dialog.get_data()

            try:
                self.project_state.import_external_mesh(
                    file_path=data['file_path'],
                    element_type=data['element_type'],
                    thickness=data['thickness'],
                    material_name=data['material_name'],
                    import_bc=data['import_bc']
                )

                QMessageBox.information(
                    self,
                    "Mesh Imported",
                    f"Mesh imported successfully from:\n{data['file_path']}\n\n"
                    "Check the console log for details."
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(
                    self,
                    "Import Error",
                    f"Failed to import mesh:\n{e}\n\n"
                    "Check that the file format is supported and meshio is installed."
                )

    def update_geometry_info(self):
        """Slot triggered when 'structure_changed' signal is emitted."""
        structure = self.project_state.structure

        if structure:
            info = []
            info.append(f"Structure Type: {type(structure).__name__}")

            n_nodes = len(structure.list_nodes) if hasattr(structure, 'list_nodes') else 0
            info.append(f"Number of Nodes: {n_nodes}")

            if hasattr(structure, 'list_blocks'):
                info.append(f"Number of Blocks: {len(structure.list_blocks)}")

                # Show material for each block
                if len(structure.list_blocks) > 0:
                    info.append("\nBlock Materials:")
                    for i, block in enumerate(structure.list_blocks):
                        if hasattr(block, 'material') and block.material:
                            # Find material name by matching the material object
                            mat_name = "Unknown"
                            for name, mat_data in self.project_state.materials.items():
                                # Handle both dict (new) and direct object (legacy)
                                mat_obj = mat_data['object'] if isinstance(mat_data, dict) else mat_data
                                if mat_obj is block.material:
                                    mat_name = name
                                    break
                            info.append(f"  Block {i}: {mat_name}")
                        else:
                            info.append(f"  Block {i}: No material")

            if hasattr(structure, 'list_fes'):
                info.append(f"Number of FE Elements: {len(structure.list_fes)}")

            info.append(f"\nTotal DOFs: {structure.nb_dofs if structure.nb_dofs else 'Not initialized'}")
            self.geometry_info.setText("\n".join(info))
        else:
            self.geometry_info.setText("No structure loaded")

    def update_contact_button_state(self):
        """Enable/disable contact configuration button based on structure type."""
        structure = self.project_state.structure

        # Enable button if structure has blocks
        has_blocks = (structure and
                     hasattr(structure, 'list_blocks') and
                     len(getattr(structure, 'list_blocks', [])) > 0)

        self.btn_configure_contact.setEnabled(has_blocks)

        # Update status label
        if has_blocks:
            if self.project_state.contact_config['enabled']:
                law = self.project_state.contact_config['law']
                nb_cps = self.project_state.contact_config['nb_cps']
                self.contact_status_label.setText(f"Configured: {law} ({nb_cps} pts/face)")
                self.contact_status_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.contact_status_label.setText("Not configured")
                self.contact_status_label.setStyleSheet("color: orange;")
        else:
            self.contact_status_label.setText("Not applicable (no blocks)")
            self.contact_status_label.setStyleSheet("color: gray;")

    def open_contact_config_dialog(self):
        """Open dialog to configure contact mechanics."""
        dialog = ContactConfigDialog(parent=self.main_window)

        # Load current configuration
        dialog.set_data(self.project_state.contact_config)

        if dialog.exec():
            config = dialog.get_data()

            # Configure contact in project state
            success = self.project_state.configure_contact(config)

            if success:
                self.update_contact_button_state()
                QMessageBox.information(
                    self,
                    "Contact Configured",
                    f"Contact mechanics configured: {config['law']}\n\n"
                    f"Contact detection and laws will be applied when analysis is run."
                )