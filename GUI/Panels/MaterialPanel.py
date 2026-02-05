"""
ConstitutiveLaw panel for material library management.

This module provides the material control panel for defining, editing,
and assigning material properties to structural elements.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel,
                             QTextEdit, QGroupBox, QListWidget, QMessageBox)

from GUI.Dialogs.AddMaterialDialog import AddMaterialDialog


class MaterialPanel(QWidget):
    """Panel for managing material library and boundary conditions."""

    def __init__(self, project_state, parent=None):
        super().__init__(parent)
        self.project_state = project_state
        self.main_window = parent
        self.init_ui()

        # Signal connections
        self.project_state.materials_changed.connect(self.update_material_list)
        self.project_state.bcs_changed.connect(self.update_bc_list)
        self.project_state.selection_mode_changed.connect(self.on_selection_mode_changed)

        # Initialize lists
        self.update_material_list()
        self.update_bc_list()

    def init_ui(self):
        layout = QVBoxLayout()

        # ConstitutiveLaw Library Section
        material_group = QGroupBox("ConstitutiveLaw Library")
        material_layout = QVBoxLayout()
        self.material_list_widget = QListWidget()
        self.material_list_widget.setMaximumHeight(150)
        material_layout.addWidget(self.material_list_widget)
        mat_btn_layout = QHBoxLayout()
        self.btn_add_material = QPushButton("Add...")
        self.btn_add_material.clicked.connect(self.open_add_material_dialog)
        self.btn_edit_material = QPushButton("Edit...")
        self.btn_edit_material.clicked.connect(self.edit_selected_material)
        self.btn_edit_material.setEnabled(False)  # Enable when material selected
        self.btn_del_material = QPushButton("Delete")
        self.btn_del_material.clicked.connect(self.delete_selected_material)
        self.btn_del_material.setEnabled(False)  # Enable when material selected
        mat_btn_layout.addWidget(self.btn_add_material)
        mat_btn_layout.addWidget(self.btn_edit_material)
        mat_btn_layout.addWidget(self.btn_del_material)

        # Connect material list selection to update button states
        self.material_list_widget.itemSelectionChanged.connect(self.update_material_button_states)
        material_layout.addLayout(mat_btn_layout)
        material_group.setLayout(material_layout)
        layout.addWidget(material_group)

        # Boundary Conditions Section
        bc_group = QGroupBox("Boundary Conditions & Loads")
        bc_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.btn_add_support = QPushButton("Add Support")
        self.btn_add_support.setCheckable(True)  # Toggle button
        self.btn_add_support.clicked.connect(self.toggle_add_support_mode)

        self.btn_add_load = QPushButton("Add Load")
        self.btn_add_load.setCheckable(True)  # Toggle button
        self.btn_add_load.clicked.connect(self.toggle_add_load_mode)

        btn_layout.addWidget(self.btn_add_support)
        btn_layout.addWidget(self.btn_add_load)
        bc_layout.addLayout(btn_layout)

        self.bc_list = QTextEdit()
        self.bc_list.setReadOnly(True)
        self.bc_list.setMaximumHeight(200)
        bc_layout.addWidget(QLabel("Applied Conditions:"))
        bc_layout.addWidget(self.bc_list)

        bc_group.setLayout(bc_layout)
        layout.addWidget(bc_group)

        layout.addStretch()
        self.setLayout(layout)

    def open_add_material_dialog(self):
        dialog = AddMaterialDialog(self.main_window)
        if dialog.exec():
            data = dialog.get_data()
            if not data["name"]:
                QMessageBox.warning(self, "Error", "ConstitutiveLaw name is required.")
                return
            self.project_state.add_new_material(
                name=data["name"],
                E=data["E"],
                nu=data["nu"],
                rho=data["rho"],
                formulation=data.get("formulation", "Basic"),
                thickness=data.get("thickness", 0.01)
            )

    def update_material_list(self):
        """Update material list with enhanced display showing formulation and thickness."""
        self.material_list_widget.clear()
        if self.project_state.materials:
            for mat_name, mat_data in self.project_state.materials.items():
                # Enhanced display format
                if isinstance(mat_data, dict):
                    formulation = mat_data.get('formulation', 'Basic')
                    if formulation == 'PlaneStress':
                        thickness = mat_data.get('thickness', 0.01)
                        display_name = f"{mat_name} (PlaneStress, t={thickness*1000:.1f}mm)"
                    elif formulation == 'PlaneStrain':
                        thickness = mat_data.get('thickness', 0.01)
                        display_name = f"{mat_name} (PlaneStrain, t={thickness*1000:.1f}mm)"
                    else:  # Basic
                        display_name = f"{mat_name} (Block)"
                else:
                    # Legacy format
                    display_name = mat_name
                self.material_list_widget.addItem(display_name)

    def toggle_add_support_mode(self, checked):
        """Toggle support selection mode."""
        if checked:
            self.btn_add_load.setChecked(False)  # Deactivate other button
            self.project_state.set_selection_mode("select_support_node")
        else:
            self.project_state.set_selection_mode("idle")

    def toggle_add_load_mode(self, checked):
        """Toggle load selection mode."""
        if checked:
            self.btn_add_support.setChecked(False)  # Deactivate other button
            self.project_state.set_selection_mode("select_load_node")
        else:
            self.project_state.set_selection_mode("idle")

    def on_selection_mode_changed(self, mode):
        """
        Slot triggered when 'selection_mode_changed' signal is emitted.
        Ensures buttons reflect current state.
        """
        if mode != "select_support_node":
            self.btn_add_support.setChecked(False)
        if mode != "select_load_node":
            self.btn_add_load.setChecked(False)

    def update_bc_list(self):
        """
        Slot triggered when 'bcs_changed' signal is emitted.
        Updates text display of supports and loads.
        """
        self.bc_list.clear()
        lines = []

        # Display supports
        if self.project_state.supports:
            lines.append("--- Supports ---")
            for node_id, dofs in self.project_state.supports.items():
                dof_str = ", ".join(["ux", "uy", "rz"][i] for i in dofs)
                lines.append(f"  Node {node_id}: Fixed in [{dof_str}]")

        # Display loads
        if self.project_state.loads:
            lines.append("\n--- Nodal Loads ---")
            for node_id, load_list in self.project_state.loads.items():
                load_str = []
                for dof_index, value in load_list:
                    dof_name = ["Fx", "Fy", "Mz"][dof_index]
                    load_str.append(f"{dof_name}={value} N/Nm")
                lines.append(f"  Node {node_id}: {'; '.join(load_str)}")

        if not lines:
            lines.append("No conditions applied.")

        self.bc_list.setText("\n".join(lines))

    def update_material_button_states(self):
        """Enable/disable edit and delete buttons based on selection."""
        has_selection = len(self.material_list_widget.selectedItems()) > 0
        self.btn_edit_material.setEnabled(has_selection)
        self.btn_del_material.setEnabled(has_selection)

    def edit_selected_material(self):
        """Edit the selected material."""
        selected_items = self.material_list_widget.selectedItems()
        if not selected_items:
            return

        # Extract material name from display format
        display_name = selected_items[0].text()
        # Parse name from display format (e.g., "Steel (PlaneStress, t=10.0mm)" -> "Steel")
        mat_name = display_name.split(" (")[0] if " (" in display_name else display_name

        if mat_name not in self.project_state.materials:
            QMessageBox.warning(self, "Error", f"ConstitutiveLaw '{mat_name}' not found.")
            return

        # Get current material data
        mat_data = self.project_state.materials[mat_name]
        if not isinstance(mat_data, dict):
            QMessageBox.warning(self, "Error", "Cannot edit legacy material format.")
            return

        # Open dialog pre-filled with current values
        dialog = AddMaterialDialog(self.main_window)
        dialog.setWindowTitle("Edit ConstitutiveLaw")
        dialog.name_input.setText(mat_name)
        dialog.name_input.setReadOnly(True)  # Don't allow name change
        dialog.E_input.setValue(mat_data['E'])
        dialog.nu_input.setValue(mat_data['nu'])
        dialog.rho_input.setValue(mat_data['rho'])
        dialog.thickness_input.setValue(mat_data['thickness'])

        # Set formulation
        formulation = mat_data['formulation']
        if formulation == 'PlaneStress':
            dialog.formulation_combo.setCurrentIndex(1)
        elif formulation == 'PlaneStrain':
            dialog.formulation_combo.setCurrentIndex(2)
        else:
            dialog.formulation_combo.setCurrentIndex(0)

        if dialog.exec():
            data = dialog.get_data()

            # Update material in place
            try:
                # Create new material object
                from Core.Objects.ConstitutiveLaw.Material import Material, PlaneStress, PlaneStrain

                if data["formulation"] == 'PlaneStress':
                    mat_obj = PlaneStress(E=data["E"], nu=data["nu"], rho=data["rho"])
                elif data["formulation"] == 'PlaneStrain':
                    mat_obj = PlaneStrain(E=data["E"], nu=data["nu"], rho=data["rho"])
                else:
                    mat_obj = Material(E=data["E"], nu=data["nu"], rho=data["rho"])

                # Update material data
                self.project_state.materials[mat_name] = {
                    'object': mat_obj,
                    'formulation': data["formulation"],
                    'E': data["E"],
                    'nu': data["nu"],
                    'rho': data["rho"],
                    'thickness': data["thickness"]
                }

                self.project_state.log_message.emit(f"ConstitutiveLaw '{mat_name}' updated.")
                self.project_state.materials_changed.emit()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to update material: {e}")

    def delete_selected_material(self):
        """Delete the selected material after confirmation."""
        selected_items = self.material_list_widget.selectedItems()
        if not selected_items:
            return

        # Extract material name from display format
        display_name = selected_items[0].text()
        mat_name = display_name.split(" (")[0] if " (" in display_name else display_name

        if mat_name not in self.project_state.materials:
            QMessageBox.warning(self, "Error", f"ConstitutiveLaw '{mat_name}' not found.")
            return

        # Check if material is assigned to any geometry
        is_assigned = False
        if self.project_state.structure:
            # Check blocks
            if hasattr(self.project_state.structure, 'list_blocks'):
                for block in self.project_state.structure.list_blocks:
                    if hasattr(block, 'material'):
                        # Find material name by matching object
                        for name, mat_data in self.project_state.materials.items():
                            mat_obj = mat_data['object'] if isinstance(mat_data, dict) else mat_data
                            if mat_obj is block.material and name == mat_name:
                                is_assigned = True
                                break
                    if is_assigned:
                        break

            # Check FEM elements
            if not is_assigned and hasattr(self.project_state.structure, 'list_fes'):
                for fe in self.project_state.structure.list_fes:
                    if hasattr(fe, 'mat'):
                        for name, mat_data in self.project_state.materials.items():
                            mat_obj = mat_data['object'] if isinstance(mat_data, dict) else mat_data
                            if mat_obj is fe.mat and name == mat_name:
                                is_assigned = True
                                break
                    if is_assigned:
                        break

        if is_assigned:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                f"ConstitutiveLaw '{mat_name}' is assigned to geometry elements.\n\n"
                "Please remove or reassign elements before deleting this material."
            )
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete material '{mat_name}'?\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            del self.project_state.materials[mat_name]
            self.project_state.log_message.emit(f"ConstitutiveLaw '{mat_name}' deleted.")
            self.project_state.materials_changed.emit()
