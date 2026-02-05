"""
Results panel for post-processing and visualization.

This module provides the results control panel for viewing displacements,
stresses, and configuring deformation visualization.
"""

import os
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QTextEdit,
                             QGroupBox, QFormLayout, QDoubleSpinBox,
                             QFileDialog, QMessageBox, QComboBox)

# Import Visualizer for plot generation
try:
    from Core.Solvers.Visualizer import Visualizer, PlotStyle
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    print("Warning: Visualizer not available for plot generation")


class ResultsPanel(QWidget):
    """Panel for displaying and exporting analysis results."""

    def __init__(self, project_state, parent=None):
        super().__init__(parent)
        self.project_state = project_state
        self.main_window = parent
        self.init_ui()

        # Signal connections
        self.project_state.results_ready.connect(self.update_results_summary)
        # Ensure it's empty at startup
        self.update_results_summary(None)

    def init_ui(self):
        layout = QVBoxLayout()

        # Export and Plotting Options
        export_group = QGroupBox("Export & Post-Processing")
        export_layout = QVBoxLayout()

        # Export buttons row
        export_buttons_layout = QHBoxLayout()
        self.btn_export_vtk = QPushButton("Export VTK")
        self.btn_export_vtk.clicked.connect(self.export_vtk)
        self.btn_export_vtk.setEnabled(False)  # Enable after analysis
        self.btn_export_vtk.setToolTip("Export results to VTK format for ParaView/VisIt")
        export_buttons_layout.addWidget(self.btn_export_vtk)
        self.btn_export_rhino = QPushButton("Export for Rhino")
        self.btn_export_rhino.clicked.connect(self.export_rhino)
        self.btn_export_rhino.setEnabled(False)  # Enable after analysis
        self.btn_export_rhino.setToolTip("Export deformed geometry to Rhino-compatible format")
        export_buttons_layout.addWidget(self.btn_export_rhino)
        export_layout.addLayout(export_buttons_layout)

        # Plotting buttons row
        plot_buttons_layout = QHBoxLayout()
        self.btn_plot_curve = QPushButton("Plot Load-Displacement")
        self.btn_plot_curve.clicked.connect(self.open_plot_curve_dialog)
        self.btn_plot_curve.setEnabled(False)  # Enable after nonlinear analysis
        self.btn_plot_curve.setToolTip("Plot load-displacement curves from HDF5 results")
        plot_buttons_layout.addWidget(self.btn_plot_curve)

        self.btn_plot_convergence = QPushButton("Plot Convergence")
        self.btn_plot_convergence.clicked.connect(self.plot_convergence_history)
        self.btn_plot_convergence.setEnabled(False)  # Enable after nonlinear analysis
        self.btn_plot_convergence.setToolTip("Plot convergence history (iterations and residuals)")
        plot_buttons_layout.addWidget(self.btn_plot_convergence)
        export_layout.addLayout(plot_buttons_layout)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Generate Results Plots Group
        plots_group = QGroupBox("Generate Result Plots")
        plots_layout = QVBoxLayout()

        # Plot type selector
        plot_type_layout = QHBoxLayout()
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Deformed Shape",
            "Displacement Field",
            "Stress Field (σxx)",
            "Stress Field (σyy)",
            "Stress Field (τxy)",
            "All Plots"
        ])
        plot_type_layout.addWidget(self.plot_type_combo)
        plots_layout.addLayout(plot_type_layout)

        # Generate and save buttons
        gen_buttons_layout = QHBoxLayout()

        self.btn_show_plot = QPushButton("Show Plot")
        self.btn_show_plot.clicked.connect(self.show_result_plot)
        self.btn_show_plot.setEnabled(False)
        self.btn_show_plot.setToolTip("Display selected plot in a new window")
        gen_buttons_layout.addWidget(self.btn_show_plot)

        self.btn_save_plot = QPushButton("Save Plot...")
        self.btn_save_plot.clicked.connect(self.save_result_plot)
        self.btn_save_plot.setEnabled(False)
        self.btn_save_plot.setToolTip("Save selected plot to file")
        gen_buttons_layout.addWidget(self.btn_save_plot)

        self.btn_save_all = QPushButton("Save All Plots...")
        self.btn_save_all.clicked.connect(self.save_all_plots)
        self.btn_save_all.setEnabled(False)
        self.btn_save_all.setToolTip("Save all standard plots to a directory")
        gen_buttons_layout.addWidget(self.btn_save_all)

        plots_layout.addLayout(gen_buttons_layout)
        plots_group.setLayout(plots_layout)
        layout.addWidget(plots_group)

        # Visualization Options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QFormLayout()
        self.deformation_scale = QDoubleSpinBox()
        self.deformation_scale.setRange(0, 10000)
        self.deformation_scale.setValue(50.0)  # Default scale of 50

        # Connect scale change to viewport redraw
        self.deformation_scale.valueChanged.connect(self.on_scale_changed)

        viz_layout.addRow("Deformation Scale:", self.deformation_scale)

        # Plot style button
        self.btn_plot_style = QPushButton("Configure Plot Style...")
        self.btn_plot_style.clicked.connect(self.open_plot_style_dialog)
        self.btn_plot_style.setToolTip("Customize visualization style (colors, line widths, etc.)")
        viz_layout.addRow("", self.btn_plot_style)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Results Summary
        summary_group = QGroupBox("Results Summary")
        summary_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        summary_layout.addWidget(self.results_text)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        layout.addStretch()
        self.setLayout(layout)

    def open_plot_style_dialog(self):
        """Open dialog to configure plot style."""
        from GUI.Dialogs.PlotStyleDialog import PlotStyleDialog

        dialog = PlotStyleDialog(
            current_style=self.project_state.plot_style,
            parent=self.main_window
        )

        if dialog.exec():
            # Update project state with new style
            self.project_state.plot_style = dialog.get_style()
            self.project_state.log_message.emit("Plot style updated")

            # Redraw current plot with new style
            if self.project_state.results:
                self.main_window.viewport.plot_results(self.project_state.results)
            else:
                self.main_window.viewport.plot_geometry()

    def on_scale_changed(self):
        """
        Slot triggered when user changes deformation scale.
        Redraws viewport with new scale.
        """
        # Only redraw if results exist
        if self.project_state.results:
            # Call viewport's plot_results method directly
            self.main_window.viewport.plot_results(self.project_state.results)

    def update_results_summary(self, solved_structure):
        """
        Slot triggered when 'results_ready' signal is emitted.
        Displays analysis results summary.
        """

        if not solved_structure:
            self.results_text.setText("No results available. Run an analysis.")
            return

        structure = solved_structure
        summary = []
        summary.append("=== Analysis Results ===\n")

        try:
            # Check for displacement results
            if hasattr(structure, 'U') and structure.U is not None and structure.U.any():
                # Find max displacement (ignoring rotations)
                # FIXED: Variable DOF compatible - use node_dof_offsets
                displacements = []

                if hasattr(structure, 'node_dof_offsets') and hasattr(structure, 'node_dof_counts'):
                    # Variable DOF system (2025 refactoring)
                    for i in range(len(structure.list_nodes)):
                        base_dof = structure.node_dof_offsets[i]
                        # Extract ux and uy (all nodes have at least 2 DOFs)
                        displacements.append(structure.U[base_dof + 0])  # ux
                        displacements.append(structure.U[base_dof + 1])  # uy
                    displacements = np.array(displacements)
                else:
                    # Legacy 3-DOF system (fallback)
                    displacements = structure.U[0::3]  # ux
                    displacements = np.append(displacements, structure.U[1::3])  # uy

                max_disp = np.max(np.abs(displacements))
                min_disp = np.min(displacements)

                summary.append(f"Max Displacement: {max_disp:.3e} m")
                summary.append(f"Min Displacement: {min_disp:.3e} m")

            # Check for modal analysis results
            if hasattr(structure, 'eig_vals') and structure.eig_vals is not None:
                summary.append(f"\nNatural Frequencies ({len(structure.eig_vals)} modes):")
                # Display first 5 modes
                for i, freq_rad_sq in enumerate(structure.eig_vals[:5]):
                    # Check if frequency is not negative (instability)
                    if freq_rad_sq > 0:
                        hz = np.sqrt(freq_rad_sq) / (2 * np.pi)
                        summary.append(f"  Mode {i + 1}: {hz:.3f} Hz")
                    else:
                        summary.append(f"  Mode {i + 1}: {np.sqrt(np.abs(freq_rad_sq)):.3f} rad/s (Unstable)")

            self.results_text.setText("\n".join(summary))

        except Exception as e:
            self.project_state.log_message.emit(f"[ERROR] Results display failed: {e}")
            self.results_text.setText(f"Error reading results: {e}")

        # Enable/disable plotting buttons based on analysis type
        # HDF5 output is generated only by force/displacement control
        # We can't automatically detect if HDF5 exists, so keep buttons available for user to try
        if solved_structure:
            self.btn_plot_curve.setEnabled(True)
            self.btn_plot_convergence.setEnabled(True)
            self.btn_export_vtk.setEnabled(True)
            self.btn_export_rhino.setEnabled(True)
            # Enable result plot generation buttons
            self.btn_show_plot.setEnabled(VISUALIZER_AVAILABLE)
            self.btn_save_plot.setEnabled(VISUALIZER_AVAILABLE)
            self.btn_save_all.setEnabled(VISUALIZER_AVAILABLE)

    def open_plot_curve_dialog(self):
        """Open dialog for plotting load-displacement curves from HDF5."""
        from GUI.Dialogs.PlotCurveDialog import PlotCurveDialog

        # Get last HDF5 file path if available (could store in project state)
        # For now, let user browse
        dialog = PlotCurveDialog(
            structure=self.project_state.structure,
            hdf5_path=None,
            parent=self.main_window
        )
        dialog.exec()

    def export_vtk(self):
        """Export results to VTK format for ParaView/VisIt."""
        from GUI.Utils.vtk_export import export_vtk

        if not self.project_state.results:
            QMessageBox.warning(
                self,
                "No Results",
                "No analysis results available. Run an analysis first."
            )
            return

        # Browse for output file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export VTK File",
            "",
            "VTK Legacy Format (*.vtk);;VTK XML Format (*.vtu);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            # Determine format from extension
            if file_path.endswith('.vtu'):
                format = 'vtu'
            else:
                format = 'vtk'

            # Get deformation scale from GUI
            scale = self.deformation_scale.value()

            # Export with displacements if available
            include_disp = self.project_state.results.U is not None

            output_path = export_vtk(
                self.project_state.results,
                file_path,
                format=format,
                include_displacements=include_disp,
                deformation_scale=scale
            )

            QMessageBox.information(
                self,
                "Export Successful",
                f"Results exported successfully to:\n{output_path}\n\n"
                f"Open this file in ParaView or VisIt to visualize."
            )
            self.project_state.log_message.emit(f"VTK export: {output_path}")

        except ImportError as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"VTK export failed:\n{e}\n\n"
                "For .vtu format, install pyevtk:\npip install pyevtk\n\n"
                "Use .vtk format instead (legacy format, no dependencies)."
            )
            self.project_state.log_message.emit(f"[ERROR] VTK export failed: {e}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export VTK:\n{e}"
            )
            self.project_state.log_message.emit(f"[ERROR] VTK export failed: {e}")

    def export_rhino(self):
        """Export deformed geometry to Rhino-compatible format."""
        from GUI.Utils.rhino_integration import RhinoExporter

        if not self.project_state.results:
            QMessageBox.warning(
                self,
                "No Results",
                "No analysis results available. Run an analysis first."
            )
            return

        # Ask user which format to export
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QRadioButton, QDialogButtonBox, QButtonGroup

        dialog = QDialog(self)
        dialog.setWindowTitle("Rhino Export Format")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Select export format:"))

        format_group = QButtonGroup(dialog)
        text_radio = QRadioButton("Text/CSV (easy import, Grasshopper-friendly)")
        text_radio.setChecked(True)
        format_group.addButton(text_radio, 0)
        layout.addWidget(text_radio)

        dm_radio = QRadioButton("Rhino .3dm file (requires rhino3dm library)")
        format_group.addButton(dm_radio, 1)
        layout.addWidget(dm_radio)

        csv_radio = QRadioButton("CSV coordinates (deformed nodes)")
        format_group.addButton(csv_radio, 2)
        layout.addWidget(csv_radio)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.setLayout(layout)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_format = format_group.checkedId()

        # Browse for output file
        if selected_format == 0:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Text File", "", "Text Files (*.txt);;All Files (*.*)"
            )
            extension = '.txt'
        elif selected_format == 1:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Rhino File", "", "Rhino Files (*.3dm);;All Files (*.*)"
            )
            extension = '.3dm'
        else:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV File", "", "CSV Files (*.csv);;All Files (*.*)"
            )
            extension = '.csv'

        if not file_path:
            return

        try:
            exporter = RhinoExporter(self.project_state.results)

            if selected_format == 0:
                # Text export
                exporter.export_to_text(file_path, include_displacements=True)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Results exported to:\n{file_path}\n\n"
                    "Import this file in Grasshopper using Python or file reader components."
                )

            elif selected_format == 1:
                # .3dm export
                exporter.export_to_3dm(file_path)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Rhino file created:\n{file_path}\n\n"
                    "Open this file in Rhino to view undeformed (blue) and deformed (red) geometry."
                )

            else:
                # CSV export
                scale = self.deformation_scale.value()
                exporter.export_deformed_coordinates(file_path, scale=scale)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Deformed coordinates exported to:\n{file_path}\n\n"
                    f"Deformation scale: {scale}\n"
                    "Import this CSV in Rhino/Grasshopper to create points."
                )

            self.project_state.log_message.emit(f"Rhino export: {file_path}")

        except ImportError as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Rhino export failed:\n{e}\n\n"
                "For .3dm format, install rhino3dm:\npip install rhino3dm\n\n"
                "Use Text or CSV format instead (no dependencies)."
            )
            self.project_state.log_message.emit(f"[ERROR] Rhino export failed: {e}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export for Rhino:\n{e}"
            )
            self.project_state.log_message.emit(f"[ERROR] Rhino export failed: {e}")

    def plot_convergence_history(self):
        """Plot convergence history (iterations and residuals) from HDF5."""
        import matplotlib.pyplot as plt

        # Browse for HDF5 file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 Results File",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            import h5py

            with h5py.File(file_path, 'r') as hf:
                # Check available datasets
                available = list(hf.keys())

                if 'Iterations' not in available:
                    QMessageBox.warning(
                        self,
                        "No Convergence Data",
                        "HDF5 file does not contain 'Iterations' dataset.\n\n"
                        "Convergence history is only available for nonlinear analyses."
                    )
                    return

                iterations = hf['Iterations'][:]
                residuals = hf['Residuals'][:] if 'Residuals' in available else None
                lambda_vals = hf['Lambda'][:] if 'Lambda' in available else None

            # Create figure with subplots
            if residuals is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

            # Plot iterations
            steps = np.arange(len(iterations))
            ax1.bar(steps, iterations, color='steelblue', alpha=0.7)
            ax1.set_xlabel('Step Number', fontsize=12)
            ax1.set_ylabel('Newton-Raphson Iterations', fontsize=12)
            ax1.set_title('Convergence History: Iterations per Step', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')

            # Plot residuals if available
            if residuals is not None:
                ax2.semilogy(steps, residuals, 'r-o', linewidth=2, markersize=4)
                ax2.set_xlabel('Step Number', fontsize=12)
                ax2.set_ylabel('Residual Norm', fontsize=12)
                ax2.set_title('Convergence History: Residual Norms', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, which='both')

            plt.tight_layout()
            plt.show()

            self.project_state.log_message.emit(f"Convergence history plotted from: {file_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Plotting Error",
                f"Failed to plot convergence history:\n{e}"
            )
            self.project_state.log_message.emit(f"[ERROR] Convergence plot failed: {e}")

    def show_result_plot(self):
        """Display selected plot type in a new matplotlib window."""
        import matplotlib.pyplot as plt

        if not self.project_state.results:
            QMessageBox.warning(self, "No Results", "Run an analysis first.")
            return

        if not VISUALIZER_AVAILABLE:
            QMessageBox.warning(self, "Visualizer Not Available",
                              "The Visualizer module is not available.")
            return

        structure = self.project_state.results
        plot_type = self.plot_type_combo.currentText()
        scale = self.deformation_scale.value()

        try:
            if plot_type == "Deformed Shape":
                fig = Visualizer.plot_deformed(structure, scale=scale)
                plt.show()

            elif plot_type == "Displacement Field":
                fig = Visualizer.plot_displacement(structure, scale=0, component='magnitude')
                plt.show()

            elif plot_type == "Stress Field (σxx)":
                fig = Visualizer.plot_stress(structure, scale=0, component='normal', angle=np.pi/2)
                plt.show()

            elif plot_type == "Stress Field (σyy)":
                fig = Visualizer.plot_stress(structure, scale=0, component='normal', angle=0)
                plt.show()

            elif plot_type == "Stress Field (τxy)":
                fig = Visualizer.plot_stress(structure, scale=0, component='shear', angle=0)
                plt.show()

            elif plot_type == "All Plots":
                # Show all plots in separate windows
                Visualizer.plot_initial(structure)
                Visualizer.plot_deformed(structure, scale=scale)
                try:
                    Visualizer.plot_displacement(structure, scale=0)
                except Exception:
                    pass  # May fail for block-only structures
                try:
                    Visualizer.plot_stress(structure, scale=0, component='normal', angle=0)
                except Exception:
                    pass  # May fail for block-only structures
                plt.show()

            self.project_state.log_message.emit(f"Plot generated: {plot_type}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Plot Error", f"Failed to generate plot:\n{e}")
            self.project_state.log_message.emit(f"[ERROR] Plot generation failed: {e}")

    def save_result_plot(self):
        """Save selected plot type to a file."""
        import matplotlib.pyplot as plt

        if not self.project_state.results:
            QMessageBox.warning(self, "No Results", "Run an analysis first.")
            return

        if not VISUALIZER_AVAILABLE:
            QMessageBox.warning(self, "Visualizer Not Available",
                              "The Visualizer module is not available.")
            return

        # Browse for output file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg);;All Files (*.*)"
        )

        if not file_path:
            return

        structure = self.project_state.results
        plot_type = self.plot_type_combo.currentText()
        scale = self.deformation_scale.value()

        # Remove extension for save_path (Visualizer adds it)
        base_path = os.path.splitext(file_path)[0]

        try:
            if plot_type == "Deformed Shape":
                Visualizer.plot_deformed(structure, scale=scale, save_path=file_path)

            elif plot_type == "Displacement Field":
                Visualizer.plot_displacement(structure, scale=0, save_path=file_path)

            elif plot_type == "Stress Field (σxx)":
                Visualizer.plot_stress(structure, scale=0, component='normal',
                                       angle=np.pi/2, save_path=file_path)

            elif plot_type == "Stress Field (σyy)":
                Visualizer.plot_stress(structure, scale=0, component='normal',
                                       angle=0, save_path=file_path)

            elif plot_type == "Stress Field (τxy)":
                Visualizer.plot_stress(structure, scale=0, component='shear',
                                       angle=0, save_path=file_path)

            plt.close('all')
            QMessageBox.information(self, "Plot Saved", f"Plot saved to:\n{file_path}")
            self.project_state.log_message.emit(f"Plot saved: {file_path}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Save Error", f"Failed to save plot:\n{e}")
            self.project_state.log_message.emit(f"[ERROR] Plot save failed: {e}")

    def save_all_plots(self):
        """Save all standard result plots to a directory."""
        import matplotlib.pyplot as plt

        if not self.project_state.results:
            QMessageBox.warning(self, "No Results", "Run an analysis first.")
            return

        if not VISUALIZER_AVAILABLE:
            QMessageBox.warning(self, "Visualizer Not Available",
                              "The Visualizer module is not available.")
            return

        # Browse for output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if not output_dir:
            return

        structure = self.project_state.results
        scale = self.deformation_scale.value()
        saved_count = 0

        try:
            # Initial (undeformed) geometry
            try:
                Visualizer.plot_initial(structure,
                                       save_path=os.path.join(output_dir, "initial.png"))
                plt.close('all')
                saved_count += 1
            except Exception as e:
                self.project_state.log_message.emit(f"[WARN] Initial plot failed: {e}")

            # Deformed shape
            try:
                Visualizer.plot_deformed(structure, scale=scale,
                                        save_path=os.path.join(output_dir, "deformed.png"))
                plt.close('all')
                saved_count += 1
            except Exception as e:
                self.project_state.log_message.emit(f"[WARN] Deformed plot failed: {e}")

            # Displacement field
            try:
                Visualizer.plot_displacement(structure, scale=0,
                                            save_path=os.path.join(output_dir, "displacement.png"))
                plt.close('all')
                saved_count += 1
            except Exception as e:
                self.project_state.log_message.emit(f"[WARN] Displacement plot failed: {e}")

            # Stress plots (may fail for block-only structures)
            try:
                # Create 2x2 stress subplot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()

                Visualizer.plot_stress(structure, scale=0, ax=axes[0],
                                      component='normal', angle=np.pi/2,
                                      title=r'$\sigma_{xx}$')
                Visualizer.plot_stress(structure, scale=0, ax=axes[1],
                                      component='normal', angle=0,
                                      title=r'$\sigma_{yy}$')
                Visualizer.plot_stress(structure, scale=0, ax=axes[2],
                                      component='shear', angle=0,
                                      title=r'$\tau_{yx}$')
                Visualizer.plot_stress(structure, scale=0, ax=axes[3],
                                      component='shear', angle=np.pi/2,
                                      title=r'$\tau_{xy}$')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "stress.png"), dpi=300, bbox_inches='tight')
                plt.close('all')
                saved_count += 1
            except Exception as e:
                self.project_state.log_message.emit(f"[WARN] Stress plot failed: {e}")

            QMessageBox.information(
                self,
                "Plots Saved",
                f"Saved {saved_count} plots to:\n{output_dir}"
            )
            self.project_state.log_message.emit(f"Saved {saved_count} plots to {output_dir}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Save Error", f"Failed to save plots:\n{e}")
            self.project_state.log_message.emit(f"[ERROR] Plot save failed: {e}")
