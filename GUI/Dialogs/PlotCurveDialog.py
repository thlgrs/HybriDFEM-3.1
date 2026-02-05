"""
Dialog for plotting load-displacement curves from HDF5 results.

This dialog allows users to visualize curves from nonlinear analyses
(Force Control and Displacement Control) stored in HDF5 format.
"""

import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLabel, QComboBox, QPushButton, QCheckBox,
                             QGroupBox, QFileDialog, QMessageBox)


class PlotCurveDialog(QDialog):
    """Dialog for selecting and plotting load-displacement curves."""

    def __init__(self, structure=None, hdf5_path=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Load-Displacement Curve")
        self.structure = structure
        self.hdf5_path = hdf5_path
        self.resize(500, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # File selection
        file_group = QGroupBox("HDF5 Results File")
        file_layout = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        if self.hdf5_path:
            self.file_label.setText(self.hdf5_path)
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_file)
        file_layout.addWidget(self.btn_browse)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Plot options
        options_group = QGroupBox("Plot Options")
        options_layout = QFormLayout()

        # DOF selection
        self.dof_combo = QComboBox()
        if self.structure and hasattr(self.structure, 'nb_dofs'):
            n_dofs = self.structure.nb_dofs
            self.dof_combo.addItems([f"DOF {i}" for i in range(n_dofs)])
        else:
            self.dof_combo.addItems(["DOF 0", "DOF 1", "DOF 2"])
        self.dof_combo.setToolTip("Select degree of freedom to plot")
        options_layout.addRow("Select DOF:", self.dof_combo)

        # X-axis selection
        self.xaxis_combo = QComboBox()
        self.xaxis_combo.addItems(["Load Factor (λ)", "Step Number"])
        self.xaxis_combo.setCurrentIndex(0)
        options_layout.addRow("X-Axis:", self.xaxis_combo)

        # Y-axis selection
        self.yaxis_combo = QComboBox()
        self.yaxis_combo.addItems(["Displacement [m]", "Displacement [mm]"])
        self.yaxis_combo.setCurrentIndex(1)  # Default to mm
        options_layout.addRow("Y-Axis:", self.yaxis_combo)

        # Show iterations
        self.show_iterations_check = QCheckBox("Show iterations per step")
        self.show_iterations_check.setChecked(False)
        options_layout.addRow("", self.show_iterations_check)

        # Show residuals
        self.show_residuals_check = QCheckBox("Show residual norms")
        self.show_residuals_check.setChecked(False)
        options_layout.addRow("", self.show_residuals_check)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.btn_plot = QPushButton("Plot")
        self.btn_plot.clicked.connect(self.plot_curve)
        self.btn_plot.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.btn_plot)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        button_layout.addWidget(self.btn_close)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def browse_file(self):
        """Browse for HDF5 file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 Results File",
            "",
            "HDF5 Files (*.h5 *.hdf5);;All Files (*.*)"
        )

        if file_path:
            self.hdf5_path = file_path
            self.file_label.setText(file_path)

    def plot_curve(self):
        """Read HDF5 file and plot selected curve."""
        if not self.hdf5_path:
            QMessageBox.warning(
                self,
                "No File",
                "Please select an HDF5 results file first."
            )
            return

        try:
            import h5py

            # Read HDF5 file
            with h5py.File(self.hdf5_path, 'r') as hf:
                # Check available datasets
                available_datasets = list(hf.keys())

                # Read data
                if 'U_conv' in available_datasets:
                    U_conv = hf['U_conv'][:]  # (n_dofs, n_steps)
                else:
                    QMessageBox.critical(self, "Error", "HDF5 file does not contain 'U_conv' dataset.")
                    return

                # Read load factor or step numbers
                if 'Lambda' in available_datasets:
                    lambda_vals = hf['Lambda'][:]
                    has_lambda = True
                else:
                    has_lambda = False
                    n_steps = U_conv.shape[1]
                    lambda_vals = np.arange(n_steps)

                # Optional: Read iterations and residuals
                iterations = hf['Iterations'][:] if 'Iterations' in available_datasets else None
                residuals = hf['Residuals'][:] if 'Residuals' in available_datasets else None

            # Get selected DOF
            dof_id = self.dof_combo.currentIndex()

            if dof_id >= U_conv.shape[0]:
                QMessageBox.warning(
                    self,
                    "Invalid DOF",
                    f"Selected DOF {dof_id} is out of range. File has {U_conv.shape[0]} DOFs."
                )
                return

            # Extract displacement for selected DOF
            displacement = U_conv[dof_id, :]

            # Convert to mm if requested
            y_label = "Displacement [m]"
            if "mm" in self.yaxis_combo.currentText():
                displacement = displacement * 1000  # Convert m to mm
                y_label = "Displacement [mm]"

            # Determine X-axis
            if self.xaxis_combo.currentIndex() == 0 and has_lambda:
                x_data = lambda_vals
                x_label = "Load Factor λ"
            else:
                x_data = np.arange(len(displacement))
                x_label = "Step Number"

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x_data, displacement, 'b-o', linewidth=2, markersize=4, label=f"DOF {dof_id}")
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f"Load-Displacement Curve (DOF {dof_id})", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Show iterations if requested
            if self.show_iterations_check.isChecked() and iterations is not None:
                ax2 = ax.twinx()
                ax2.bar(x_data, iterations, alpha=0.3, color='orange', label='Iterations')
                ax2.set_ylabel('Newton-Raphson Iterations', fontsize=12, color='orange')
                ax2.tick_params(axis='y', labelcolor='orange')
                ax2.legend(loc='upper right')

            plt.tight_layout()
            plt.show()

            # Optionally plot residuals in separate figure
            if self.show_residuals_check.isChecked() and residuals is not None:
                fig2, ax3 = plt.subplots(figsize=(10, 5))
                ax3.semilogy(x_data, residuals, 'r-o', linewidth=2, markersize=4)
                ax3.set_xlabel(x_label, fontsize=12)
                ax3.set_ylabel('Residual Norm', fontsize=12)
                ax3.set_title('Convergence History', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3, which='both')
                plt.tight_layout()
                plt.show()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Plotting Error",
                f"Failed to plot curve:\n{e}\n\nCheck that the HDF5 file is valid."
            )
