"""
Dialog for selecting and customizing PlotStyle presets.

This dialog allows users to choose publication-ready plot styles
or customize their own styling for structure visualizations.
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
                             QLabel, QComboBox, QPushButton, QGroupBox,
                             QDoubleSpinBox, QSpinBox, QCheckBox,
                             QDialogButtonBox, QColorDialog)
from PyQt6.QtCore import Qt

from Core.Solvers.Visualizer import PlotStyle


class PlotStyleDialog(QDialog):
    """Dialog for selecting and customizing plot styles."""

    def __init__(self, current_style=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Style Configuration")
        self.setModal(True)
        self.resize(500, 700)

        # Store current style or use default
        self.current_style = current_style if current_style else PlotStyle()

        self.init_ui()
        self.load_current_style()

    def init_ui(self):
        layout = QVBoxLayout()

        # Preset selection
        preset_group = QGroupBox("Preset Styles")
        preset_layout = QVBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Default (Colorful)",
            "Scientific (Black & White)",
            "Publication (High DPI, LaTeX)",
            "Presentation (Bold)",
            "Custom"
        ])
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(QLabel("Select preset:"))
        preset_layout.addWidget(self.preset_combo)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Customization options
        custom_group = QGroupBox("Customization")
        custom_layout = QFormLayout()

        # Node styling
        self.node_size_spin = QDoubleSpinBox()
        self.node_size_spin.setRange(0, 200)
        self.node_size_spin.setValue(60)
        self.node_size_spin.setSingleStep(5)
        custom_layout.addRow("Node Size:", self.node_size_spin)

        self.show_nodes_check = QCheckBox()
        self.show_nodes_check.setChecked(True)
        custom_layout.addRow("Show Nodes:", self.show_nodes_check)

        # Element styling
        self.element_linewidth_spin = QDoubleSpinBox()
        self.element_linewidth_spin.setRange(0.1, 10.0)
        self.element_linewidth_spin.setValue(2.0)
        self.element_linewidth_spin.setSingleStep(0.1)
        custom_layout.addRow("Element Line Width:", self.element_linewidth_spin)

        # Block styling
        self.block_linewidth_spin = QDoubleSpinBox()
        self.block_linewidth_spin.setRange(0.1, 10.0)
        self.block_linewidth_spin.setValue(2.0)
        self.block_linewidth_spin.setSingleStep(0.1)
        custom_layout.addRow("Block Line Width:", self.block_linewidth_spin)

        self.block_alpha_spin = QDoubleSpinBox()
        self.block_alpha_spin.setRange(0.0, 1.0)
        self.block_alpha_spin.setValue(0.7)
        self.block_alpha_spin.setSingleStep(0.1)
        custom_layout.addRow("Block Transparency:", self.block_alpha_spin)

        # Figure styling
        self.figsize_width_spin = QDoubleSpinBox()
        self.figsize_width_spin.setRange(4, 20)
        self.figsize_width_spin.setValue(12)
        self.figsize_width_spin.setSingleStep(1)
        custom_layout.addRow("Figure Width (inches):", self.figsize_width_spin)

        self.figsize_height_spin = QDoubleSpinBox()
        self.figsize_height_spin.setRange(3, 16)
        self.figsize_height_spin.setValue(10)
        self.figsize_height_spin.setSingleStep(1)
        custom_layout.addRow("Figure Height (inches):", self.figsize_height_spin)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSingleStep(50)
        custom_layout.addRow("DPI (resolution):", self.dpi_spin)

        # Axes styling
        self.grid_check = QCheckBox()
        self.grid_check.setChecked(False)
        custom_layout.addRow("Show Grid:", self.grid_check)

        self.use_latex_check = QCheckBox()
        self.use_latex_check.setChecked(False)
        self.use_latex_check.setToolTip("Requires LaTeX installation on system")
        custom_layout.addRow("Use LaTeX Rendering:", self.use_latex_check)

        custom_group.setLayout(custom_layout)
        layout.addWidget(custom_group)

        # Preview info
        info_label = QLabel(
            "<b>Note:</b> Style will be applied to all viewport plots.\n"
            "Scientific and Publication styles are best for papers.\n"
            "Presentation style is optimized for slides and posters."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Add reset button
        reset_button = QPushButton("Reset to Default")
        reset_button.clicked.connect(self.reset_to_default)
        button_box.addButton(reset_button, QDialogButtonBox.ButtonRole.ResetRole)

        layout.addWidget(button_box)

        self.setLayout(layout)

    def on_preset_changed(self, index):
        """Load preset style when user selects from dropdown."""
        if index == 0:
            # Default
            style = PlotStyle()
        elif index == 1:
            # Scientific
            style = PlotStyle.scientific()
        elif index == 2:
            # Publication
            style = PlotStyle.publication()
        elif index == 3:
            # Presentation
            style = PlotStyle.presentation()
        else:
            # Custom - don't change anything
            return

        # Update UI with preset values (using correct PlotStyle attribute names)
        self.node_size_spin.setValue(style.node_size)
        self.show_nodes_check.setChecked(True)  # Default to showing nodes
        self.element_linewidth_spin.setValue(style.fem_linewidth)
        self.block_linewidth_spin.setValue(style.block_linewidth)
        self.block_alpha_spin.setValue(style.block_alpha_def)
        self.figsize_width_spin.setValue(style.figsize[0])
        self.figsize_height_spin.setValue(style.figsize[1])
        self.dpi_spin.setValue(style.dpi)
        self.grid_check.setChecked(style.grid)
        self.use_latex_check.setChecked(style.use_latex)

    def load_current_style(self):
        """Load current style into UI controls."""
        self.node_size_spin.setValue(self.current_style.node_size)
        self.show_nodes_check.setChecked(True)  # Default to showing nodes
        self.element_linewidth_spin.setValue(self.current_style.fem_linewidth)
        self.block_linewidth_spin.setValue(self.current_style.block_linewidth)
        self.block_alpha_spin.setValue(self.current_style.block_alpha_def)
        self.figsize_width_spin.setValue(self.current_style.figsize[0])
        self.figsize_height_spin.setValue(self.current_style.figsize[1])
        self.dpi_spin.setValue(self.current_style.dpi)
        self.grid_check.setChecked(self.current_style.grid)
        self.use_latex_check.setChecked(self.current_style.use_latex)

    def reset_to_default(self):
        """Reset to default style."""
        self.preset_combo.setCurrentIndex(0)
        self.on_preset_changed(0)

    def get_style(self):
        """
        Get the configured PlotStyle object.

        Returns
        -------
        PlotStyle
            Configured style based on user selections
        """
        # Create style with current UI values (using correct PlotStyle attribute names)
        style = PlotStyle(
            # Node styling
            node_size=self.node_size_spin.value(),

            # Element styling (FEM)
            fem_linewidth=self.element_linewidth_spin.value(),

            # Block styling
            block_linewidth=self.block_linewidth_spin.value(),
            block_alpha_def=self.block_alpha_spin.value(),
            block_alpha_orig=self.block_alpha_spin.value(),

            # Figure styling
            figsize=(self.figsize_width_spin.value(), self.figsize_height_spin.value()),
            dpi=self.dpi_spin.value(),

            # Axes styling
            grid=self.grid_check.isChecked(),
            use_latex=self.use_latex_check.isChecked(),
        )

        return style
