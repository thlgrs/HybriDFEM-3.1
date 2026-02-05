"""
Contact Configuration Dialog for Block Structures

This dialog allows users to configure contact detection and contact laws
for rigid block structures (Structure_Block and Hybrid).
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QFormLayout, QGroupBox,
                             QDoubleSpinBox, QSpinBox, QRadioButton, QButtonGroup,
                             QDialogButtonBox, QLabel)


class ContactConfigDialog(QDialog):
    """Dialog for configuring contact mechanics in block structures."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Contact Configuration")
        self.setModal(True)
        self.resize(500, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Introduction label
        intro_label = QLabel(
            "<b>Configure Contact Detection and Contact Laws</b><br>"
            "Contact mechanics enables rigid blocks to interact through "
            "compression, friction, and cohesion."
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        # Contact Detection Parameters
        detection_group = QGroupBox("Contact Detection")
        detection_layout = QFormLayout()

        self.tolerance_input = QDoubleSpinBox()
        self.tolerance_input.setRange(1e-12, 1e-6)
        self.tolerance_input.setValue(1e-9)
        self.tolerance_input.setDecimals(12)
        self.tolerance_input.setSingleStep(1e-10)
        self.tolerance_input.setToolTip("Geometric tolerance for contact detection (m)")
        detection_layout.addRow("Tolerance [m]:", self.tolerance_input)

        self.margin_input = QDoubleSpinBox()
        self.margin_input.setRange(0.0, 0.1)
        self.margin_input.setValue(0.01)
        self.margin_input.setDecimals(4)
        self.margin_input.setSingleStep(0.001)
        self.margin_input.setToolTip("Search margin for interface detection (m)")
        detection_layout.addRow("Margin [m]:", self.margin_input)

        self.nb_cps_input = QSpinBox()
        self.nb_cps_input.setRange(2, 25)
        self.nb_cps_input.setValue(5)
        self.nb_cps_input.setToolTip("Number of contact points per interface")
        detection_layout.addRow("Points per Face:", self.nb_cps_input)

        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)

        # Contact Law Selection
        law_group = QGroupBox("Contact Law")
        law_layout = QVBoxLayout()

        self.law_button_group = QButtonGroup(self)

        self.notension_radio = QRadioButton("NoTension_EP (Elastic No-Tension)")
        self.notension_radio.setChecked(True)
        self.notension_radio.setToolTip(
            "Linear elastic contact with zero stiffness in tension.\n"
            "Suitable for simple compression-only interfaces."
        )
        self.law_button_group.addButton(self.notension_radio, 0)
        law_layout.addWidget(self.notension_radio)

        self.coulomb_radio = QRadioButton("Coulomb (Friction with Plasticity)")
        self.coulomb_radio.setToolTip(
            "Mohr-Coulomb plasticity with friction, cohesion, and dilation.\n"
            "Suitable for frictional sliding and complex contact behavior."
        )
        self.law_button_group.addButton(self.coulomb_radio, 1)
        law_layout.addWidget(self.coulomb_radio)

        law_group.setLayout(law_layout)
        layout.addWidget(law_group)

        # Connect radio button to update parameter visibility
        self.notension_radio.toggled.connect(self.update_parameter_visibility)

        # NoTension_EP Parameters
        self.notension_group = QGroupBox("NoTension_EP Parameters")
        notension_layout = QFormLayout()

        self.kn_notension_input = QDoubleSpinBox()
        self.kn_notension_input.setRange(1e8, 1e12)
        self.kn_notension_input.setValue(1e10)
        self.kn_notension_input.setDecimals(2)
        self.kn_notension_input.setSingleStep(1e9)
        self.kn_notension_input.setToolTip("Normal stiffness (compression only) [N/m]")
        notension_layout.addRow("Normal Stiffness kn [N/m]:", self.kn_notension_input)

        self.ks_notension_input = QDoubleSpinBox()
        self.ks_notension_input.setRange(1e7, 1e11)
        self.ks_notension_input.setValue(1e9)
        self.ks_notension_input.setDecimals(2)
        self.ks_notension_input.setSingleStep(1e8)
        self.ks_notension_input.setToolTip("Shear stiffness (elastic) [N/m]")
        notension_layout.addRow("Shear Stiffness ks [N/m]:", self.ks_notension_input)

        self.notension_group.setLayout(notension_layout)
        layout.addWidget(self.notension_group)

        # Coulomb Parameters
        self.coulomb_group = QGroupBox("Coulomb Parameters")
        coulomb_layout = QFormLayout()

        self.kn_coulomb_input = QDoubleSpinBox()
        self.kn_coulomb_input.setRange(1e8, 1e12)
        self.kn_coulomb_input.setValue(1e10)
        self.kn_coulomb_input.setDecimals(2)
        self.kn_coulomb_input.setSingleStep(1e9)
        self.kn_coulomb_input.setToolTip("Normal stiffness (compression only) [N/m]")
        coulomb_layout.addRow("Normal Stiffness kn [N/m]:", self.kn_coulomb_input)

        self.ks_coulomb_input = QDoubleSpinBox()
        self.ks_coulomb_input.setRange(1e7, 1e11)
        self.ks_coulomb_input.setValue(1e9)
        self.ks_coulomb_input.setDecimals(2)
        self.ks_coulomb_input.setSingleStep(1e8)
        self.ks_coulomb_input.setToolTip("Shear stiffness (elastic) [N/m]")
        coulomb_layout.addRow("Shear Stiffness ks [N/m]:", self.ks_coulomb_input)

        self.mu_input = QDoubleSpinBox()
        self.mu_input.setRange(0.0, 1.5)
        self.mu_input.setValue(0.3)
        self.mu_input.setDecimals(2)
        self.mu_input.setSingleStep(0.05)
        self.mu_input.setToolTip("Friction coefficient (tan(φ) for Mohr-Coulomb)")
        coulomb_layout.addRow("Friction Coefficient μ:", self.mu_input)

        self.c_input = QDoubleSpinBox()
        self.c_input.setRange(0.0, 1e6)
        self.c_input.setValue(0.0)
        self.c_input.setDecimals(2)
        self.c_input.setSingleStep(1000.0)
        self.c_input.setToolTip("Cohesion (adhesive strength) [Pa]")
        coulomb_layout.addRow("Cohesion c [Pa]:", self.c_input)

        self.psi_input = QDoubleSpinBox()
        self.psi_input.setRange(0.0, 45.0)
        self.psi_input.setValue(0.0)
        self.psi_input.setDecimals(1)
        self.psi_input.setSingleStep(1.0)
        self.psi_input.setToolTip("Dilation angle (volume change during shear) [degrees]")
        coulomb_layout.addRow("Dilation Angle ψ [°]:", self.psi_input)

        self.ft_input = QDoubleSpinBox()
        self.ft_input.setRange(0.0, 1e6)
        self.ft_input.setValue(0.0)
        self.ft_input.setDecimals(2)
        self.ft_input.setSingleStep(1000.0)
        self.ft_input.setToolTip("Tensile cutoff (max tension before failure) [Pa]")
        coulomb_layout.addRow("Tensile Cutoff ft [Pa]:", self.ft_input)

        self.coulomb_group.setLayout(coulomb_layout)
        layout.addWidget(self.coulomb_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Set initial visibility
        self.update_parameter_visibility()

    def update_parameter_visibility(self):
        """Show/hide parameter groups based on selected contact law."""
        is_notension = self.notension_radio.isChecked()
        self.notension_group.setVisible(is_notension)
        self.coulomb_group.setVisible(not is_notension)

    def get_data(self):
        """Return configuration as dictionary."""
        data = {
            'enabled': True,
            'tolerance': self.tolerance_input.value(),
            'margin': self.margin_input.value(),
            'nb_cps': self.nb_cps_input.value(),
        }

        if self.notension_radio.isChecked():
            data['law'] = 'NoTension_EP'
            data['kn'] = self.kn_notension_input.value()
            data['ks'] = self.ks_notension_input.value()
            data['mu'] = 0.0
            data['c'] = 0.0
            data['psi'] = 0.0
            data['ft'] = 0.0
        else:  # Coulomb
            data['law'] = 'Coulomb'
            data['kn'] = self.kn_coulomb_input.value()
            data['ks'] = self.ks_coulomb_input.value()
            data['mu'] = self.mu_input.value()
            data['c'] = self.c_input.value()
            data['psi'] = self.psi_input.value()
            data['ft'] = self.ft_input.value()

        return data

    def set_data(self, data):
        """Load configuration from dictionary."""
        self.tolerance_input.setValue(data.get('tolerance', 1e-9))
        self.margin_input.setValue(data.get('margin', 0.01))
        self.nb_cps_input.setValue(data.get('nb_cps', 5))

        law = data.get('law', 'NoTension_EP')
        if law == 'NoTension_EP':
            self.notension_radio.setChecked(True)
            self.kn_notension_input.setValue(data.get('kn', 1e10))
            self.ks_notension_input.setValue(data.get('ks', 1e9))
        else:
            self.coulomb_radio.setChecked(True)
            self.kn_coulomb_input.setValue(data.get('kn', 1e10))
            self.ks_coulomb_input.setValue(data.get('ks', 1e9))
            self.mu_input.setValue(data.get('mu', 0.3))
            self.c_input.setValue(data.get('c', 0.0))
            self.psi_input.setValue(data.get('psi', 0.0))
            self.ft_input.setValue(data.get('ft', 0.0))

        self.update_parameter_visibility()
