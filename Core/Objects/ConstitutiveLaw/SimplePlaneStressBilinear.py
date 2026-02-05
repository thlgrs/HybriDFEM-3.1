import numpy as np

from Core.Objects.ConstitutiveLaw.Material import PlaneStress


class SimplePlaneStressBilinear(PlaneStress):
    """
    Simplified bilinear elastic-plastic material for plane stress conditions.

    This class extends PlaneStress to add simple plasticity:
    - Elastic behavior up to yield stress: σ = E·ε
    - Plastic behavior after yield: σ = σ_y + E_t·(ε - ε_y)

    The implementation uses a simplified approach based on maximum principal strain
    to determine the elastic/plastic state. This is not a rigorous plasticity model
    (no proper yield surface, return mapping, etc.), but it demonstrates the capability
    to handle nonlinear FEM materials with tangent stiffness.

    Parameters
    ----------
    E : float
        Young's modulus (Pa)
    nu : float
        Poisson's ratio
    rho : float
        Density (kg/m³)
    fy : float
        Yield stress (Pa)
    Et : float
        Tangent modulus in plastic range (Pa), typically 0.01-0.1 of E
    corr_fact : float, optional
        Correction factor for shear (default: 1)
    shear_def : bool, optional
        Include shear deformation (default: True)

    Attributes
    ----------
    fy : float
        Yield stress
    Et : float
        Plastic tangent modulus
    ey : float
        Yield strain (fy / E)
    yielded : bool
        True if material has yielded in current state

    Notes
    -----
    **Limitations**:
    - Simplified plasticity (uses max principal strain criterion, not von Mises)
    - No proper yield surface
    - No plastic strain state variables (uses total strain only)
    - Isotropic hardening only

    **Advantages**:
    - Simple to implement and understand
    - Demonstrates nonlinear FEM capability
    - Compatible with Newton-Raphson solvers
    - Provides tangent stiffness for iteration

    **When to Use**:
    - Demonstration and testing of nonlinear solvers
    - Preliminary analysis with softening behavior
    - Educational purposes

    **When NOT to Use**:
    - Production structural analysis (use proper plasticity models)
    - Accurate prediction of plastic deformation
    - Multi-axial loading with complex stress states

    Examples
    --------
    >>> # Create a mild steel bilinear material
    >>> mat = SimplePlaneStressBilinear(
    ...     E=200e9,      # 200 GPa
    ...     nu=0.3,
    ...     rho=7850,     # kg/m³
    ...     fy=250e6,     # 250 MPa yield stress
    ...     Et=20e9       # 20 GPa plastic modulus (10% of E)
    ... )
    >>>
    >>> # Update with strain vector
    >>> strain = np.array([0.002, 0.0, 0.0])  # 0.2% strain in x-direction
    >>> mat.update_2D(strain)
    >>>
    >>> # Get current stress
    >>> stress = mat.get_forces_2D()
    >>>
    >>> # Get tangent stiffness (will be reduced if yielded)
    >>> D_tan = mat.get_k_tan_2D()
    """

    def __init__(self, E, nu, rho, fy, Et, corr_fact=1, shear_def=True):
        """
        Initialize bilinear elastic-plastic plane stress material.

        Parameters
        ----------
        E : float
            Young's modulus (Pa), must be positive
        nu : float
            Poisson's ratio, must be in range (-1, 0.5)
        rho : float
            Density (kg/m³), must be positive
        fy : float
            Yield stress (Pa), must be positive
        Et : float
            Tangent modulus (Pa), typically 0.01-0.1 of E
        corr_fact : float, optional
            Shear correction factor (default: 1)
        shear_def : bool, optional
            Include shear deformation (default: True)
        """
        # Initialize parent PlaneStress class
        super().__init__(E, nu, rho, corr_fact, shear_def)

        # Validate plasticity parameters
        if fy <= 0:
            raise ValueError(f"Yield stress must be positive, got fy={fy}")
        if Et < 0:
            raise ValueError(f"Tangent modulus must be non-negative, got Et={Et}")
        if Et > E:
            raise ValueError(f"Tangent modulus ({Et}) cannot exceed elastic modulus ({E})")

        # Store plasticity parameters
        self.fy = fy                # Yield stress (Pa)
        self.Et = Et                # Tangent modulus in plastic range (Pa)
        self.ey = fy / E            # Yield strain

        # Plasticity state variable
        self.yielded = False        # True if currently in plastic state

        # Update tag
        self.tag = 'PLANE_STRESS_BILINEAR'

        # Commit initial state
        self.commit()

    def update_2D(self, strain_vector):
        """
        Update stress state based on strain, accounting for plasticity.

        Uses a simplified criterion based on maximum principal strain to determine
        if the material has yielded. This is not rigorous (proper plasticity would
        use stress-based yield criterion like von Mises), but it demonstrates the
        concept of tangent stiffness changing during iteration.

        Parameters
        ----------
        strain_vector : np.ndarray
            Strain vector in Voigt notation [ε_xx, ε_yy, γ_xy]

        Notes
        -----
        Simplified algorithm:
        1. Compute equivalent strain as max(|ε_xx|, |ε_yy|)
        2. If ε_equiv <= ε_y: Elastic (use D_elastic)
        3. If ε_equiv > ε_y: Plastic (use D_plastic = factor * D_elastic)
        4. Compute stress as σ = D_used * ε
        """
        # Extract strain components
        eps_xx = strain_vector[0]
        eps_yy = strain_vector[1]
        gamma_xy = strain_vector[2]

        # Simplified yield check: use maximum principal strain
        # (In proper plasticity, would use stress-based criterion)
        eps_max = max(abs(eps_xx), abs(eps_yy))

        # Determine constitutive matrix based on strain level
        if eps_max <= self.ey:
            # Elastic regime
            D_use = self.D
            self.yielded = False
        else:
            # Plastic regime - reduce stiffness
            # Apply reduction factor to elastic stiffness
            factor = self.Et / self.stiff['E']
            D_use = factor * self.D
            self.yielded = True

        # Compute stress from strain using appropriate stiffness
        stress_vector = D_use @ strain_vector

        # Update stress state
        self.stress['sigma_xx'] = stress_vector[0]
        self.stress['sigma_yy'] = stress_vector[1]
        self.stress['tau_xy'] = stress_vector[2]

        # Update strain state
        self.strain['epsilon_xx'] = strain_vector[0]
        self.strain['epsilon_yy'] = strain_vector[1]
        self.strain['gamma_xy'] = strain_vector[2]

    def get_k_tan_2D(self):
        """
        Return tangent stiffness matrix.

        For elastic state: Returns D_elastic
        For plastic state: Returns D_plastic = (Et/E) * D_elastic

        This is the key method that enables nonlinear FEM analysis. The element
        stiffness matrix Ke uses this tangent stiffness, which changes based on
        the material state (elastic vs plastic).

        Returns
        -------
        np.ndarray
            Tangent constitutive matrix (3x3)

        Notes
        -----
        The tangent stiffness is used in the Newton-Raphson iteration:
        - K_tan = ∫ B^T * D_tan * B dV
        - During iteration, D_tan changes as elements yield/unload
        - This requires updating K_tan at each iteration (stiff="tan" option)
        """
        if self.yielded:
            # Plastic tangent stiffness (reduced)
            factor = self.Et / self.stiff['E']
            return factor * self.D
        else:
            # Elastic tangent stiffness
            return self.D

    def get_k_init_2D(self):
        """
        Return initial (elastic) stiffness matrix.

        Always returns the elastic stiffness, regardless of current state.
        Used for initial stiffness-based iteration (stiff="init" option).

        Returns
        -------
        np.ndarray
            Initial constitutive matrix (3x3)
        """
        # Always return elastic stiffness for initial state
        E, nu = self.stiff0['E'], self.nu
        c = E / (1 - nu ** 2)
        return c * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1 - nu) / 2]
        ])

    def get_yield_status(self):
        """
        Return current yield status.

        Returns
        -------
        bool
            True if material has yielded, False otherwise
        """
        return self.yielded

    def get_equivalent_strain(self):
        """
        Return equivalent strain used for yield check.

        Returns
        -------
        float
            Maximum principal strain
        """
        eps_xx = self.strain['epsilon_xx']
        eps_yy = self.strain['epsilon_yy']
        return max(abs(eps_xx), abs(eps_yy))

    def __str__(self):
        """String representation of the material."""
        status = "YIELDED" if self.yielded else "ELASTIC"
        return (f"SimplePlaneStressBilinear("
                f"E={self.stiff['E']/1e9:.1f}GPa, "
                f"fy={self.fy/1e6:.1f}MPa, "
                f"Et={self.Et/1e9:.1f}GPa, "
                f"state={status})")

    def __repr__(self):
        """Developer representation of the material."""
        return (f"SimplePlaneStressBilinear(E={self.stiff['E']}, nu={self.nu}, "
                f"rho={self.rho}, fy={self.fy}, Et={self.Et})")
