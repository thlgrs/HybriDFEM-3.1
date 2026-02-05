"""
Tests for ConstitutiveLaw classes.

Tests cover:
- ConstitutiveLaw initialization
- PlaneStress constitutive matrix
- PlaneStrain constitutive matrix
- ConstitutiveLaw properties
"""
import numpy as np
import pytest

from Core.Objects.ConstitutiveLaw.Material import Material, PlaneStress, PlaneStrain
from tests.conftest import is_symmetric


@pytest.mark.unit
@pytest.mark.material
class TestMaterial:
    """Tests for base ConstitutiveLaw class."""

    def test_initialization(self):
        """Test ConstitutiveLaw can be created with basic properties."""
        mat = Material(E=200e9, nu=0.3, rho=7850)

        assert mat.stiff['E'] == 200e9
        assert mat.rho == 7850

    def test_get_forces_initial(self):
        """Test initial forces are zero."""
        mat = Material(E=200e9, nu=0.3)
        forces = mat.get_forces()

        assert np.allclose(forces, [0, 0])

    def test_update(self):
        """Test stress-strain update."""
        mat = Material(E=200e9, nu=0.3)

        # Apply strain increment
        mat.update([0.001, 0.0])  # 0.1% strain

        # Check stress = E * strain
        expected_stress = 200e9 * 0.001
        assert np.isclose(mat.stress['s'], expected_stress)


@pytest.mark.unit
@pytest.mark.material
class TestPlaneStress:
    """Tests for PlaneStress material."""

    def test_initialization(self, steel):
        """Test PlaneStress properties."""
        assert steel.stiff['E'] == 200e9
        assert steel.nu == 0.3
        assert steel.rho == 7850.0
        assert steel.tag == 'PLANE_STRESS'

    def test_constitutive_matrix_shape(self, steel):
        """Test D matrix is 3x3."""
        D = steel.D
        assert D.shape == (3, 3)

    def test_constitutive_matrix_symmetric(self, steel):
        """Test D matrix is symmetric."""
        D = steel.D
        assert is_symmetric(D)

    def test_constitutive_matrix_values(self):
        """Test D matrix values for known material."""
        # Simple case: E=1, nu=0 => D = [[1,0,0],[0,1,0],[0,0,0.5]]
        mat = PlaneStress(E=1.0, nu=0.0, rho=1.0)
        D = mat.D

        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.5]
        ])
        assert np.allclose(D, expected)

    def test_stress_from_strain(self, steel):
        """Test stress computation from strain."""
        # Apply uniaxial strain in x
        strain = np.array([0.001, 0.0, 0.0])  # eps_xx only
        stress = steel.D @ strain

        # For plane stress: sigma_xx = E/(1-nu^2) * eps_xx
        E, nu = 200e9, 0.3
        expected_sigma_xx = E / (1 - nu**2) * 0.001

        assert np.isclose(stress[0], expected_sigma_xx)


@pytest.mark.unit
@pytest.mark.material
class TestPlaneStrain:
    """Tests for PlaneStrain material."""

    def test_initialization(self):
        """Test PlaneStrain properties."""
        mat = PlaneStrain(E=200e9, nu=0.3, rho=7850)

        assert mat.stiff['E'] == 200e9
        assert mat.nu == 0.3
        assert mat.tag == 'PLANE_STRAIN'

    def test_constitutive_matrix_shape(self):
        """Test D matrix is 3x3."""
        mat = PlaneStrain(E=200e9, nu=0.3, rho=7850)
        D = mat.D
        assert D.shape == (3, 3)

    def test_constitutive_matrix_symmetric(self):
        """Test D matrix is symmetric."""
        mat = PlaneStrain(E=200e9, nu=0.3, rho=7850)
        D = mat.D
        assert is_symmetric(D)

    def test_plane_strain_stiffer_than_plane_stress(self):
        """Test that plane strain is stiffer than plane stress."""
        E, nu, rho = 200e9, 0.3, 7850
        mat_stress = PlaneStress(E, nu, rho)
        mat_strain = PlaneStrain(E, nu, rho)

        # For same strain, plane strain gives higher stress
        strain = np.array([0.001, 0.0, 0.0])
        stress_ps = mat_stress.D @ strain
        stress_pn = mat_strain.D @ strain

        assert stress_pn[0] > stress_ps[0]
