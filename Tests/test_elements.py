"""
Tests for FEM elements.

Tests cover:
- Triangle3 (linear triangle)
- Triangle6 (quadratic triangle)
- Shape functions
- Stiffness and mass matrices
"""
import numpy as np
import pytest

from Core.Objects.FEM.Element2D import Geometry2D
from tests.conftest import is_symmetric


@pytest.mark.unit
@pytest.mark.fem
class TestTriangle3:
    """Tests for Triangle3 (linear triangle) element."""

    def test_initialization(self, triangle3):
        """Test Triangle3 basic properties."""
        assert triangle3.nd == 3  # 3 nodes
        assert triangle3.dpn == 2  # 2 DOF per node
        assert triangle3.edof == 6  # 6 total DOFs

    def test_area(self, triangle3):
        """Test area computation for unit right triangle."""
        # Area of unit right triangle = 0.5 * 1 * 1 = 0.5
        assert np.isclose(triangle3.area, 0.5)

    def test_shape_functions_at_nodes(self, triangle3):
        """Test shape functions are 1 at their node, 0 at others."""
        # Node 1 at (xi=0, eta=0)
        N, _, _ = triangle3.N_dN(0.0, 0.0)
        assert np.allclose(N, [1.0, 0.0, 0.0])

        # Node 2 at (xi=1, eta=0)
        N, _, _ = triangle3.N_dN(1.0, 0.0)
        assert np.allclose(N, [0.0, 1.0, 0.0])

        # Node 3 at (xi=0, eta=1)
        N, _, _ = triangle3.N_dN(0.0, 1.0)
        assert np.allclose(N, [0.0, 0.0, 1.0])

    def test_shape_functions_sum_to_one(self, triangle3):
        """Test partition of unity: shape functions sum to 1."""
        # Test at centroid
        N, _, _ = triangle3.N_dN(1/3, 1/3)
        assert np.isclose(np.sum(N), 1.0)

        # Test at random point
        N, _, _ = triangle3.N_dN(0.25, 0.25)
        assert np.isclose(np.sum(N), 1.0)

    def test_stiffness_matrix_shape(self, triangle3):
        """Test stiffness matrix is 6x6."""
        Ke = triangle3.Ke()
        assert Ke.shape == (6, 6)

    def test_stiffness_matrix_symmetric(self, triangle3):
        """Test stiffness matrix is symmetric."""
        Ke = triangle3.Ke()
        assert is_symmetric(Ke)

    def test_stiffness_matrix_positive_semidefinite(self, triangle3):
        """Test stiffness matrix has 3 rigid body modes (3 zero eigenvalues)."""
        Ke = triangle3.Ke()

        # Check eigenvalues
        eigenvalues = np.linalg.eigvalsh(Ke)

        # Should have 3 zero eigenvalues (rigid body modes in 2D)
        # Use relative tolerance based on max eigenvalue
        max_eig = np.max(np.abs(eigenvalues))
        num_zero = np.sum(np.abs(eigenvalues) < max_eig * 1e-10)
        assert num_zero == 3

        # Remaining eigenvalues should be positive
        num_positive = np.sum(eigenvalues > max_eig * 1e-10)
        assert num_positive == 3

    def test_mass_matrix_shape(self, triangle3):
        """Test mass matrix is 6x6."""
        Me = triangle3.Me_consistent()
        assert Me.shape == (6, 6)

    def test_mass_matrix_symmetric(self, triangle3):
        """Test mass matrix is symmetric."""
        Me = triangle3.Me_consistent()
        assert is_symmetric(Me)

    def test_stress_for_rigid_body_motion(self, triangle3):
        """Test zero stress for pure translation."""
        # Pure x-translation: all nodes move same amount
        u = np.array([0.001, 0.0, 0.001, 0.0, 0.001, 0.0])

        stress, strain = triangle3.compute_stress(u)

        # Should give zero strain and stress
        assert np.allclose(strain, [0, 0, 0], atol=1e-12)
        assert np.allclose(stress, [0, 0, 0], atol=1e-10)


@pytest.mark.unit
@pytest.mark.fem
class TestTriangle6:
    """Tests for Triangle6 (quadratic triangle) element."""

    def test_initialization(self, triangle6):
        """Test Triangle6 basic properties."""
        assert triangle6.nd == 6  # 6 nodes
        assert triangle6.dpn == 2  # 2 DOF per node
        assert triangle6.edof == 12  # 12 total DOFs

    def test_area(self, triangle6):
        """Test area computation (based on corner nodes)."""
        # Area of unit right triangle = 0.5
        assert np.isclose(triangle6.area, 0.5)

    def test_shape_functions_at_corner_nodes(self, triangle6):
        """Test shape functions at corner nodes."""
        # Node 1 at (0, 0)
        N, _, _ = triangle6.N_dN(0.0, 0.0)
        assert np.isclose(N[0], 1.0)
        assert np.allclose(N[1:], 0.0, atol=1e-10)

        # Node 2 at (1, 0)
        N, _, _ = triangle6.N_dN(1.0, 0.0)
        assert np.isclose(N[1], 1.0)

        # Node 3 at (0, 1)
        N, _, _ = triangle6.N_dN(0.0, 1.0)
        assert np.isclose(N[2], 1.0)

    def test_shape_functions_at_midside_nodes(self, triangle6):
        """Test shape functions at mid-side nodes."""
        # Node 4 at mid of edge 1-2: (0.5, 0)
        N, _, _ = triangle6.N_dN(0.5, 0.0)
        assert np.isclose(N[3], 1.0)  # N4 = 1 at node 4
        assert np.isclose(N[0], 0.0)  # N1 = 0 at node 4
        assert np.isclose(N[1], 0.0)  # N2 = 0 at node 4

    def test_shape_functions_sum_to_one(self, triangle6):
        """Test partition of unity for quadratic element."""
        # Test at centroid
        N, _, _ = triangle6.N_dN(1/3, 1/3)
        assert np.isclose(np.sum(N), 1.0)

        # Test at midpoint
        N, _, _ = triangle6.N_dN(0.5, 0.25)
        assert np.isclose(np.sum(N), 1.0)

    def test_stiffness_matrix_shape(self, triangle6):
        """Test stiffness matrix is 12x12."""
        Ke = triangle6.Ke()
        assert Ke.shape == (12, 12)

    def test_stiffness_matrix_symmetric(self, triangle6):
        """Test stiffness matrix is symmetric."""
        Ke = triangle6.Ke()
        # Use relaxed tolerance for quadratic elements (more numerical integration)
        assert is_symmetric(Ke, tol=1e-6)

    def test_quadrature_rule(self, triangle6):
        """Test 3-point quadrature rule for Triangle6."""
        XI, ETA, W = triangle6.quad_rule()

        assert len(XI) == 3
        assert len(ETA) == 3
        assert len(W) == 3

        # Weights sum to 0.5 (area of reference triangle)
        assert np.isclose(np.sum(W), 0.5)


@pytest.mark.unit
@pytest.mark.fem
class TestGeometry2D:
    """Tests for Geometry2D class."""

    def test_initialization(self):
        """Test geometry initialization."""
        geom = Geometry2D(t=0.025)
        assert geom.t == 0.025

    def test_different_thicknesses(self, thin_plate, thick_plate):
        """Test different thickness configurations."""
        assert thin_plate.t == 0.01
        assert thick_plate.t == 0.10
