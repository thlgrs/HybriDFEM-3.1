"""
Tests for Quadrilateral FEM elements.

Tests cover:
- Quad4 (bilinear quadrilateral)
- Quad8 (serendipity quadratic quadrilateral)
- Quad9 (Lagrangian biquadratic quadrilateral)
- Shape functions
- Stiffness matrices
"""
import numpy as np
import pytest

from Core.Objects.FEM.Quads import Quad4, Quad8, Quad9
from tests.conftest import is_symmetric


@pytest.fixture
def quad4_nodes():
    """Unit square nodes for Quad4 (counter-clockwise)."""
    return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


@pytest.fixture
def quad8_nodes():
    """Unit square nodes for Quad8 (corners + mid-sides)."""
    return [
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),  # Corners
        (0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)  # Mid-sides
    ]


@pytest.fixture
def quad9_nodes():
    """Unit square nodes for Quad9 (corners + mid-sides + center)."""
    return [
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0),  # Corners
        (0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5),  # Mid-sides
        (0.5, 0.5)  # Center
    ]


@pytest.fixture
def quad4(quad4_nodes, steel, thin_plate):
    return Quad4(nodes=quad4_nodes, mat=steel, geom=thin_plate)


@pytest.fixture
def quad8(quad8_nodes, steel, thin_plate):
    return Quad8(nodes=quad8_nodes, mat=steel, geom=thin_plate)


@pytest.fixture
def quad9(quad9_nodes, steel, thin_plate):
    return Quad9(nodes=quad9_nodes, mat=steel, geom=thin_plate)


@pytest.mark.unit
@pytest.mark.fem
class TestQuad4:
    """Tests for Quad4 element."""

    def test_initialization(self, quad4):
        assert quad4.nd == 4
        assert quad4.dpn == 2
        assert quad4.edof == 8

    def test_shape_functions_at_nodes(self, quad4):
        """Test N_i = 1 at node i, 0 at others."""
        # Corners in natural coords: (-1,-1), (1,-1), (1,1), (-1,1)
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]

        for i, (xi, eta) in enumerate(corners):
            N, _, _ = quad4.N_dN(xi, eta)
            expected = np.zeros(4)
            expected[i] = 1.0
            assert np.allclose(N, expected)

    def test_partition_of_unity(self, quad4):
        """Test sum(N) = 1."""
        N, _, _ = quad4.N_dN(0.0, 0.0)  # Center
        assert np.isclose(np.sum(N), 1.0)

        N, _, _ = quad4.N_dN(0.5, -0.2)  # Random point
        assert np.isclose(np.sum(N), 1.0)

    def test_stiffness_matrix(self, quad4):
        K = quad4.Ke()
        assert K.shape == (8, 8)
        assert is_symmetric(K)
        # Rigid body modes check (3 zero eigenvalues for 2D)
        eigenvalues = np.linalg.eigvalsh(K)
        # For Quad4 with full integration (2x2), we expect 3 rigid body modes
        max_eig = np.max(np.abs(eigenvalues))
        num_zero = np.sum(np.abs(eigenvalues) < max_eig * 1e-10)
        assert num_zero == 3

    def test_area(self, quad4):
        # Unit square area = 1
        assert np.isclose(quad4.area, 1.0)


@pytest.mark.unit
@pytest.mark.fem
class TestQuad8:
    """Tests for Quad8 element."""

    def test_initialization(self, quad8):
        assert quad8.nd == 8
        assert quad8.edof == 16

    def test_shape_functions_at_nodes(self, quad8):
        # Test first 4 nodes (corners)
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for i, (xi, eta) in enumerate(corners):
            N, _, _ = quad8.N_dN(xi, eta)
            assert np.isclose(N[i], 1.0)
            assert np.isclose(np.sum(np.abs(N)) - 1.0, 0.0)  # Sum of abs might be > 1 if some negative?
            # Actually partition of unity holds, sum(N)=1. But N_j should be 0 for j!=i
            N_others = np.delete(N, i)
            assert np.allclose(N_others, 0.0, atol=1e-10)

        # Test mid-side nodes (indices 4-7)
        midsides = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for i, (xi, eta) in enumerate(midsides):
            node_idx = i + 4
            N, _, _ = quad8.N_dN(xi, eta)
            assert np.isclose(N[node_idx], 1.0)
            N_others = np.delete(N, node_idx)
            assert np.allclose(N_others, 0.0, atol=1e-10)

    def test_partition_of_unity(self, quad8):
        N, _, _ = quad8.N_dN(0.3, 0.4)
        assert np.isclose(np.sum(N), 1.0)

    def test_stiffness_matrix(self, quad8):
        K = quad8.Ke()
        assert K.shape == (16, 16)
        # Use relaxed tolerance for quadratic elements
        assert is_symmetric(K, tol=1e-6)
        # Check rigid body modes
        eigenvalues = np.linalg.eigvalsh(K)
        max_eig = np.max(np.abs(eigenvalues))
        num_zero = np.sum(np.abs(eigenvalues) < max_eig * 1e-10)
        assert num_zero == 3


@pytest.mark.unit
@pytest.mark.fem
class TestQuad9:
    """Tests for Quad9 element."""

    def test_initialization(self, quad9):
        assert quad9.nd == 9
        assert quad9.edof == 18

    def test_shape_functions_center(self, quad9):
        """Test center node (index 8) at (0,0)."""
        N, _, _ = quad9.N_dN(0.0, 0.0)
        assert np.isclose(N[8], 1.0)
        assert np.allclose(np.delete(N, 8), 0.0)

    def test_partition_of_unity(self, quad9):
        N, _, _ = quad9.N_dN(0.1, -0.7)
        assert np.isclose(np.sum(N), 1.0)

    def test_stiffness_matrix(self, quad9):
        K = quad9.Ke()
        assert K.shape == (18, 18)
        # Use relaxed tolerance
        assert is_symmetric(K, tol=1e-6)

        eigenvalues = np.linalg.eigvalsh(K)
        max_eig = np.max(np.abs(eigenvalues))
        num_zero = np.sum(np.abs(eigenvalues) < max_eig * 1e-10)
        assert num_zero == 3
