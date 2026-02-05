"""
Tests for Block_2D (rigid block element).

Tests cover:
- Block initialization
- Geometric properties (area, center, inertia)
- Mass matrix
- Rigid body kinematics
- Constraint matrix for coupling
"""
import numpy as np
import pytest

from Core.Objects.DFEM.Block import Block_2D


@pytest.mark.unit
@pytest.mark.dfem
class TestBlock2D:
    """Tests for Block_2D rigid block element."""

    def test_initialization(self, square_block):
        """Test block initialization."""
        assert square_block.nb_vertices == 4
        assert square_block.DOFS_PER_NODE == 3  # ux, uy, rotation_z

    def test_area_square(self, square_block):
        """Test area for unit square."""
        assert np.isclose(square_block.A, 1.0)

    def test_area_rectangle(self, rectangular_vertices, concrete):
        """Test area for 2x1 rectangle."""
        block = Block_2D(vertices=rectangular_vertices, material=concrete)
        assert np.isclose(block.A, 2.0)

    def test_center_square(self, square_block):
        """Test center of unit square at (0.5, 0.5)."""
        assert np.allclose(square_block.center, [0.5, 0.5])

    def test_center_rectangle(self, rectangular_vertices, concrete):
        """Test center of 2x1 rectangle at (1.0, 0.5)."""
        block = Block_2D(vertices=rectangular_vertices, material=concrete)
        assert np.allclose(block.center, [1.0, 0.5])

    def test_mass(self, square_block, concrete):
        """Test mass = rho * A * thickness."""
        # mass = 2400 * 1.0 * 1.0 = 2400 kg
        expected_mass = 2400.0 * 1.0 * 1.0
        assert np.isclose(square_block.m, expected_mass)

    def test_mass_matrix_shape(self, square_block):
        """Test mass matrix is 3x3."""
        M = square_block.get_mass()
        assert M.shape == (3, 3)

    def test_mass_matrix_diagonal(self, square_block):
        """Test mass matrix is diagonal: diag(m, m, I)."""
        M = square_block.get_mass()

        # Check diagonal structure
        assert M[0, 0] == square_block.m
        assert M[1, 1] == square_block.m
        assert M[2, 2] == square_block.I

        # Check off-diagonal is zero
        assert M[0, 1] == 0
        assert M[0, 2] == 0
        assert M[1, 2] == 0

    def test_rotational_inertia_positive(self, square_block):
        """Test rotational inertia is positive."""
        assert square_block.I > 0

    def test_reference_point_default(self, square_block):
        """Test default reference point is center."""
        assert np.allclose(square_block.ref_point, square_block.center)

    def test_reference_point_custom(self, square_vertices, concrete):
        """Test custom reference point."""
        custom_ref = np.array([0.0, 0.0])
        block = Block_2D(vertices=square_vertices, material=concrete,
                        ref_point=custom_ref)

        assert np.allclose(block.ref_point, custom_ref)

    def test_displacement_at_point_translation(self, square_block):
        """Test displacement at point for pure translation."""
        # Pure x-translation
        square_block.disps = np.array([0.1, 0.0, 0.0])

        point = np.array([0.0, 0.0])
        u = square_block.displacement_at_point(point)

        assert np.allclose(u, [0.1, 0.0])

    def test_displacement_at_point_rotation(self, square_block):
        """Test displacement at point for pure rotation."""
        # Pure rotation about center (0.5, 0.5)
        theta = 0.01  # small angle
        square_block.disps = np.array([0.0, 0.0, theta])

        # Point at corner (0, 0), relative to center (-0.5, -0.5)
        point = np.array([0.0, 0.0])
        u = square_block.displacement_at_point(point)

        # Small angle: u = -theta * (y - yc), v = theta * (x - xc)
        # xc, yc = 0.5, 0.5
        expected_u = -theta * (0.0 - 0.5)  # = 0.005
        expected_v = theta * (0.0 - 0.5)   # = -0.005

        assert np.allclose(u, [expected_u, expected_v])

    def test_constraint_matrix_for_node(self, square_block):
        """Test constraint matrix for coupling."""
        node_pos = np.array([0.0, 0.0])
        C = square_block.constraint_matrix_for_node(node_pos)

        # C should be 2x3: [u_node, v_node] = C @ [u_block, v_block, theta]
        assert C.shape == (2, 3)

        # Check structure for point at (0,0) with ref_point at (0.5, 0.5)
        # u_node = u_block - (y-yc)*theta = u_block - (-0.5)*theta
        # v_node = v_block + (x-xc)*theta = v_block + (-0.5)*theta

        expected = np.array([
            [1, 0, 0.5],   # u = 1*u + 0*v + (-(0-0.5))*theta
            [0, 1, -0.5]   # v = 0*u + 1*v + (0-0.5)*theta
        ])
        assert np.allclose(C, expected)

    def test_valid_polygon(self, square_block):
        """Test valid (non-self-intersecting) polygon."""
        assert square_block.is_valid_polygon() == True

    def test_min_enclosing_circle(self, square_block):
        """Test minimum enclosing circle exists."""
        assert hasattr(square_block, 'circle_center')
        assert hasattr(square_block, 'circle_radius')
        assert square_block.circle_radius > 0

    def test_triangle_block(self, concrete):
        """Test triangular block."""
        vertices = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ])
        block = Block_2D(vertices=vertices, material=concrete)

        # Area of triangle = 0.5 * base * height = 0.5 * 1 * 1 = 0.5
        assert np.isclose(block.A, 0.5)
        assert block.nb_vertices == 3


@pytest.mark.unit
@pytest.mark.dfem
class TestBlockDOFs:
    """Tests for block DOF handling."""

    def test_dofs_per_node(self, square_block):
        """Test block has 3 DOFs per node."""
        assert Block_2D.DOFS_PER_NODE == 3

    def test_make_connect(self, square_block):
        """Test DOF connectivity setup."""
        square_block.make_connect(index=0)

        # DOFs should be [0, 1, 2] for first block
        assert np.allclose(square_block.dofs, [0, 1, 2])

    def test_make_connect_second_block(self, square_vertices, concrete):
        """Test DOF connectivity for second block."""
        block = Block_2D(vertices=square_vertices, material=concrete)
        block.make_connect(index=1)

        # DOFs should be [3, 4, 5] for second block
        assert np.allclose(block.dofs, [3, 4, 5])
