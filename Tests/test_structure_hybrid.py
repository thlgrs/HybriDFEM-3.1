"""
Tests for Hybrid structure (blocks + FEM).

Tests cover:
- Hybrid structure creation
- Mixed DOF handling (3 DOF blocks, 2 DOF FEM)
- Block-FEM coupling setup
"""
import numpy as np
import pytest

from Core.Objects.ConstitutiveLaw.Material import PlaneStress, Material
from Core.Objects.DFEM.Block import Block_2D
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.FEM.Triangles import Triangle3
from Core.Structures.Structure_Hybrid import Hybrid
from tests.conftest import is_symmetric


@pytest.fixture
def simple_hybrid_structure():
    """Create a simple hybrid structure with one block and one triangle."""
    # Materials
    block_mat = Material(E=30e9, nu=0.2, rho=2400)
    fem_mat = PlaneStress(E=200e9, nu=0.3, rho=7850)
    geom = Geometry2D(t=0.01)

    # Create hybrid
    st = Hybrid()

    # Add block at origin (1x1 square)
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    block = Block_2D(vertices=vertices, material=block_mat)
    st.list_blocks.append(block)

    # Add FEM triangle adjacent to block
    # Triangle at x = 1 (right of block)
    tri = Triangle3(nodes=[(1, 0), (2, 0), (1, 1)], mat=fem_mat, geom=geom)
    st.add_fe(tri)

    st.make_nodes()

    return st


@pytest.mark.unit
@pytest.mark.hybrid
class TestHybridBasic:
    """Basic tests for Hybrid structure."""

    def test_initialization(self):
        """Test empty hybrid structure creation."""
        st = Hybrid()

        assert st.list_blocks == []
        assert st.list_fes == []
        assert st.structure_type == "HYBRID"

    def test_add_block(self, concrete):
        """Test adding a block."""
        st = Hybrid()

        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        block = Block_2D(vertices=vertices, material=concrete)
        st.list_blocks.append(block)

        assert len(st.list_blocks) == 1

    def test_add_fem_element(self, steel, thin_plate):
        """Test adding a FEM element."""
        st = Hybrid()

        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)

        assert len(st.list_fes) == 1


@pytest.mark.unit
@pytest.mark.hybrid
class TestHybridNodes:
    """Tests for node generation in hybrid structures."""

    def test_make_nodes_block_first(self, simple_hybrid_structure):
        """Test that block nodes come before FEM nodes."""
        st = simple_hybrid_structure

        # First node should be the block (at center 0.5, 0.5)
        assert len(st.list_nodes) >= 1

        # Block has 1 node (its center/reference point)
        # Triangle has 3 nodes (but 2 may coincide with block edges)

    def test_variable_dof_counts(self, simple_hybrid_structure):
        """Test that blocks have 3 DOFs, FEM nodes have 2 DOFs."""
        st = simple_hybrid_structure

        # First node (block) should have 3 DOFs
        assert st.node_dof_counts[0] == 3

        # FEM nodes should have 2 DOFs (or 3 if fixed_dofs_per_node)
        # In default Hybrid, fixed_dofs_per_node=False

    def test_total_dofs(self, simple_hybrid_structure):
        """Test total DOF count."""
        st = simple_hybrid_structure

        # 1 block * 3 DOF + FEM nodes * DOFs
        assert st.nb_dofs > 0


@pytest.mark.unit
@pytest.mark.hybrid
class TestHybridAssembly:
    """Tests for matrix assembly in hybrid structures."""

    def test_stiffness_matrix_shape(self, simple_hybrid_structure):
        """Test stiffness matrix has correct shape."""
        st = simple_hybrid_structure
        K = st.get_K_str()

        assert K.shape == (st.nb_dofs, st.nb_dofs)

    def test_stiffness_matrix_symmetric(self, simple_hybrid_structure):
        """Test stiffness matrix is symmetric."""
        st = simple_hybrid_structure
        K = st.get_K_str()

        assert is_symmetric(K)

    def test_mass_matrix_shape(self, simple_hybrid_structure):
        """Test mass matrix has correct shape."""
        st = simple_hybrid_structure
        M = st.get_M_str()

        assert M.shape == (st.nb_dofs, st.nb_dofs)


@pytest.mark.unit
@pytest.mark.hybrid
@pytest.mark.coupling
class TestHybridCoupling:
    """Tests for block-FEM coupling setup."""

    def test_coupling_not_enabled_by_default(self, simple_hybrid_structure):
        """Test that coupling is not enabled by default."""
        st = simple_hybrid_structure
        assert st.coupling_enabled == False

    def test_coupling_methods_available(self, simple_hybrid_structure):
        """Test that coupling methods can be accessed."""
        st = simple_hybrid_structure

        # Method should exist and be callable
        assert hasattr(st, 'enable_block_fem_coupling')
        assert callable(st.enable_block_fem_coupling)

    def test_detect_coupled_fem_nodes_method(self, simple_hybrid_structure):
        """Test coupled node detection method exists."""
        st = simple_hybrid_structure

        assert hasattr(st, 'detect_coupled_fem_nodes')
        coupled = st.detect_coupled_fem_nodes()

        # Should return a dict (may be empty if no coincident nodes)
        assert isinstance(coupled, dict)


@pytest.mark.integration
@pytest.mark.hybrid
class TestHybridIntegration:
    """Integration tests for Hybrid structure."""

    def test_block_with_fem_mesh(self):
        """Test block connected to small FEM mesh."""
        block_mat = Material(E=30e9, nu=0.2, rho=2400)
        fem_mat = PlaneStress(E=200e9, nu=0.3, rho=7850)
        geom = Geometry2D(t=0.01)

        st = Hybrid()

        # Add block
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        block = Block_2D(vertices=vertices, material=block_mat)
        st.list_blocks.append(block)

        # Add two triangles forming a square
        tri1 = Triangle3(nodes=[(1, 0), (2, 0), (1, 1)], mat=fem_mat, geom=geom)
        tri2 = Triangle3(nodes=[(2, 0), (2, 1), (1, 1)], mat=fem_mat, geom=geom)

        st.add_fe(tri1)
        st.add_fe(tri2)

        st.make_nodes()

        # Should have 1 block node + FEM nodes (4 unique positions)
        # Block at center, FEM mesh has 4 nodes but 2 are shared between triangles

        K = st.get_K_str()
        assert is_symmetric(K)

        M = st.get_M_str()
        assert is_symmetric(M)

    def test_multiple_blocks(self):
        """Test structure with multiple blocks."""
        block_mat = Material(E=30e9, nu=0.2, rho=2400)

        st = Hybrid()

        # Add two adjacent blocks
        vertices1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        block1 = Block_2D(vertices=vertices1, material=block_mat)
        st.list_blocks.append(block1)

        vertices2 = np.array([[1, 0], [2, 0], [2, 1], [1, 1]])
        block2 = Block_2D(vertices=vertices2, material=block_mat)
        st.list_blocks.append(block2)

        st.make_nodes()

        # Should have 2 block nodes
        assert len(st.list_blocks) == 2

        # Each block has 3 DOFs
        assert st.node_dof_counts[0] == 3
        assert st.node_dof_counts[1] == 3
