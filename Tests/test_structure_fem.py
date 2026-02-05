"""
Tests for Structure_FEM.

Tests cover:
- Structure creation and element addition
- Node generation and DOF management
- Boundary conditions
- Stiffness and mass assembly
"""
import pytest

from Core.Objects.FEM.Triangles import Triangle3
from Core.Structures.Structure_FEM import Structure_FEM
from tests.conftest import is_symmetric


@pytest.mark.unit
@pytest.mark.fem
class TestStructureFEMBasic:
    """Basic tests for Structure_FEM."""

    def test_initialization(self):
        """Test empty structure creation."""
        st = Structure_FEM()

        assert st.list_fes == []
        assert st.structure_type == "FEM"

    def test_add_element(self, steel, thin_plate):
        """Test adding a triangle element."""
        st = Structure_FEM()

        nodes = [(0, 0), (1, 0), (0, 1)]
        tri = Triangle3(nodes=nodes, mat=steel, geom=thin_plate)
        st.add_fe(tri)

        assert len(st.list_fes) == 1

    def test_add_multiple_elements(self, steel, thin_plate):
        """Test adding multiple elements."""
        st = Structure_FEM()

        # Two triangles sharing an edge
        tri1 = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        tri2 = Triangle3(nodes=[(1, 0), (1, 1), (0, 1)], mat=steel, geom=thin_plate)

        st.add_fe(tri1)
        st.add_fe(tri2)

        assert len(st.list_fes) == 2


@pytest.mark.unit
@pytest.mark.fem
class TestStructureFEMNodes:
    """Tests for node generation."""

    def test_make_nodes_single_element(self, steel, thin_plate):
        """Test node generation for single element."""
        st = Structure_FEM()
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)
        st.make_nodes()

        assert len(st.list_nodes) == 3

    def test_make_nodes_shared_nodes(self, steel, thin_plate):
        """Test that coincident nodes are merged."""
        st = Structure_FEM()

        # Two triangles sharing edge (0,1)-(1,0)
        tri1 = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        tri2 = Triangle3(nodes=[(1, 0), (1, 1), (0, 1)], mat=steel, geom=thin_plate)

        st.add_fe(tri1)
        st.add_fe(tri2)
        st.make_nodes()

        # Should have 4 unique nodes, not 6
        assert len(st.list_nodes) == 4

    def test_dof_count(self, steel, thin_plate):
        """Test DOF count after node generation."""
        st = Structure_FEM()
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)
        st.make_nodes()

        # 3 nodes * 3 DOF per node (fixed_dofs_per_node=True by default)
        assert st.nb_dofs == 9


@pytest.mark.unit
@pytest.mark.fem
class TestStructureFEMBoundaryConditions:
    """Tests for boundary conditions."""

    def test_fix_node(self, steel, thin_plate):
        """Test fixing a node."""
        st = Structure_FEM()
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)
        st.make_nodes()

        # Fix node 0 (all DOFs)
        st.fix_node(0, [0, 1, 2])

        assert len(st.dof_fix) == 3

    def test_fix_node_partial(self, steel, thin_plate):
        """Test fixing specific DOFs of a node."""
        st = Structure_FEM()
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)
        st.make_nodes()

        # Fix only x and y (not rotation)
        st.fix_node(0, [0, 1])

        assert len(st.dof_fix) == 2

    def test_load_node(self, steel, thin_plate):
        """Test applying load to a node."""
        st = Structure_FEM()
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)
        st.make_nodes()

        # Apply vertical load at node 2
        st.load_node(2, 1, -1000.0)  # node 2, DOF 1 (y), force -1000

        # Check load vector
        node_dof_start = st.node_dof_offsets[2]
        assert st.P[node_dof_start + 1] == -1000.0


@pytest.mark.unit
@pytest.mark.fem
class TestStructureFEMAssembly:
    """Tests for matrix assembly."""

    def test_stiffness_matrix_shape(self, steel, thin_plate):
        """Test stiffness matrix has correct shape."""
        st = Structure_FEM()
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)
        st.make_nodes()

        K = st.get_K_str()

        assert K.shape == (st.nb_dofs, st.nb_dofs)

    def test_stiffness_matrix_symmetric(self, steel, thin_plate):
        """Test assembled stiffness matrix is symmetric."""
        st = Structure_FEM()
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)
        st.make_nodes()

        K = st.get_K_str()

        assert is_symmetric(K)

    def test_mass_matrix_shape(self, steel, thin_plate):
        """Test mass matrix has correct shape."""
        st = Structure_FEM()
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)
        st.make_nodes()

        M = st.get_M_str()

        assert M.shape == (st.nb_dofs, st.nb_dofs)

    def test_mass_matrix_symmetric(self, steel, thin_plate):
        """Test assembled mass matrix is symmetric."""
        st = Structure_FEM()
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        st.add_fe(tri)
        st.make_nodes()

        M = st.get_M_str()

        assert is_symmetric(M)


@pytest.mark.integration
@pytest.mark.fem
class TestStructureFEMIntegration:
    """Integration tests for Structure_FEM."""

    def test_two_element_mesh(self, steel, thin_plate):
        """Test a simple two-element mesh."""
        st = Structure_FEM()

        # Create square from two triangles
        tri1 = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=steel, geom=thin_plate)
        tri2 = Triangle3(nodes=[(1, 0), (1, 1), (0, 1)], mat=steel, geom=thin_plate)

        st.add_fe(tri1)
        st.add_fe(tri2)
        st.make_nodes()

        # Should have 4 nodes
        assert len(st.list_nodes) == 4

        # Stiffness matrix should be assembled correctly
        K = st.get_K_str()
        assert is_symmetric(K)

        # Check that interior nodes have contributions from both elements
        # Node at (1,0) and (0,1) are shared
