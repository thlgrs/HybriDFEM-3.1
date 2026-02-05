"""
Tests for Static solver.

Tests cover:
- Linear static analysis
- Boundary condition application
- Load application
- Solution verification
"""
import numpy as np
import pytest

from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.FEM.Triangles import Triangle3
from Core.Solvers.Static import StaticLinear
from Core.Structures.Structure_FEM import Structure_FEM


@pytest.fixture
def cantilever_beam():
    """
    Create a simple cantilever-like structure from triangles.

    Fixed at left edge (x=0), load at right edge (x=2).
    Uses 2 DOFs per node (standard for 2D solid elements).
    """
    mat = PlaneStress(E=200e9, nu=0.3, rho=7850)
    geom = Geometry2D(t=0.01)

    # Use fixed_dofs_per_node=False for pure 2D solid elements (2 DOF per node)
    st = Structure_FEM(fixed_dofs_per_node=False)

    # Create a 2x1 rectangular mesh from 4 triangles
    # Bottom-left triangle
    st.add_fe(Triangle3(nodes=[(0, 0), (1, 0), (0, 0.5)], mat=mat, geom=geom))
    # Bottom-right triangle
    st.add_fe(Triangle3(nodes=[(1, 0), (1, 0.5), (0, 0.5)], mat=mat, geom=geom))
    # Top-left triangle
    st.add_fe(Triangle3(nodes=[(1, 0), (2, 0), (1, 0.5)], mat=mat, geom=geom))
    # Top-right triangle
    st.add_fe(Triangle3(nodes=[(2, 0), (2, 0.5), (1, 0.5)], mat=mat, geom=geom))

    st.make_nodes()

    return st


@pytest.mark.unit
@pytest.mark.solver
class TestStaticSolverBasic:
    """Basic tests for Static solver."""

    def test_solve_method_exists(self):
        """Test that StaticLinear.solve method exists."""
        assert hasattr(StaticLinear, 'solve')

    def test_solve_returns_structure(self, cantilever_beam):
        """Test that solve returns a structure."""
        st = cantilever_beam

        # Fix left edge nodes (x=0) - both DOFs
        for node_id in range(len(st.list_nodes)):
            node_pos = st.list_nodes[node_id]
            if np.isclose(node_pos[0], 0.0):
                st.fix_node(node_id, [0, 1])  # Fix ux, uy

        # Apply load at right edge (x=2)
        for node_id in range(len(st.list_nodes)):
            node_pos = st.list_nodes[node_id]
            if np.isclose(node_pos[0], 2.0):
                st.load_node(node_id, 1, -1000.0)  # Downward force

        # Solve
        st_solved = StaticLinear.solve(st)

        assert st_solved is not None
        assert hasattr(st_solved, 'U')


@pytest.mark.integration
@pytest.mark.solver
class TestStaticSolverResults:
    """Tests for solver result verification."""

    def test_fixed_nodes_zero_displacement(self, cantilever_beam):
        """Test that fixed nodes have zero displacement."""
        st = cantilever_beam

        # Fix left edge
        fixed_nodes = []
        for node_id in range(len(st.list_nodes)):
            node_pos = st.list_nodes[node_id]
            if np.isclose(node_pos[0], 0.0):
                st.fix_node(node_id, [0, 1])
                fixed_nodes.append(node_id)

        # Apply load
        for node_id in range(len(st.list_nodes)):
            node_pos = st.list_nodes[node_id]
            if np.isclose(node_pos[0], 2.0):
                st.load_node(node_id, 1, -1000.0)

        # Solve
        st_solved = StaticLinear.solve(st)

        # Check fixed DOFs are zero
        for node_id in fixed_nodes:
            dof_start = st_solved.node_dof_offsets[node_id]
            dof_count = st_solved.node_dof_counts[node_id]
            for i in range(dof_count):
                assert np.isclose(st_solved.U[dof_start + i], 0.0, atol=1e-12)

    def test_loaded_nodes_nonzero_displacement(self, cantilever_beam):
        """Test that loaded nodes have non-zero displacement."""
        st = cantilever_beam

        # Fix left edge
        for node_id in range(len(st.list_nodes)):
            node_pos = st.list_nodes[node_id]
            if np.isclose(node_pos[0], 0.0):
                st.fix_node(node_id, [0, 1])

        # Apply load and track loaded nodes
        loaded_nodes = []
        for node_id in range(len(st.list_nodes)):
            node_pos = st.list_nodes[node_id]
            if np.isclose(node_pos[0], 2.0):
                st.load_node(node_id, 1, -1000.0)
                loaded_nodes.append(node_id)

        # Solve
        st_solved = StaticLinear.solve(st)

        # Check loaded nodes have displacement
        for node_id in loaded_nodes:
            dof_start = st_solved.node_dof_offsets[node_id]
            # Y displacement should be negative (downward load)
            assert st_solved.U[dof_start + 1] < 0

    def test_equilibrium(self, cantilever_beam):
        """Test that internal forces equal external forces at equilibrium."""
        st = cantilever_beam

        # Fix left edge
        for node_id in range(len(st.list_nodes)):
            node_pos = st.list_nodes[node_id]
            if np.isclose(node_pos[0], 0.0):
                st.fix_node(node_id, [0, 1])

        # Apply load
        for node_id in range(len(st.list_nodes)):
            node_pos = st.list_nodes[node_id]
            if np.isclose(node_pos[0], 2.0):
                st.load_node(node_id, 1, -1000.0)

        # Solve
        st_solved = StaticLinear.solve(st)

        # Check residual is small at free DOFs
        st_solved.get_P_r()
        residual = st_solved.P - st_solved.P_r

        # Residual at free DOFs should be near zero
        free_dofs = st_solved.dof_free
        residual_free = residual[free_dofs]

        assert np.allclose(residual_free, 0.0, atol=1e-6)


@pytest.mark.unit
@pytest.mark.solver
class TestStaticSolverSimple:
    """Simple test cases for solver verification."""

    def test_single_triangle_point_load(self):
        """Test single triangle with point load."""
        mat = PlaneStress(E=200e9, nu=0.3, rho=7850)
        geom = Geometry2D(t=0.01)

        # Use 2 DOF per node for pure 2D elements
        st = Structure_FEM(fixed_dofs_per_node=False)
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=mat, geom=geom)
        st.add_fe(tri)
        st.make_nodes()

        # Fix nodes 0 and 1 (bottom edge) - only x, y DOFs
        st.fix_node(0, [0, 1])
        st.fix_node(1, [0, 1])

        # Load node 2 (top)
        st.load_node(2, 1, -1000.0)

        # Solve
        st_solved = StaticLinear.solve(st)

        # Node 2 should move down
        dof_start = st_solved.node_dof_offsets[2]
        v_disp = st_solved.U[dof_start + 1]
        assert v_disp < 0

    def test_no_load_no_displacement(self):
        """Test that no load gives no displacement."""
        mat = PlaneStress(E=200e9, nu=0.3, rho=7850)
        geom = Geometry2D(t=0.01)

        st = Structure_FEM(fixed_dofs_per_node=False)
        tri = Triangle3(nodes=[(0, 0), (1, 0), (0, 1)], mat=mat, geom=geom)
        st.add_fe(tri)
        st.make_nodes()

        # Fix all nodes (all DOFs)
        st.fix_node(0, [0, 1])
        st.fix_node(1, [0, 1])
        st.fix_node(2, [0, 1])

        # No load applied

        # Solve
        st_solved = StaticLinear.solve(st)

        # All displacements should be zero
        assert np.allclose(st_solved.U, 0.0)
