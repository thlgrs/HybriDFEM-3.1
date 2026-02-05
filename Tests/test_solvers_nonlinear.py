"""
Tests for Nonlinear Static solvers.

Tests cover:
- Force Control solver
- Displacement Control solver
- Augmented solvers (for coupling)
"""
import numpy as np
import pytest

from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.FEM.Triangles import Triangle3
from Core.Solvers.Static import StaticNonLinear
from Core.Structures.Structure_FEM import Structure_FEM


@pytest.fixture
def simple_fem_structure():
    """Single triangle element, fixed at base."""
    mat = PlaneStress(E=200e9, nu=0.3, rho=7850.0)
    geom = Geometry2D(t=0.01)
    st = Structure_FEM(fixed_dofs_per_node=False)

    # Triangle (0,0), (1,0), (0,1)
    tri = Triangle3([(0, 0), (1, 0), (0, 1)], mat, geom)
    st.add_fe(tri)
    st.make_nodes()

    # Fix nodes 0 (0,0) and 2 (0,1) -> left edge x=0
    st.fix_node(0, [0, 1])
    st.fix_node(2, [0, 1])

    # Load node 1 (1,0) -> tip
    st.add_nodal_load(1, np.array([0, -1000]))

    return st


@pytest.mark.unit
@pytest.mark.solver
def test_force_control_linear_structure(simple_fem_structure, tmp_path):
    """Test force control solver on linear structure (should converge)."""
    st = simple_fem_structure

    # Use temporary directory
    st = StaticNonLinear.solve_forcecontrol(
        st, steps=5, tol=1e-6, dir_name=str(tmp_path)
    )

    # Check displacement
    dofs = st.get_dofs_from_node(1)
    u_tip = st.U[dofs[1]]

    assert u_tip < 0
    assert np.isclose(st.P_r[dofs[1]], -1000, rtol=1e-3)


@pytest.mark.unit
@pytest.mark.solver
def test_disp_control_linear_structure(simple_fem_structure, tmp_path):
    """Test displacement control solver."""
    st = simple_fem_structure

    # Target displacement -0.01 m at Node 1, DOF y (index 1)
    target_disp = -0.01

    st = StaticNonLinear.solve_dispcontrol(
        st,
        steps=5,
        disp=target_disp,
        node=1,
        dof=1,  # Use integer index for DOF
        dir_name=str(tmp_path)
    )

    dofs = st.get_dofs_from_node(1)
    u_tip = st.U[dofs[1]]

    assert np.isclose(u_tip, target_disp, rtol=1e-4)


@pytest.mark.unit
@pytest.mark.solver
def test_force_control_augmented(tmp_path):
    """Test augmented force control solver with coupling."""
    from Core.Structures.Structure_Hybrid import Hybrid
    from Core.Objects.DFEM.Block import Block_2D

    st = Hybrid()
    mat = PlaneStress(E=200e9, nu=0.3, rho=7850)

    # Block and FEM next to each other
    # Block on left, FEM on right
    # Block Ref Point at (0,0)
    block_verts = np.array([[-1.0, -0.5], [0.0, -0.5], [0.0, 0.5], [-1.0, 0.5]])
    block = Block_2D(vertices=block_verts, material=mat, ref_point=np.array([0.0, 0.0]))
    st.list_blocks.append(block)

    # FEM Triangle
    geom = Geometry2D(t=0.01)
    tri = Triangle3([(0, 0), (1, 0), (0, 1)], mat, geom)
    st.add_fe(tri)

    st.make_nodes()

    # Enable coupling
    st.enable_block_fem_coupling(method='lagrange', tolerance=1.0)

    # Fix block (Node 0)
    st.fix_node(0, [0, 1, 2])

    # Load FEM node (Node 2 at 1,0? No, indices depend on order)
    # Node 0: Block
    # Node 1: FEM(0,0)
    # Node 2: FEM(1,0)
    # Node 3: FEM(0,1)

    # Load Node 2
    st.add_nodal_load(2, np.array([0, -100]))

    st = StaticNonLinear.solve_forcecontrol_augmented(
        st, steps=2, dir_name=str(tmp_path)
    )

    assert st.U is not None
