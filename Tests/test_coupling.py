"""
Tests for Block-FEM coupling methods.

Tests cover:
- Coupling method configuration
- Constraint Coupling (Condensation)
- Penalty Coupling
- Lagrange Coupling
- Mortar Coupling
"""
import numpy as np
import pytest

from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.Coupling import (
    Condensation,
    PenaltyCoupling,
    LagrangeCoupling,
    MortarCoupling
)
from Core.Objects.DFEM.Block import Block_2D
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.FEM.Triangles import Triangle3
from Core.Structures.Structure_Hybrid import Hybrid


@pytest.fixture
def hybrid_setup():
    """
    Hybrid setup: Block and FEM placed next to each other.
    Block: Square [-1, 0] x [-0.5, 0.5].
    Interface at x=0.
    Set Block Reference Point at (0, 0) (on the interface).
    FEM: Triangle [(0,0), (1,0), (0,1)].
    Node at (0,0) coincides with Block Reference Point.
    """
    st = Hybrid(merge_coincident_nodes=False)

    # Material
    mat = PlaneStress(E=200e9, nu=0.3, rho=7850.0)
    geom = Geometry2D(t=0.01)

    # Block on left
    # Vertices: (-1, -0.5), (0, -0.5), (0, 0.5), (-1, 0.5)
    block_verts = np.array([[-1.0, -0.5], [0.0, -0.5], [0.0, 0.5], [-1.0, 0.5]])
    # Reference point at (0,0) to coincide with FEM node
    block = Block_2D(vertices=block_verts, material=mat, ref_point=np.array([0.0, 0.0]))
    st.list_blocks.append(block)

    # FEM on right
    # Nodes: (0,0), (1,0), (0,1)
    tri = Triangle3(nodes=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], mat=mat, geom=geom)
    st.add_fe(tri)

    st.make_nodes()
    return st


@pytest.mark.unit
@pytest.mark.coupling
def test_detect_coupled_nodes(hybrid_setup):
    st = hybrid_setup
    # Tolerance must be enough to match (0,0)
    coupled = st.detect_coupled_fem_nodes(tolerance=1e-6)

    # Node 0: Block Ref Point (0,0)
    # Node 1: FEM Node (0,0)
    # Node 2: FEM Node (1,0)
    # Node 3: FEM Node (0,1)

    # We expect FEM Node 1 to be coupled to Block 0
    assert 1 in coupled
    assert coupled[1] == 0
    assert len(coupled) == 1


@pytest.mark.unit
@pytest.mark.coupling
def test_constraint_coupling_setup(hybrid_setup):
    st = hybrid_setup
    st.enable_block_fem_coupling(method='constraint')

    assert st.coupling_enabled
    assert isinstance(st.constraint_coupling, Condensation)
    assert st.coupling_T is not None
    # Block (3) + 3 FEM nodes (2*3=6) = 9 full DOFs
    # 1 coupled FEM node (2 DOFs removed) -> 7 reduced DOFs
    assert st.nb_dofs_full == 9
    assert st.nb_dofs_reduced == 7
    assert st.coupling_T.shape == (9, 7)


@pytest.mark.unit
@pytest.mark.coupling
def test_penalty_coupling_setup(hybrid_setup):
    st = hybrid_setup
    st.enable_block_fem_coupling(method='penalty', penalty=1e5)

    assert st.coupling_enabled
    assert isinstance(st.penalty_coupling, PenaltyCoupling)
    assert st.penalty_coupling.penalty_factor == 1e5
    assert st.penalty_coupling.active


@pytest.mark.unit
@pytest.mark.coupling
def test_lagrange_coupling_setup(hybrid_setup):
    st = hybrid_setup
    st.enable_block_fem_coupling(method='lagrange')

    assert st.coupling_enabled
    assert isinstance(st.lagrange_coupling, LagrangeCoupling)
    assert st.lagrange_coupling.active

    C = st.lagrange_coupling.constraint_matrix_C
    # 1 coupled node => 2 constraints
    # C shape: (2, 9)
    assert C.shape == (2, 9)


@pytest.mark.unit
@pytest.mark.coupling
@pytest.mark.mortar
def test_mortar_coupling_setup():
    """
    Mortar setup: Interface detection.
    Block: [-1, 0] x [0, 1]. Right face x=0.
    FEM: [0, 1] x [0, 1]. Left edge x=0.
    """
    st = Hybrid(merge_coincident_nodes=True)
    mat = PlaneStress(E=200e9, nu=0.3, rho=7850.0)
    geom = Geometry2D(t=0.01)

    # Block
    block_verts = np.array([[-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
    block = Block_2D(vertices=block_verts, material=mat)
    st.list_blocks.append(block)

    # FEM: Two triangles forming square [0,1]x[0,1]
    # Nodes: (0,0), (1,0), (1,1), (0,1)
    # Edge (0,0)-(0,1) is the interface with Block
    tri1 = Triangle3(nodes=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], mat=mat, geom=geom)
    tri2 = Triangle3(nodes=[(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], mat=mat, geom=geom)
    st.add_fe(tri1)
    st.add_fe(tri2)

    st.make_nodes()

    # Enable Mortar
    # interface_orientation=None to capture vertical interface
    st.enable_block_fem_coupling(
        method='mortar',
        interface_tolerance=0.1,
        interface_orientation=None
    )

    assert st.coupling_enabled
    assert isinstance(st.mortar_coupling, MortarCoupling)
    assert st.mortar_coupling.active
    assert st.mortar_coupling.constraint_matrix_C is not None
    # Check that constraints were generated
    assert st.mortar_coupling.constraint_matrix_C.shape[0] > 0
