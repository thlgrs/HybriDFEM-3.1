"""
Shared fixtures for HybriDFEM tests.

This module provides simple, reusable fixtures for testing.
"""
import numpy as np
import pytest

# Import Core components
from Core.Objects.ConstitutiveLaw.Material import Material, PlaneStress
from Core.Objects.DFEM.Block import Block_2D
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.FEM.Triangles import Triangle3, Triangle6


# =============================================================================
# ConstitutiveLaw Fixtures
# =============================================================================

@pytest.fixture
def steel():
    """Steel material: E=200 GPa, nu=0.3, rho=7850 kg/m3."""
    return PlaneStress(E=200e9, nu=0.3, rho=7850.0)


@pytest.fixture
def concrete():
    """Concrete material: E=30 GPa, nu=0.2, rho=2400 kg/m3."""
    return PlaneStress(E=30e9, nu=0.2, rho=2400.0)


@pytest.fixture
def generic_material():
    """Generic 1D material for beams/springs."""
    return Material(E=200e9, nu=0.3, rho=7850.0)


# =============================================================================
# Geometry Fixtures
# =============================================================================

@pytest.fixture
def thin_plate():
    """Thin plate geometry: t=10mm."""
    return Geometry2D(t=0.01)


@pytest.fixture
def thick_plate():
    """Thick plate geometry: t=100mm."""
    return Geometry2D(t=0.10)


# =============================================================================
# Element Fixtures
# =============================================================================

@pytest.fixture
def unit_triangle_nodes():
    """Standard unit right triangle: (0,0), (1,0), (0,1)."""
    return [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]


@pytest.fixture
def triangle3(unit_triangle_nodes, steel, thin_plate):
    """Pre-configured Triangle3 element."""
    return Triangle3(nodes=unit_triangle_nodes, mat=steel, geom=thin_plate)


@pytest.fixture
def triangle6_nodes():
    """6-node triangle: corners + mid-sides."""
    return [
        (0.0, 0.0),  # corner 1
        (1.0, 0.0),  # corner 2
        (0.0, 1.0),  # corner 3
        (0.5, 0.0),  # mid-side 1-2
        (0.5, 0.5),  # mid-side 2-3
        (0.0, 0.5),  # mid-side 3-1
    ]


@pytest.fixture
def triangle6(triangle6_nodes, steel, thin_plate):
    """Pre-configured Triangle6 element."""
    return Triangle6(nodes=triangle6_nodes, mat=steel, geom=thin_plate)


# =============================================================================
# Block Fixtures
# =============================================================================

@pytest.fixture
def square_vertices():
    """Unit square vertices (counter-clockwise)."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])


@pytest.fixture
def square_block(square_vertices, concrete):
    """Pre-configured square block."""
    return Block_2D(vertices=square_vertices, b=1.0, material=concrete)


@pytest.fixture
def rectangular_vertices():
    """2x1 rectangle vertices (counter-clockwise)."""
    return np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [0.0, 1.0]
    ])


# =============================================================================
# Helper Functions
# =============================================================================

def is_symmetric(matrix, tol=1e-10):
    """Check if matrix is symmetric."""
    return np.allclose(matrix, matrix.T, rtol=tol, atol=tol)


def is_positive_definite(matrix):
    """Check if matrix is positive definite."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues > 0)


def is_positive_semidefinite(matrix, tol=1e-10):
    """Check if matrix is positive semi-definite (allows zero eigenvalues)."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.all(eigenvalues >= -tol)
