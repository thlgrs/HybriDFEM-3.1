# Development Guide

This guide covers coding standards, testing practices, and best practices for developing HybriDFEM.

## Coding Standards

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Classes | PascalCase | `Triangle3`, `Block_2D`, `Structure_FEM` |
| Functions/Methods | snake_case | `get_stiffness`, `make_nodes`, `solve_linear` |
| Constants | UPPER_CASE | `CONDITION_NUMBER_THRESHOLD`, `MAX_ITERATIONS` |
| Private | Leading underscore | `_compute_area`, `_stiffness_fem` |
| Module-level | snake_case | `model_builders.py`, `mesh_generation.py` |

### Documentation Style

Use Google-style docstrings:

```python
def compute_stiffness(self, material: PlaneStress) -> np.ndarray:
    """Compute element stiffness matrix.

    Assembles the element stiffness matrix using numerical integration
    with Gauss quadrature.

    Args:
        material: Constitutive model for plane stress/strain.
            Must have E (Young's modulus) and nu (Poisson's ratio).

    Returns:
        Symmetric stiffness matrix of shape (ndof, ndof).

    Raises:
        ValueError: If material properties are invalid.
        RuntimeError: If Jacobian is singular (degenerate element).

    Example:
        >>> mat = PlaneStress(E=200e9, nu=0.3)
        >>> K = element.compute_stiffness(mat)
        >>> assert K.shape == (6, 6)  # Triangle3 has 6 DOFs
    """
```

### Type Hints

Use type hints throughout the codebase:

```python
from typing import List, Tuple, Union, Optional, Dict
import numpy as np

def get_node_id(
    self,
    node: Union[int, Tuple[float, float], np.ndarray],
    tol: float = 1e-8,
    optimized: bool = False
) -> Optional[int]:
    """Find global node index by coordinates."""
    ...

def create_mesh(
    nodes: List[Tuple[float, float]],
    connectivity: List[List[int]],
    material: Union[PlaneStress, PlaneStrain]
) -> Dict[str, np.ndarray]:
    ...
```

### Import Organization

Organize imports in this order:

```python
# 1. Standard library
import math
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional

# 2. Third-party packages
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# 3. Local imports (Core package)
from Core.Structures.Structure_2D import Structure_2D
from Core.Objects.FEM import Triangle3, Triangle6
from Core.Objects.ConstitutiveLaw import PlaneStress
```

### Code Style

Follow PEP 8 with these specifics:

```python
# Line length: 100 characters max (prefer 80-90)

# Indentation: 4 spaces (no tabs)

# Blank lines:
# - 2 blank lines between top-level definitions
# - 1 blank line between methods in a class
# - Group related code blocks with blank lines

class MyClass:
    """Class docstring."""

    def __init__(self):
        self.value = 0

    def method_one(self):
        """First method."""
        pass

    def method_two(self):
        """Second method."""
        pass


def top_level_function():
    """Top-level function."""
    pass
```

### Matrix Operations

Use appropriate libraries:

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, eigsh

# Dense operations (small matrices)
K = np.zeros((n, n))
K_inv = np.linalg.inv(K)
eig_vals = np.linalg.eigvalsh(K)

# Sparse operations (large matrices)
K_sparse = sp.csr_matrix(K)
u = spsolve(K_sparse, P)
eig_vals, eig_vecs = eigsh(K_sparse, k=10, M=M_sparse, which='SM')

# Natural coordinates convention
xi, eta = 0.5, 0.5  # or use Greek: ξ, η in comments
```

## Testing

### Test Structure

Tests are in the `tests/` directory:

```
tests/
├── conftest.py           # Shared fixtures
├── test_elements.py      # FEM element tests
├── test_quads.py         # Quadrilateral element tests
├── test_blocks.py        # Block tests
├── test_materials.py     # Material model tests
├── test_coupling.py      # Coupling method tests
├── test_solvers.py       # Linear solver tests
├── test_solvers_nonlinear.py  # Nonlinear solver tests
├── test_structure_fem.py # Structure_FEM tests
└── test_structure_hybrid.py   # Hybrid structure tests
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific file
pytest tests/test_elements.py

# Run tests matching pattern
pytest -k "triangle"

# Run by marker
pytest -m fem           # FEM tests
pytest -m dfem          # DFEM tests
pytest -m hybrid        # Hybrid tests
pytest -m solver        # Solver tests
pytest -m "not slow"    # Skip slow tests

# Run with coverage (if pytest-cov installed)
pytest --cov=Core --cov-report=html
```

### Test Markers

Available markers (defined in `pytest.ini`):

```python
@pytest.mark.unit        # Individual component tests
@pytest.mark.integration # Combined system tests
@pytest.mark.slow        # Long-running tests
@pytest.mark.fem         # FEM element tests
@pytest.mark.dfem        # DFEM block tests
@pytest.mark.hybrid      # Hybrid coupling tests
@pytest.mark.solver      # Solver tests
@pytest.mark.material    # Material model tests
@pytest.mark.contact     # Contact mechanics tests
@pytest.mark.coupling    # Coupling method tests
@pytest.mark.mortar      # Mortar method tests
@pytest.mark.lagrange    # Lagrange multiplier tests
```

### Writing Tests

Use pytest fixtures from `conftest.py`:

```python
import pytest
import numpy as np
from conftest import is_symmetric, is_positive_semidefinite


class TestTriangle3:
    """Tests for Triangle3 element."""

    @pytest.mark.fem
    def test_stiffness_matrix_symmetry(self, triangle3):
        """Stiffness matrix must be symmetric."""
        K = triangle3.get_K_loc()
        assert is_symmetric(K), "Stiffness matrix not symmetric"

    @pytest.mark.fem
    def test_stiffness_matrix_positive_semidefinite(self, triangle3):
        """Stiffness matrix must be positive semi-definite."""
        K = triangle3.get_K_loc()
        assert is_positive_semidefinite(K), "K not positive semi-definite"

    @pytest.mark.fem
    def test_shape_functions_partition_of_unity(self, triangle3):
        """Shape functions must sum to 1 at any point."""
        # Test at several points
        test_points = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (1/3, 1/3)]
        for xi, eta in test_points:
            N = triangle3.N(xi, eta)
            assert np.isclose(np.sum(N), 1.0), f"Sum of N = {np.sum(N)} at ({xi}, {eta})"
```

### Available Fixtures

From `tests/conftest.py`:

```python
# Materials
@pytest.fixture
def steel():
    """Steel: E=200 GPa, nu=0.3, rho=7850 kg/m3."""
    return PlaneStress(E=200e9, nu=0.3, rho=7850.0)

@pytest.fixture
def concrete():
    """Concrete: E=30 GPa, nu=0.2, rho=2400 kg/m3."""
    return PlaneStress(E=30e9, nu=0.2, rho=2400.0)

# Geometries
@pytest.fixture
def thin_plate():
    """t=10mm."""
    return Geometry2D(t=0.01)

@pytest.fixture
def thick_plate():
    """t=100mm."""
    return Geometry2D(t=0.10)

# Elements
@pytest.fixture
def triangle3(unit_triangle_nodes, steel, thin_plate):
    """Pre-configured Triangle3."""
    return Triangle3(nodes=unit_triangle_nodes, mat=steel, geom=thin_plate)

# Blocks
@pytest.fixture
def square_block(square_vertices, concrete):
    """Pre-configured 1x1 square block."""
    return Block_2D(vertices=square_vertices, b=1.0, material=concrete)
```

### Helper Functions

```python
from conftest import is_symmetric, is_positive_definite, is_positive_semidefinite

# Check matrix properties
assert is_symmetric(K, tol=1e-10)
assert is_positive_definite(K)
assert is_positive_semidefinite(M, tol=1e-10)
```

## Debugging

### Common Debugging Techniques

#### 1. Check Matrix Properties

```python
def debug_stiffness(K):
    """Debug stiffness matrix issues."""
    print(f"Shape: {K.shape}")
    print(f"Symmetric: {np.allclose(K, K.T)}")
    print(f"Min eigenvalue: {np.min(np.linalg.eigvalsh(K)):.2e}")
    print(f"Condition number: {np.linalg.cond(K):.2e}")
    print(f"Rank: {np.linalg.matrix_rank(K)} (expected: {K.shape[0]})")
```

#### 2. Visualize Structure

```python
# Quick plot to check geometry
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
St.plot(ax=ax, show_nodes=True, show_elements=True)
ax.set_title("Debug: Structure geometry")
plt.show()
```

#### 3. Print DOF Information

```python
def debug_dofs(St):
    """Print DOF mapping for debugging."""
    print(f"Total DOFs: {St.nb_dofs}")
    print(f"Fixed DOFs: {St.dof_fix}")
    print(f"Free DOFs: {St.dof_free}")
    print(f"\nNode DOF offsets: {St.node_dof_offsets}")
    print(f"Node DOF counts: {St.node_dof_counts}")
```

#### 4. Check Coupling

```python
def debug_coupling(St):
    """Debug hybrid coupling."""
    print(f"Coupling enabled: {St.coupling_enabled}")
    print(f"Coupled nodes: {St.coupled_fem_nodes}")
    if hasattr(St, 'coupling_T') and St.coupling_T is not None:
        T = St.coupling_T
        print(f"Transformation matrix shape: {T.shape}")
        print(f"nb_dofs_full: {St.nb_dofs_full}")
        print(f"nb_dofs_reduced: {St.nb_dofs_reduced}")
```

### Logging

For complex debugging, use Python's logging:

```python
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In your code
logger.debug(f"Assembling element {i}, K shape: {K_elem.shape}")
logger.info(f"Solved in {iterations} iterations")
logger.warning(f"Condition number {cond:.2e} is high")
```

## Performance Tips

### 1. Use NumPy Vectorization

```python
# SLOW: Python loops
result = []
for i in range(len(nodes)):
    result.append(nodes[i] + offset)

# FAST: NumPy vectorization
result = nodes + offset
```

### 2. Pre-allocate Arrays

```python
# SLOW: Growing arrays
K = np.array([])
for elem in elements:
    K = np.append(K, elem.get_K())

# FAST: Pre-allocate
n = sum(elem.ndof for elem in elements)
K = np.zeros((n, n))
# ... assemble in place
```

### 3. Use Sparse Matrices for Large Problems

```python
import scipy.sparse as sp

# For large systems
K_sparse = sp.lil_matrix((n, n))  # For assembly
# ... assemble ...
K_sparse = K_sparse.tocsr()  # Convert for solving
u = sp.linalg.spsolve(K_sparse, P)
```

### 4. Profile Before Optimizing

```python
import cProfile
import pstats

# Profile your code
cProfile.run('Static.solve_linear(St)', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## Error Handling

### Custom Exceptions

HybriDFEM defines custom exceptions in `Core/Solvers/Static.py`:

```python
from Core import ConvergenceError, SingularSystemError

try:
    St = StaticNonLinear.solve_forcecontrol(St, max_iter=10)
except ConvergenceError as e:
    print(f"Solver did not converge: {e}")
except SingularSystemError as e:
    print(f"System is singular: {e}")
```

### Validation Pattern

```python
def set_parameter(self, value: float) -> None:
    """Set parameter with validation."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric, got {type(value)}")
    if value <= 0:
        raise ValueError(f"Parameter must be positive, got {value}")
    self._parameter = float(value)
```

## File Organization for New Features

When adding a new feature, follow this structure:

```
Core/
└── Objects/
    └── NewFeature/
        ├── __init__.py       # Exports
        ├── base.py           # Abstract base class
        ├── implementation.py # Concrete implementation
        └── helpers.py        # Utility functions

tests/
└── test_new_feature.py       # Tests

Examples/
└── NewFeature/
    └── example_script.py     # Example usage
```

Update `Core/__init__.py` to export new public classes.

---

*Previous: [Core Concepts](03_core_concepts.md) | Next: [Git Workflow](05_git_workflow.md)*
