# Extending HybriDFEM

This guide explains how to extend HybriDFEM with new elements, solvers, materials, and coupling methods.

## Adding a New FEM Element

### Step 1: Understand the Base Class

All 2D elements inherit from `Element2D` in `Core/Objects/FEM/Element2D.py`:

```python
class Element2D(BaseFE):
    """Base class for 2D plane elements."""

    def __init__(self, nodes, mat, geom):
        self.nodes = nodes      # List of (x, y) tuples
        self.mat = mat          # PlaneStress or PlaneStrain
        self.geom = geom        # Geometry2D (thickness)
        self.connect = []       # Global node IDs (set later)

    @abstractmethod
    def N(self, xi, eta):
        """Shape functions at natural coordinates."""
        pass

    @abstractmethod
    def dN(self, xi, eta):
        """Shape function derivatives."""
        pass

    def get_jacobian(self, xi, eta):
        """Compute Jacobian matrix."""
        ...

    def get_K_loc(self):
        """Compute element stiffness matrix."""
        ...
```

### Step 2: Create Your Element Class

Let's create a hypothetical `Triangle10` (cubic triangle with 10 nodes):

```python
# File: Core/Objects/FEM/Triangles.py (add to existing file)
# Or create a new file: Core/Objects/FEM/HigherOrder.py

import numpy as np
from Core.Objects.FEM.Element2D import Element2D


class Triangle10(Element2D):
    """10-node cubic triangular element.

    Nodes are arranged as:
         2
        / \
       9   8
      /     \
     5   6   7
    /         \
   0---3---4---1

    Natural coordinates: standard triangle (0,0), (1,0), (0,1)
    """

    NODES_PER_ELEMENT = 10
    DOF_PER_NODE = 2
    INTEGRATION_ORDER = 4  # Need 4th order for cubic

    def __init__(self, nodes, mat, geom):
        super().__init__(nodes, mat, geom)
        if len(nodes) != 10:
            raise ValueError(f"Triangle10 requires 10 nodes, got {len(nodes)}")

    def N(self, xi, eta):
        """Shape functions at (xi, eta).

        Returns:
            ndarray(10,): Shape function values
        """
        zeta = 1 - xi - eta  # Third barycentric coordinate

        N = np.zeros(10)

        # Corner nodes (cubic Lagrange)
        N[0] = 0.5 * zeta * (3*zeta - 1) * (3*zeta - 2)
        N[1] = 0.5 * xi * (3*xi - 1) * (3*xi - 2)
        N[2] = 0.5 * eta * (3*eta - 1) * (3*eta - 2)

        # Edge nodes (internal)
        N[3] = 4.5 * zeta * xi * (3*zeta - 1)
        N[4] = 4.5 * zeta * xi * (3*xi - 1)
        N[5] = 4.5 * zeta * eta * (3*zeta - 1)
        N[6] = 27 * zeta * xi * eta
        N[7] = 4.5 * xi * eta * (3*xi - 1)
        N[8] = 4.5 * xi * eta * (3*eta - 1)
        N[9] = 4.5 * zeta * eta * (3*eta - 1)

        return N

    def dN(self, xi, eta):
        """Shape function derivatives w.r.t. (xi, eta).

        Returns:
            ndarray(10, 2): [dN/dxi, dN/deta] for each node
        """
        zeta = 1 - xi - eta
        dN = np.zeros((10, 2))

        # Derivatives of corner nodes
        # dN[0]/dxi = d/dxi [0.5 * zeta * (3*zeta - 1) * (3*zeta - 2)]
        # ... (compute all derivatives)

        # This is a simplified placeholder - actual derivatives need full derivation
        raise NotImplementedError("Derivatives need implementation")

        return dN

    def get_integration_points(self):
        """Return Gauss points for this element.

        Returns:
            List of (xi, eta, weight) tuples
        """
        # Use 4th order Gauss quadrature for triangles
        # Example points (actual values from reference)
        return [
            (1/3, 1/3, -27/96),  # Center point
            (0.6, 0.2, 25/96),   # Near edges
            (0.2, 0.6, 25/96),
            (0.2, 0.2, 25/96),
        ]
```

### Step 3: Register the Element

Update `Core/Objects/FEM/__init__.py`:

```python
from .Triangles import Triangle3, Triangle6, Triangle10
from .Quads import Quad4, Quad8, Quad9

__all__ = [
    'Triangle3', 'Triangle6', 'Triangle10',  # Added Triangle10
    'Quad4', 'Quad8', 'Quad9',
]
```

### Step 4: Write Tests

Create `tests/test_triangle10.py`:

```python
import pytest
import numpy as np
from Core.Objects.FEM import Triangle10
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.ConstitutiveLaw import PlaneStress
from conftest import is_symmetric, is_positive_semidefinite


@pytest.fixture
def triangle10_nodes():
    """10-node cubic triangle nodes."""
    return [
        (0.0, 0.0),    # 0
        (1.0, 0.0),    # 1
        (0.0, 1.0),    # 2
        (1/3, 0.0),    # 3
        (2/3, 0.0),    # 4
        (0.0, 1/3),    # 5
        (1/3, 1/3),    # 6
        (2/3, 1/3),    # 7
        (1/3, 2/3),    # 8
        (0.0, 2/3),    # 9
    ]


@pytest.fixture
def triangle10(triangle10_nodes):
    mat = PlaneStress(E=200e9, nu=0.3, rho=7850)
    geom = Geometry2D(t=0.01)
    return Triangle10(nodes=triangle10_nodes, mat=mat, geom=geom)


class TestTriangle10:

    @pytest.mark.fem
    def test_partition_of_unity(self, triangle10):
        """Shape functions must sum to 1."""
        test_points = [(0, 0), (1, 0), (0, 1), (1/3, 1/3), (0.5, 0.25)]
        for xi, eta in test_points:
            N = triangle10.N(xi, eta)
            assert np.isclose(np.sum(N), 1.0), f"Sum={np.sum(N)} at ({xi},{eta})"

    @pytest.mark.fem
    def test_node_interpolation(self, triangle10, triangle10_nodes):
        """Shape functions must be 1 at their node, 0 at others."""
        natural_coords = [
            (0, 0), (1, 0), (0, 1),  # Corners
            (1/3, 0), (2/3, 0),      # Edge 0-1
            # ... add all 10 natural coords
        ]
        for i, (xi, eta) in enumerate(natural_coords[:3]):  # Test corners
            N = triangle10.N(xi, eta)
            assert np.isclose(N[i], 1.0), f"N[{i}] at node {i} = {N[i]}"

    @pytest.mark.fem
    def test_stiffness_symmetry(self, triangle10):
        """Stiffness matrix must be symmetric."""
        K = triangle10.get_K_loc()
        assert is_symmetric(K)

    @pytest.mark.fem
    def test_stiffness_positive_semidefinite(self, triangle10):
        """Stiffness must be positive semi-definite."""
        K = triangle10.get_K_loc()
        assert is_positive_semidefinite(K)
```

---

## Adding a New Solver

### Step 1: Understand the Solver Interface

Solvers in `Core/Solvers/` are typically static methods that take a structure and return it modified.

### Step 2: Create the Solver

Example: Adding a quasi-static solver:

```python
# File: Core/Solvers/QuasiStatic.py

import numpy as np
from scipy.sparse.linalg import spsolve
from Core.Solvers.Static import StaticBase, ConvergenceError


class QuasiStatic(StaticBase):
    """Quasi-static solver with load ramping.

    Incrementally applies load with equilibrium checks at each step.
    """

    @staticmethod
    def solve(St, n_steps=10, tol=1e-6, max_iter=20, verbose=True):
        """Solve with quasi-static loading.

        Args:
            St: Structure object
            n_steps: Number of load increments
            tol: Equilibrium tolerance
            max_iter: Max iterations per step
            verbose: Print progress

        Returns:
            Modified structure with solution
        """
        if verbose:
            print(f"Quasi-static analysis: {n_steps} steps")

        # Store total load
        P_total = St.P.copy()

        # Initialize
        St.U = np.zeros(St.nb_dofs)
        St.P = np.zeros(St.nb_dofs)

        # Load stepping
        for step in range(1, n_steps + 1):
            # Increment load
            load_factor = step / n_steps
            St.P = load_factor * P_total

            if verbose:
                print(f"  Step {step}/{n_steps}, LF={load_factor:.2f}")

            # Newton-Raphson iteration
            for iteration in range(max_iter):
                # Assemble stiffness
                K = St.get_K_str()

                # Compute residual
                P_r = St.get_P_r()
                residual = St.P - P_r

                # Check convergence
                res_norm = np.linalg.norm(residual[St.dof_free])
                if res_norm < tol:
                    if verbose:
                        print(f"    Converged in {iteration+1} iterations")
                    break

                # Solve for increment
                K_ff = K[np.ix_(St.dof_free, St.dof_free)]
                dU = np.zeros(St.nb_dofs)
                dU[St.dof_free] = spsolve(K_ff, residual[St.dof_free])

                # Update displacement
                St.U += dU

            else:
                raise ConvergenceError(f"Step {step} did not converge")

        return St
```

### Step 3: Register the Solver

Update `Core/Solvers/__init__.py`:

```python
from .Static import StaticLinear, StaticNonLinear, Static
from .Dynamic import Dynamic
from .Modal import Modal
from .QuasiStatic import QuasiStatic  # Add this

__all__ = [
    'StaticLinear', 'StaticNonLinear', 'Static',
    'Dynamic', 'Modal',
    'QuasiStatic',  # Add this
]
```

---

## Adding a New Material Model

### Step 1: Understand Material Interface

Materials provide constitutive matrices via `get_D()`.

### Step 2: Create the Material

Example: Orthotropic material:

```python
# File: Core/Objects/ConstitutiveLaw/Orthotropic.py

import numpy as np


class Orthotropic:
    """Orthotropic material for plane stress.

    Attributes:
        E1, E2: Young's moduli in principal directions
        nu12: Poisson's ratio
        G12: Shear modulus
        rho: Density
    """

    def __init__(self, E1, E2, nu12, G12, rho=0.0):
        self.E1 = E1
        self.E2 = E2
        self.nu12 = nu12
        self.nu21 = nu12 * E2 / E1  # From symmetry
        self.G12 = G12
        self.rho = rho

        # Validate
        if 1 - self.nu12 * self.nu21 <= 0:
            raise ValueError("Invalid Poisson's ratios: 1 - nu12*nu21 <= 0")

    def get_D(self):
        """Return 3x3 constitutive matrix for plane stress."""
        denom = 1 - self.nu12 * self.nu21

        D = np.array([
            [self.E1 / denom,        self.nu12 * self.E2 / denom, 0],
            [self.nu21 * self.E1 / denom, self.E2 / denom,        0],
            [0,                           0,                       self.G12]
        ])

        return D

    def __repr__(self):
        return (f"Orthotropic(E1={self.E1:.2e}, E2={self.E2:.2e}, "
                f"nu12={self.nu12}, G12={self.G12:.2e})")
```

### Step 3: Register the Material

Update `Core/Objects/ConstitutiveLaw/__init__.py`:

```python
from .Material import Material, PlaneStress, PlaneStrain
from .Orthotropic import Orthotropic  # Add this

__all__ = [
    'Material', 'PlaneStress', 'PlaneStrain',
    'Orthotropic',  # Add this
]
```

---

## Adding a New Coupling Method

### Step 1: Understand the Coupling Interface

All coupling methods inherit from `BaseCoupling`:

```python
class BaseCoupling(ABC):
    @abstractmethod
    def build_constraint_matrix(self, structure):
        """Build the global constraint matrix."""
        pass

    @abstractmethod
    def activate(self):
        """Activate the coupling."""
        pass
```

### Step 2: Create the Coupling Method

Example: Adding a weighted coupling method:

```python
# File: Core/Objects/Coupling/WeightedCoupling.py

import numpy as np
from Core.Objects.Coupling.BaseCoupling import BaseCoupling


class WeightedCoupling(BaseCoupling):
    """Weighted coupling with distance-based influence.

    Uses distance weighting to distribute coupling forces
    across multiple FEM nodes near block interfaces.
    """

    def __init__(self, influence_radius=0.1, weight_function='inverse'):
        self.influence_radius = influence_radius
        self.weight_function = weight_function
        self.active = False
        self.coupled_nodes = {}
        self.weights = {}

    def build_constraint_matrix(self, structure):
        """Build weighted constraint relationships."""
        self.structure = structure

        # Find FEM nodes within influence radius of each block
        for block_idx, block in enumerate(structure.list_blocks):
            block_pos = structure.list_nodes[block.connect]

            for fem_node_id in range(len(structure.list_blocks), len(structure.list_nodes)):
                fem_pos = structure.list_nodes[fem_node_id]
                dist = np.linalg.norm(fem_pos - block_pos)

                if dist <= self.influence_radius:
                    # Compute weight
                    if self.weight_function == 'inverse':
                        weight = 1.0 / (dist + 1e-10)
                    elif self.weight_function == 'linear':
                        weight = 1.0 - dist / self.influence_radius
                    else:
                        weight = 1.0

                    key = (fem_node_id, block_idx)
                    self.coupled_nodes[key] = {
                        'fem_pos': fem_pos,
                        'block_pos': block_pos,
                        'distance': dist
                    }
                    self.weights[key] = weight

        # Normalize weights per block
        for block_idx in range(len(structure.list_blocks)):
            total_weight = sum(
                self.weights[k] for k in self.weights
                if k[1] == block_idx
            )
            if total_weight > 0:
                for key in self.weights:
                    if key[1] == block_idx:
                        self.weights[key] /= total_weight

    def activate(self):
        """Activate the coupling."""
        self.active = True

    def get_coupling_forces(self, U):
        """Compute coupling forces based on displacement."""
        # Implementation details...
        pass
```

### Step 3: Integrate with Hybrid Structure

Modify `Core/Structures/Structure_Hybrid.py` to recognize the new method:

```python
def enable_block_fem_coupling(self, method='constraint', ...):
    # ... existing code ...

    elif method == 'weighted':
        from Core.Objects.Coupling import WeightedCoupling
        self.weighted_coupling = WeightedCoupling(
            influence_radius=kwargs.get('influence_radius', 0.1),
            weight_function=kwargs.get('weight_function', 'inverse')
        )
        self.weighted_coupling.build_constraint_matrix(self)
        self.weighted_coupling.activate()
        self.coupling_enabled = True
```

---

## Best Practices for Extensions

### 1. Follow Existing Patterns

Study existing implementations before adding new ones:
- Element patterns: `Core/Objects/FEM/Triangles.py`
- Coupling patterns: `Core/Objects/Coupling/Condensation.py`
- Solver patterns: `Core/Solvers/Static.py`

### 2. Write Tests First (TDD)

```python
# Write the test first
def test_my_new_feature():
    result = my_new_feature()
    assert result == expected

# Then implement to make test pass
```

### 3. Document Thoroughly

Every public class and method needs docstrings:

```python
def my_method(self, param: float) -> np.ndarray:
    """Short description.

    Longer description if needed.

    Args:
        param: Description of parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param is invalid.

    Example:
        >>> obj.my_method(1.0)
        array([...])
    """
```

### 4. Maintain Backward Compatibility

- Don't change existing method signatures
- Use default parameters for new functionality
- Deprecate old APIs gracefully

```python
def old_method(self):
    import warnings
    warnings.warn("old_method is deprecated, use new_method", DeprecationWarning)
    return self.new_method()
```

### 5. Run Full Test Suite

Before submitting:

```bash
pytest -v
pytest -m "not slow"  # Quick check
```

---

## Architecture Decision Records

When making significant changes, document your reasoning:

```markdown
# ADR-001: New Coupling Method

## Context
We need to support non-matching meshes with curved interfaces.

## Decision
Implement mortar coupling with segment-based integration.

## Consequences
- Pro: Works with any mesh configuration
- Pro: Optimal for curved interfaces
- Con: More complex implementation
- Con: Additional integration parameters
```

---

*Previous: [API Reference](07_api_reference.md) | Back to: [Documentation Index](../README.md)*
