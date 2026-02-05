# API Reference

This document provides a quick reference for the main classes and functions in HybriDFEM.

## Structures

### Structure_FEM

Pure continuous FEM structure with 2 DOF per node.

```python
from Core import Structure_FEM

St = Structure_FEM(fixed_dofs_per_node=False)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `list_fes` | List[BaseFE] | Finite elements |
| `list_nodes` | List[ndarray] | Node coordinates |
| `nb_dofs` | int | Total DOFs |
| `U` | ndarray | Displacement vector |
| `P` | ndarray | External force vector |

**Methods:**

| Method | Description |
|--------|-------------|
| `add_fe(element)` | Add finite element |
| `make_nodes()` | Build node system |
| `fix_node(node_ids, dofs)` | Fix DOFs |
| `load_node(node_ids, dofs, force)` | Apply force |
| `get_K_str()` | Assemble stiffness matrix |
| `get_M_str()` | Assemble mass matrix |

---

### Structure_Block

Pure discrete block structure with 3 DOF per node.

```python
from Core import Structure_Block

St = Structure_Block()
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `list_blocks` | List[Block_2D] | Rigid blocks |
| `list_cfs` | List[CF_2D] | Contact faces |
| `list_nodes` | List[ndarray] | Node coordinates |

**Methods:**

| Method | Description |
|--------|-------------|
| `add_block(block)` | Add block |
| `add_block_from_vertices(vertices, b, material, ref_point)` | Create and add block |
| `detect_interfaces(eps, margin)` | Detect block interfaces |
| `make_cfs(lin_geom, nb_cps, contact)` | Create contact faces |
| `make_nodes()` | Build node system |

---

### Hybrid

Combined block-FEM structure with coupling.

```python
from Core import Hybrid

St = Hybrid(fixed_dofs_per_node=False, merge_coincident_nodes=True)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `list_blocks` | List[Block_2D] | Rigid blocks |
| `list_fes` | List[BaseFE] | Finite elements |
| `coupling_enabled` | bool | Coupling active |
| `coupled_fem_nodes` | Dict[int, int] | FEM node -> block mapping |

**Methods:**

| Method | Description |
|--------|-------------|
| `add_block(block)` | Add block |
| `add_fe(element)` | Add FEM element |
| `make_nodes()` | Build node system |
| `enable_block_fem_coupling(...)` | Enable coupling |
| `expand_displacement(u_reduced)` | Expand to full DOFs |

**`enable_block_fem_coupling` parameters:**

```python
St.enable_block_fem_coupling(
    tolerance=1e-9,           # Node matching tolerance
    method='constraint',      # 'constraint', 'penalty', 'lagrange', 'mortar'
    penalty='auto',           # Penalty value (for method='penalty')
    integration_order=2,      # Gauss order (for method='mortar')
    interface_tolerance=1e-4, # Interface detection (for method='mortar')
    interface_orientation='horizontal'  # 'horizontal', 'vertical', None
)
```

---

### Block Generators

Pre-built structure generators.

```python
from Core import BeamBlock, ArchBlock, WallBlock, TaperedBeamBlock, VoronoiBlock
```

| Generator | Description | Key Parameters |
|-----------|-------------|----------------|
| `BeamBlock` | Straight beam | `length`, `height`, `n_blocks` |
| `TaperedBeamBlock` | Tapered beam | `length`, `height_start`, `height_end`, `n_blocks` |
| `ArchBlock` | Circular arch | `radius`, `angle`, `n_blocks` |
| `WallBlock` | Rectangular wall | `width`, `height`, `nx`, `ny` |
| `VoronoiBlock` | Voronoi tessellation | `width`, `height`, `n_seeds` |

---

## FEM Elements

### Triangle3

3-node linear triangle (Constant Strain Triangle).

```python
from Core.Objects.FEM import Triangle3

elem = Triangle3(
    nodes=[(0,0), (1,0), (0,1)],  # Counter-clockwise
    mat=PlaneStress(E=200e9, nu=0.3),
    geom=Geometry2D(t=0.01)
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `N(xi, eta)` | ndarray(3,) | Shape functions |
| `dN(xi, eta)` | ndarray(3,2) | Shape function derivatives |
| `get_K_loc()` | ndarray(6,6) | Element stiffness |
| `get_M_loc()` | ndarray(6,6) | Element mass |

---

### Triangle6

6-node quadratic triangle (Linear Strain Triangle).

```python
from Core.Objects.FEM import Triangle6

elem = Triangle6(
    nodes=[...],  # 6 nodes: 3 corners + 3 mid-sides
    mat=...,
    geom=...
)
```

---

### Quad4

4-node bilinear quadrilateral.

```python
from Core.Objects.FEM import Quad4

elem = Quad4(
    nodes=[(0,0), (1,0), (1,1), (0,1)],  # Counter-clockwise
    mat=...,
    geom=...
)
```

---

### Quad8

8-node serendipity quadrilateral.

```python
from Core.Objects.FEM import Quad8

elem = Quad8(
    nodes=[...],  # 4 corners + 4 mid-sides
    mat=...,
    geom=...
)
```

---

### Quad9

9-node Lagrangian quadrilateral.

```python
from Core.Objects.FEM import Quad9

elem = Quad9(
    nodes=[...],  # 4 corners + 4 mid-sides + 1 center
    mat=...,
    geom=...
)
```

---

### Geometry2D

Element geometry definition.

```python
from Core.Objects.FEM.Element2D import Geometry2D

geom = Geometry2D(t=0.01)  # Thickness in meters
```

---

## DFEM Components

### Block_2D

Rigid block with 3 DOF.

```python
from Core.Objects.DFEM import Block_2D

block = Block_2D(
    vertices=np.array([[0,0], [1,0], [1,1], [0,1]]),
    b=0.1,           # Thickness
    material=material,
    ref_point=None   # Auto-compute centroid, or specify
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `vertices` | ndarray | Corner coordinates |
| `ref_point` | ndarray | Reference point |
| `area` | float | Block area |
| `mass` | float | Block mass |
| `inertia` | float | Moment of inertia |
| `connect` | int | Global node ID |
| `dofs` | ndarray | DOF indices |

**Methods:**

| Method | Description |
|--------|-------------|
| `constraint_matrix_for_node(pos)` | Get constraint matrix for position |
| `get_K_loc()` | Element stiffness (from contact) |
| `get_M_loc()` | Mass matrix |

---

### Contact (CF_2D)

Contact face between blocks.

```python
from Core.Objects.DFEM import CF_2D
```

Created automatically by `Structure_Block.make_cfs()`.

---

## Materials

### PlaneStress

For thin plates (sigma_z = 0).

```python
from Core.Objects.ConstitutiveLaw import PlaneStress

mat = PlaneStress(
    E=200e9,     # Young's modulus [Pa]
    nu=0.3,      # Poisson's ratio [-]
    rho=7850     # Density [kg/m3]
)
```

---

### PlaneStrain

For thick structures (epsilon_z = 0).

```python
from Core.Objects.ConstitutiveLaw import PlaneStrain

mat = PlaneStrain(E=30e9, nu=0.2, rho=2400)
```

---

### Material

Generic material for 1D elements.

```python
from Core.Objects.ConstitutiveLaw import Material

mat = Material(E=200e9, nu=0.3, rho=7850)
```

---

### Contact

Contact constitutive law.

```python
from Core.Objects.ConstitutiveLaw import Contact

contact = Contact(
    k_n=1e10,    # Normal stiffness [N/m]
    k_s=1e9,     # Shear stiffness [N/m]
    mu=0.6,      # Friction coefficient
    c=0          # Cohesion [Pa]
)
```

---

## Solvers

### StaticLinear (Static)

Linear static solver.

```python
from Core import Static, StaticLinear

# Solve linear system
St = Static.solve_linear(St)

# For Lagrange/Mortar coupling
St = Static.solve_linear_saddle_point(St)
```

---

### StaticNonLinear

Nonlinear static solver.

```python
from Core import StaticNonLinear

# Force control
St = StaticNonLinear.solve_forcecontrol(
    St,
    load_steps=10,
    max_iter=50,
    tol=1e-6
)

# Displacement control
St = StaticNonLinear.solve_dispcontrol(
    St,
    control_dof=dof_id,
    target_disp=0.01,
    steps=20
)
```

---

### Dynamic

Dynamic time-history solver.

```python
from Core import Dynamic

# Central Difference Method
St = Dynamic.CDM(
    St,
    dt=0.001,           # Time step [s]
    t_end=1.0,          # End time [s]
    output_interval=10  # Save every N steps
)
```

---

### Modal

Eigenvalue analysis.

```python
from Core import Modal

St = Modal.solve_modal(
    St,
    modes=10,        # Number of modes
    normalize=True   # Mass-normalize
)

# Results
frequencies = St.frequencies  # [Hz]
mode_shapes = St.mode_shapes
```

---

## Visualization

### Visualizer

Post-processing visualization.

```python
from Core import Visualizer, PlotStyle

viz = Visualizer(St)

# Deformed shape
viz.plot_deformed_shape(scale=100, title="Deformed")

# Stress contour
viz.plot_stress_contour(component='von_mises', scale=50)
```

**PlotStyle options:**

```python
style = PlotStyle()
style.node_size = 50
style.fem_linewidth = 1.0
style.block_linewidth = 1.5
style.use_latex = False
```

---

## Coupling Classes

### Condensation

DOF elimination coupling.

```python
from Core.Objects.Coupling import Condensation

coupling = Condensation(verbose=True)
```

---

### PenaltyCoupling

Penalty spring coupling.

```python
from Core.Objects.Coupling import PenaltyCoupling

coupling = PenaltyCoupling(
    penalty_factor=1000.0,
    auto_scale=True
)
```

---

### LagrangeCoupling

Lagrange multiplier coupling.

```python
from Core.Objects.Coupling import LagrangeCoupling

coupling = LagrangeCoupling(verbose=True)
```

---

### MortarCoupling

Mortar method coupling.

```python
from Core.Objects.Coupling import MortarCoupling

coupling = MortarCoupling(
    integration_order=2,
    interface_tolerance=1e-4,
    interface_orientation='horizontal'
)
```

---

## Exceptions

```python
from Core import ConvergenceError, SingularSystemError

try:
    St = StaticNonLinear.solve_forcecontrol(St, max_iter=10)
except ConvergenceError as e:
    print(f"Did not converge: {e}")
except SingularSystemError as e:
    print(f"Singular system: {e}")
```

---

## Common Imports

```python
# Main API
from Core import (
    # Structures
    Structure_2D,
    Structure_FEM,
    Structure_Block,
    Hybrid,

    # Block generators
    BeamBlock,
    ArchBlock,
    WallBlock,

    # Solvers
    Static,
    StaticLinear,
    StaticNonLinear,
    Dynamic,
    Modal,

    # Visualization
    Visualizer,
    PlotStyle,

    # Exceptions
    ConvergenceError,
    SingularSystemError,
)

# Elements
from Core.Objects.FEM import (
    Triangle3,
    Triangle6,
    Quad4,
    Quad8,
    Quad9,
)
from Core.Objects.FEM.Element2D import Geometry2D

# DFEM
from Core.Objects.DFEM import Block_2D

# Materials
from Core.Objects.ConstitutiveLaw import (
    PlaneStress,
    PlaneStrain,
    Material,
    Contact,
)

# Coupling
from Core.Objects.Coupling import (
    Condensation,
    PenaltyCoupling,
    LagrangeCoupling,
    MortarCoupling,
)
```

---

*Previous: [Examples Guide](06_examples_guide.md) | Next: [Extending HybriDFEM](08_extending.md)*
