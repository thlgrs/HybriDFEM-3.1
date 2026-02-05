# Architecture Overview

This document explains the HybriDFEM codebase organization, class hierarchy, and design patterns.

## Directory Structure

```
HybriDFEM/
├── Core/                          # Main framework library
│   ├── __init__.py                # Public API exports
│   ├── Objects/                   # Building blocks
│   │   ├── FEM/                   # Continuous finite elements
│   │   │   ├── BaseFE.py          # Abstract base for all FE
│   │   │   ├── Element2D.py       # 2D plane element base
│   │   │   ├── Triangles.py       # Triangle3, Triangle6
│   │   │   ├── Quads.py           # Quad4, Quad8, Quad9
│   │   │   ├── Timoshenko.py      # Beam element
│   │   │   └── Mesh.py            # Mesh utilities
│   │   ├── DFEM/                  # Discrete rigid blocks
│   │   │   ├── Block.py           # Block_2D class
│   │   │   ├── ContactFace.py     # CF_2D contact faces
│   │   │   ├── ContactPair.py     # CP_2D contact pairs
│   │   │   └── Surface.py         # Surface definitions
│   │   ├── Coupling/              # Block-FEM coupling methods
│   │   │   ├── BaseCoupling.py    # Abstract base class
│   │   │   ├── Condensation.py    # DOF elimination
│   │   │   ├── Penalty.py         # Virtual spring method
│   │   │   ├── LagrangeNodal.py   # Lagrange multipliers
│   │   │   ├── Mortar.py          # Mortar method
│   │   │   └── ...                # Supporting classes
│   │   └── ConstitutiveLaw/       # Material models
│   │       ├── Material.py        # PlaneStress, PlaneStrain
│   │       ├── Contact.py         # Contact constitutive law
│   │       └── Spring.py          # Spring elements
│   ├── Solvers/                   # Analysis algorithms
│   │   ├── Solver.py              # Base solver class
│   │   ├── Static.py              # Linear/nonlinear static
│   │   ├── Dynamic.py             # Time-history (CDM)
│   │   ├── Modal.py               # Eigenvalue analysis
│   │   ├── Visualizer.py          # Post-processing
│   │   └── Plotter.py             # Simple plots
│   └── Structures/                # High-level structure classes
│       ├── Structure_2D.py        # Abstract base class
│       ├── Structure_FEM.py       # Pure FEM (2 DOF/node)
│       ├── Structure_Block.py     # Pure blocks (3 DOF/node)
│       └── Structure_Hybrid.py    # Combined with coupling
├── Examples/                      # Demonstration scripts
│   ├── Structure_FEM/             # Pure FEM examples
│   ├── Structure_Block/           # Pure block examples
│   ├── Structure_Hybrid/          # Hybrid coupling examples
│   ├── Refactored_Legacy/         # Migrated legacy examples
│   ├── convergence/               # Convergence studies
│   └── utils/                     # Shared utilities
│       ├── model_builders.py      # High-level builders
│       ├── mesh_generation.py     # Mesh utilities
│       ├── solvers.py             # Solver runners
│       ├── visualization.py       # Visualization helpers
│       └── ...
├── GUI/                           # PyQt6 graphical interface
├── tests/                         # Test suite
└── Legacy/                        # Archived code (DO NOT MODIFY)
```

## Class Hierarchy

### Structure Classes

The structure classes form the core of HybriDFEM:

```
Structure_2D (ABC)
├── Structure_FEM       # Pure continuous FEM
├── Structure_Block     # Pure discrete blocks
└── Hybrid             # Combined (inherits from both)
    └── Hybrid(Structure_Block, Structure_FEM)
```

#### Structure_2D (Base Class)

Location: `Core/Structures/Structure_2D.py`

The abstract base class providing:
- Node management (`list_nodes`, `get_node_id`, `_add_node_if_new`)
- DOF management (`node_dof_counts`, `node_dof_offsets`, `nb_dofs`)
- Boundary conditions (`fix_node`, `load_node`)
- Solution vectors (`U`, `P`, `P_fixed`, `P_r`)
- Matrix storage (`K`, `M`, `K0`, `K_LG`)

```python
class Structure_2D(ABC):
    DOF_PER_NODE = 3  # Default: [ux, uy, rz]

    # Abstract methods that subclasses must implement
    @abstractmethod
    def make_nodes(self): pass

    @abstractmethod
    def get_K_str(self): pass  # Stiffness matrix

    @abstractmethod
    def get_M_str(self): pass  # Mass matrix

    @abstractmethod
    def get_P_r(self): pass    # Internal force vector
```

#### Structure_FEM

Location: `Core/Structures/Structure_FEM.py`

For pure continuous FEM analyses:
- 2 DOF per node (`[ux, uy]`)
- Stores elements in `list_fes`
- Methods: `add_fe()`, `_stiffness_fem()`, `_mass_fem()`

#### Structure_Block

Location: `Core/Structures/Structure_Block.py`

For pure discrete block analyses:
- 3 DOF per node (`[ux, uy, rz]`)
- Stores blocks in `list_blocks`, contact faces in `list_cfs`
- Methods: `add_block()`, `detect_interfaces()`, `make_cfs()`
- Block generators: `BeamBlock`, `ArchBlock`, `WallBlock`, `VoronoiBlock`

#### Hybrid

Location: `Core/Structures/Structure_Hybrid.py`

For combined block-FEM analyses with coupling:
- Multiple inheritance from both `Structure_Block` and `Structure_FEM`
- Variable DOF per node (2 for FEM nodes, 3 for block nodes)
- Coupling methods: `constraint`, `penalty`, `lagrange`, `mortar`

```python
class Hybrid(Structure_Block, Structure_FEM):
    def enable_block_fem_coupling(
        self,
        tolerance: float = 1e-9,
        method: str = 'constraint',  # or 'penalty', 'lagrange', 'mortar'
        penalty=None,
        integration_order: int = 2,
        interface_tolerance: float = 1e-4,
        interface_orientation: str = 'horizontal'
    ):
        ...
```

### FEM Elements

Location: `Core/Objects/FEM/`

```
BaseFE (ABC)
└── Element2D
    ├── Triangle3    # 3-node linear (CST)
    ├── Triangle6    # 6-node quadratic (LST)
    ├── Quad4        # 4-node bilinear
    ├── Quad8        # 8-node serendipity
    └── Quad9        # 9-node Lagrangian
```

Each element provides:
- Shape functions: `N(xi, eta)`, `dN(xi, eta)`
- Jacobian: `get_jacobian(xi, eta)`
- Stiffness: `get_K_loc()` -> local element stiffness
- Mass: `get_M_loc()` -> local element mass

### DFEM Components

Location: `Core/Objects/DFEM/`

```
Block_2D          # Rigid block with 3 DOF
├── vertices      # Corner coordinates
├── ref_point     # Reference point (centroid or edge midpoint)
├── connect       # Global node ID
└── dofs          # DOF indices [ux, uy, rz]

ContactFace (CF_2D)    # Interface between blocks
├── master_block       # Block on one side
├── slave_block        # Block on other side
└── contact_pairs      # CP_2D instances

ContactPair (CP_2D)    # Point contact
├── position           # Contact point location
├── normal             # Contact normal direction
└── constitutive       # Contact law (friction, cohesion)
```

### Coupling Methods

Location: `Core/Objects/Coupling/`

| Class | Method | Description |
|-------|--------|-------------|
| `Condensation` | constraint | DOF elimination via transformation matrix |
| `PenaltyCoupling` | penalty | Virtual springs at coupled nodes |
| `LagrangeCoupling` | lagrange | Lagrange multiplier enforcement |
| `MortarCoupling` | mortar | Distributed multipliers for non-matching meshes |

All inherit from `BaseCoupling`:

```python
class BaseCoupling(ABC):
    @abstractmethod
    def build_constraint_matrix(self, structure): pass

    @abstractmethod
    def activate(self): pass
```

### Solvers

Location: `Core/Solvers/`

```
Solver (Base)
├── StaticLinear     # K * u = P
├── StaticNonLinear  # Newton-Raphson iteration
├── Dynamic          # Central Difference Method
└── Modal            # Eigenvalue analysis
```

Usage pattern:

```python
from Core import Static, StaticNonLinear, Dynamic, Modal

# Linear static
St = Static.solve_linear(St)

# Nonlinear static (force control)
St = StaticNonLinear.solve_forcecontrol(St, load_steps=10)

# Nonlinear static (displacement control)
St = StaticNonLinear.solve_dispcontrol(St, control_dof=0, target_disp=0.01)

# Dynamic (CDM)
St = Dynamic.CDM(St, dt=0.001, t_end=1.0)

# Modal analysis
St = Modal.solve_modal(St, modes=10)
```

## Data Flow

### Typical Analysis Workflow

```
1. Create Structure
   └── Structure = Hybrid() / Structure_FEM() / Structure_Block()

2. Add Elements/Blocks
   └── St.add_fe(element) / St.add_block(block)

3. Build Node System
   └── St.make_nodes()
       ├── Creates global node list
       ├── Assigns DOF indices
       └── Initializes solution vectors

4. Apply Boundary Conditions
   └── St.fix_node(node_id, dofs)
       St.load_node(node_id, dofs, force)

5. Enable Coupling (Hybrid only)
   └── St.enable_block_fem_coupling(method='constraint')

6. Solve
   └── Static.solve_linear(St)
       ├── Assembles K = St.get_K_str()
       ├── Partitions by BC: K_ff, K_fs
       └── Solves: u_f = K_ff^-1 * P_f

7. Post-process
   └── Visualizer(St).plot_deformed_shape()
```

### Matrix Assembly

The stiffness matrix assembly follows this pattern:

```python
def get_K_str(self):
    # Initialize global matrix
    self.K = np.zeros((self.nb_dofs, self.nb_dofs))

    # Add block contributions
    self._stiffness_block()  # Contact stiffness from CF

    # Add FEM contributions
    self._stiffness_fem()    # Element stiffness

    # Add coupling (if enabled)
    if self.coupling_enabled:
        # Transform or augment based on method
        ...

    return self.K
```

## Design Patterns

### 1. Template Method Pattern

Base classes define the algorithm skeleton, subclasses implement specifics:

```python
# In Structure_2D (template)
def solve(self):
    K = self.get_K_str()  # Abstract - implemented by subclass
    M = self.get_M_str()  # Abstract
    # ... solve logic
```

### 2. Strategy Pattern

Coupling methods are interchangeable strategies:

```python
# Method selection at runtime
St.enable_block_fem_coupling(method='constraint')  # or 'penalty', 'lagrange', 'mortar'
```

### 3. Composite Pattern

Structures aggregate elements/blocks:

```python
St.list_fes = [elem1, elem2, ...]    # FEM elements
St.list_blocks = [block1, block2, ...] # Rigid blocks
```

### 4. Factory Pattern

Block generators create pre-configured structures:

```python
from Core import BeamBlock, ArchBlock, WallBlock

beam = BeamBlock(length=10, height=1, n_blocks=20)
arch = ArchBlock(radius=5, angle=180, n_blocks=30)
wall = WallBlock(width=5, height=10, nx=10, ny=20)
```

## Key Files to Study

When learning the codebase, study these files in order:

| Priority | File | What You'll Learn |
|----------|------|-------------------|
| 1 | `Core/__init__.py` | Public API, what's exposed |
| 2 | `Core/Structures/Structure_2D.py` | Base class, DOF management |
| 3 | `Core/Structures/Structure_FEM.py` | FEM assembly pattern |
| 4 | `Core/Objects/FEM/Triangles.py` | Element implementation |
| 5 | `Core/Structures/Structure_Hybrid.py` | Coupling integration |
| 6 | `Core/Objects/Coupling/Condensation.py` | Constraint coupling |
| 7 | `Core/Solvers/Static.py` | Solver algorithms |
| 8 | `tests/conftest.py` | Test fixtures, examples |

## Import Conventions

### Preferred Imports

```python
# Main API
from Core import Hybrid, Static, Visualizer

# Elements
from Core.Objects.FEM import Triangle3, Triangle6, Quad4, Quad8

# Materials
from Core.Objects.ConstitutiveLaw import PlaneStress, PlaneStrain, Material

# Coupling (when needed)
from Core.Objects.Coupling import Condensation, PenaltyCoupling
```

### Avoid Internal Imports

```python
# DON'T do this (internal modules may change)
from Core.Objects.FEM.Triangles import Triangle3  # Use Core.Objects.FEM instead
from Core.Solvers.Static import StaticLinear      # Use Core instead
```

## Memory Considerations

For large problems:

1. **Use sparse matrices**: The solvers automatically convert to sparse when beneficial
2. **Avoid storing full history**: Dynamic solvers can write to HDF5 incrementally
3. **Element loop assembly**: Matrices are assembled element-by-element, not all at once

```python
# Large problem settings
St = Hybrid()
# ... add many elements ...
St.make_nodes()

# Solver will use sparse internally
St = Static.solve_linear(St)
```

---

*Previous: [Getting Started](01_getting_started.md) | Next: [Core Concepts](03_core_concepts.md)*
