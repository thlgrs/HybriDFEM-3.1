# Core Concepts

This document explains the fundamental concepts you need to understand when working with HybriDFEM.

## Degrees of Freedom (DOF) Conventions

### FEM Nodes: 2 DOF per Node

For continuous finite elements, each node has 2 degrees of freedom:

```
Node DOFs: [ux, uy]
- ux: displacement in x-direction
- uy: displacement in y-direction
```

Example: A mesh with 10 nodes has 20 total DOFs.

### Block Nodes: 3 DOF per Node

For discrete rigid blocks, each block has 1 node (at reference point) with 3 DOFs:

```
Block DOFs: [ux, uy, rz]
- ux: x-translation of reference point
- uy: y-translation of reference point
- rz: rotation about z-axis (out-of-plane)
```

Example: A structure with 5 blocks has 5 nodes and 15 total DOFs.

### Hybrid Structures: Variable DOF

Hybrid structures have mixed DOF counts:
- FEM nodes: 2 DOFs each
- Block nodes: 3 DOFs each

The `node_dof_offsets` array tracks where each node's DOFs start in the global system.

```python
# Example: 2 blocks + 6 FEM nodes
# Blocks: nodes 0,1 (3 DOFs each) -> offsets [0, 3, 6]
# FEM: nodes 2-7 (2 DOFs each) -> offsets continue [8, 10, 12, 14, 16, 18]
```

## Element Types

### Triangular Elements

| Element | Nodes | Order | Shape Functions | Best For |
|---------|-------|-------|-----------------|----------|
| `Triangle3` | 3 | Linear | Constant strain (CST) | Quick analyses, coarse meshes |
| `Triangle6` | 6 | Quadratic | Linear strain (LST) | Accuracy, stress recovery |

**Node numbering (counter-clockwise):**

```
Triangle3:           Triangle6:
    2                    2
   /\                   /\
  /  \                 5  4
 /    \               /    \
0------1             0---3---1
```

### Quadrilateral Elements

| Element | Nodes | Order | Description |
|---------|-------|-------|-------------|
| `Quad4` | 4 | Bilinear | Standard 4-node quad |
| `Quad8` | 8 | Serendipity | Mid-side nodes, no center |
| `Quad9` | 9 | Lagrangian | Mid-side + center nodes |

**Node numbering:**

```
Quad4:               Quad8/Quad9:
3-------2            3---6---2
|       |            |       |
|       |            7   8   5    (8 = center for Quad9 only)
|       |            |       |
0-------1            0---4---1
```

### Beam Element

`Timoshenko`: 2-node beam with shear deformation
- 3 DOFs per node: `[ux, uy, rz]`
- Includes shear correction factor

## Coordinate Systems

### Global Coordinates (x, y)

The physical coordinate system:
- x: horizontal (typically)
- y: vertical (typically)
- z: out-of-plane (for rotations)

### Natural (Local) Coordinates (xi, eta)

Used for element shape functions:
- Range: `[-1, 1]` for both xi and eta
- Origin at element center

```
        eta
         ^
         |
    (-1,1)-----(1,1)
         |     |
  -------+-----+--> xi
         |     |
   (-1,-1)-----(1,-1)
```

### Jacobian Transformation

The Jacobian relates natural to global coordinates:

```python
# Shape function derivatives in natural coords
dN_dxi, dN_deta = element.dN(xi, eta)

# Jacobian matrix
J = [[dx/dxi, dy/dxi],
     [dx/deta, dy/deta]]

# Derivatives in global coords
dN_dx = J^-1 @ dN_dxi_eta
```

## Constitutive Laws (Materials)

### PlaneStress

For thin plates where stress perpendicular to the plane is zero (`sigma_z = 0`):

```python
from Core.Objects.ConstitutiveLaw import PlaneStress

mat = PlaneStress(
    E=200e9,     # Young's modulus [Pa]
    nu=0.3,      # Poisson's ratio [-]
    rho=7850     # Density [kg/m3] (for mass matrix)
)
```

Constitutive matrix:

```
        E        [1    nu   0         ]
D = ----------- [nu   1    0         ]
    (1 - nu^2)  [0    0   (1-nu)/2   ]
```

### PlaneStrain

For thick structures where strain perpendicular to the plane is zero (`epsilon_z = 0`):

```python
from Core.Objects.ConstitutiveLaw import PlaneStrain

mat = PlaneStrain(E=30e9, nu=0.2, rho=2400)
```

### Material (Generic)

For beams and 1D elements:

```python
from Core.Objects.ConstitutiveLaw import Material

mat = Material(E=200e9, nu=0.3, rho=7850)
```

### Contact

For block contact interfaces:

```python
from Core.Objects.ConstitutiveLaw import Contact

contact = Contact(
    k_n=1e10,      # Normal stiffness [N/m]
    k_s=1e9,       # Shear stiffness [N/m]
    mu=0.6,        # Friction coefficient [-]
    c=0            # Cohesion [Pa]
)
```

## Coupling Methods

Coupling connects rigid blocks to continuous FEM elements. HybriDFEM supports four methods:

### 1. Constraint Coupling (Condensation)

**Method**: DOF elimination via transformation matrix

**How it works**:
- Detects FEM nodes coincident with blocks
- Eliminates coupled FEM DOFs from the system
- Uses transformation: `u_full = T * u_reduced`

**Advantages**:
- Exact constraint satisfaction
- Fastest solve (reduced system size)
- No parameters to tune

**Limitations**:
- Requires matching meshes (exact nodal coincidence)

```python
St.enable_block_fem_coupling(method='constraint', tolerance=1e-9)
```

### 2. Penalty Coupling

**Method**: Virtual springs at coupled nodes

**How it works**:
- Adds penalty stiffness at coupled DOFs
- Constraint: `k_penalty * (u_fem - C * q_block) = 0`

**Advantages**:
- Simple implementation
- Keeps original DOF count

**Limitations**:
- Approximate (small constraint violation)
- Needs penalty parameter tuning

```python
St.enable_block_fem_coupling(
    method='penalty',
    penalty='auto',  # or explicit value like 1e12
    tolerance=1e-9
)
```

### 3. Lagrange Multiplier Coupling

**Method**: Lagrange multipliers enforce constraints

**How it works**:
- Adds multiplier DOFs for each constraint
- Solves augmented saddle-point system

**Advantages**:
- Exact constraint satisfaction
- Direct access to interface forces (multipliers)

**Limitations**:
- Larger system (added DOFs)
- Saddle-point system (needs special solver)

```python
St.enable_block_fem_coupling(method='lagrange', tolerance=1e-9)
```

### 4. Mortar Coupling

**Method**: Distributed Lagrange multipliers along interface

**How it works**:
- Detects block-FEM interfaces geometrically
- Uses numerical integration along interface
- Evaluates FEM shape functions at integration points

**Advantages**:
- Works with non-matching meshes
- Optimal for curved interfaces
- Exact weak constraint satisfaction

**Limitations**:
- More complex implementation
- Requires integration parameters

```python
St.enable_block_fem_coupling(
    method='mortar',
    integration_order=2,       # Gauss quadrature order
    interface_tolerance=1e-4,  # Distance for interface detection
    interface_orientation='horizontal'  # or 'vertical', None
)
```

### Choosing a Coupling Method

| Scenario | Recommended Method |
|----------|-------------------|
| Matching meshes, fast solve | `constraint` |
| Simple implementation | `penalty` |
| Need interface forces | `lagrange` |
| Non-matching meshes | `mortar` |
| Curved interfaces | `mortar` |

## Solver Types

### Linear Static (`StaticLinear` / `Static`)

Solves: `K * u = P`

```python
from Core import Static

St = Static.solve_linear(St)
```

For Lagrange/Mortar coupling (saddle-point system):

```python
St = Static.solve_linear_saddle_point(St)
```

### Nonlinear Static (`StaticNonLinear`)

Newton-Raphson iteration for material/geometric nonlinearity.

**Force Control**:
```python
from Core import StaticNonLinear

St = StaticNonLinear.solve_forcecontrol(
    St,
    load_steps=10,      # Number of increments
    max_iter=50,        # Max iterations per step
    tol=1e-6            # Convergence tolerance
)
```

**Displacement Control** (for softening):
```python
St = StaticNonLinear.solve_dispcontrol(
    St,
    control_dof=dof_id,   # DOF to control
    target_disp=0.01,     # Target displacement
    steps=20
)
```

### Dynamic (`Dynamic`)

Central Difference Method (explicit time integration):

```python
from Core import Dynamic

St = Dynamic.CDM(
    St,
    dt=0.001,           # Time step [s]
    t_end=1.0,          # End time [s]
    output_interval=10  # Save every N steps
)
```

### Modal (`Modal`)

Eigenvalue analysis for natural frequencies:

```python
from Core import Modal

St = Modal.solve_modal(
    St,
    modes=10,           # Number of modes to compute
    normalize=True      # Mass-normalize eigenvectors
)

# Results
frequencies = St.frequencies  # Natural frequencies [Hz]
mode_shapes = St.mode_shapes  # Eigenvectors
```

## Boundary Conditions

### Fixing DOFs

```python
# Fix single node
St.fix_node(node_ids=0, dofs=[0, 1])  # Fix ux, uy

# Fix multiple nodes
St.fix_node(node_ids=[0, 1, 2], dofs=[0, 1])

# Fix all DOFs of a block node
St.fix_node(node_ids=block_node, dofs=[0, 1, 2])  # ux, uy, rz
```

### Applying Loads

```python
# Point load
St.load_node(node_ids=10, dofs=[0], force=1000)  # Fx = 1000 N

# Distributed load (apply to multiple nodes)
top_nodes = [5, 6, 7, 8, 9]
for node in top_nodes:
    St.load_node(node_ids=node, dofs=[1], force=-100)  # Fy = -100 N each
```

### Finding Nodes

```python
# By coordinate
node_id = St.get_node_id([0.5, 1.0], tol=1e-8)

# By position (using helper functions)
from Examples.utils.boundary_conditions import find_node_sets

node_sets = find_node_sets(St, config)
bottom_nodes = node_sets['bottom']
top_nodes = node_sets['top']
left_nodes = node_sets['left']
right_nodes = node_sets['right']
```

## Results and Post-Processing

### Displacement

```python
# Full displacement vector
U = St.U  # Shape: (nb_dofs,)

# Node displacement
node_id = 5
dofs = St.get_dofs_from_node(node_id)
ux, uy = St.U[dofs[0]], St.U[dofs[1]]
```

### Visualization

```python
from Core import Visualizer

viz = Visualizer(St)

# Deformed shape
viz.plot_deformed_shape(scale=100)

# With stress contours
viz.plot_stress_contour(component='von_mises', scale=50)
```

### Exporting Results

```python
# To HDF5
import h5py
with h5py.File('results.h5', 'w') as f:
    f.create_dataset('displacement', data=St.U)
    f.create_dataset('nodes', data=St.list_nodes)

# To CSV
import pandas as pd
df = pd.DataFrame({
    'node': range(len(St.list_nodes)),
    'x': [n[0] for n in St.list_nodes],
    'y': [n[1] for n in St.list_nodes],
})
df.to_csv('nodes.csv', index=False)
```

## Common Pitfalls

### 1. Forgetting `make_nodes()`

```python
St = Structure_FEM()
St.list_fes.append(element)
# WRONG: Forgot make_nodes()
St.fix_node(0, [0, 1])  # Error: nb_dofs not defined

# CORRECT:
St.make_nodes()  # Must call this first!
St.fix_node(0, [0, 1])
```

### 2. Wrong DOF Count for Node Type

```python
# Block node has 3 DOFs
St.fix_node(block_node, [0, 1])  # Only fixes 2 - rotation still free!
St.fix_node(block_node, [0, 1, 2])  # Correct: fix all 3
```

### 3. Element Node Ordering

Nodes must be counter-clockwise for correct Jacobian sign:

```python
# WRONG (clockwise)
nodes = [(0,0), (0,1), (1,0)]

# CORRECT (counter-clockwise)
nodes = [(0,0), (1,0), (0,1)]
```

### 4. Coupling Before `make_nodes()`

```python
St = Hybrid()
# Add elements and blocks...
St.enable_block_fem_coupling(...)  # WRONG: nodes not built yet

# CORRECT:
St.make_nodes()
St.enable_block_fem_coupling(method='constraint')
```

---

*Previous: [Architecture Overview](02_architecture.md) | Next: [Development Guide](04_development_guide.md)*
