# Mesh Generation with GMSH

This guide explains how to generate finite element meshes using GMSH in HybriDFEM.

## Overview

HybriDFEM integrates with [GMSH](https://gmsh.info/), an open-source 3D finite element mesh generator, to create high-quality meshes for FEM analysis. The integration is provided through the `Mesh` class in `Core/Objects/FEM/Mesh.py`.

## Prerequisites

Ensure GMSH and meshio are installed:

```bash
pip install gmsh meshio
```

## Basic Usage

### Creating a Simple Rectangular Mesh

```python
from Core.Objects.FEM.Mesh import Mesh
from Core.Structures.Structure_FEM import Structure_FEM
from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D

# 1. Define geometry as a list of corner points (counter-clockwise)
points = [
    (0, 0),      # Bottom-left
    (2, 0),      # Bottom-right
    (2, 1),      # Top-right
    (0, 1),      # Top-left
]

# 2. Create Mesh object
mesh = Mesh(
    points=points,
    element_type='triangle',   # 'triangle' or 'quad'
    element_size=0.1,          # Target element size [m]
    order=1,                   # 1=linear, 2=quadratic
    name="my_mesh"
)

# 3. Generate the mesh (calls GMSH internally)
mesh.generate_mesh()

# 4. Create FEM structure from mesh
material = PlaneStress(E=200e9, nu=0.3, rho=7850)
geometry = Geometry2D(t=0.01)  # 10mm thickness

St = Structure_FEM.from_mesh(mesh, material, geometry)
St.make_nodes()

print(f"Generated {len(St.list_nodes)} nodes, {len(St.list_fes)} elements")
```

## Mesh Class Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | List[Tuple] | None | Corner points defining the polygon boundary |
| `mesh_file` | str | None | Path to existing mesh file (alternative to points) |
| `element_type` | str | 'triangle' | Element type: 'triangle'/'tri' or 'quad' |
| `element_size` | float | 0.1 | Target element size in model units |
| `order` | int | 2 | Element order: 1 (linear) or 2 (quadratic) |
| `name` | str | 'myMesh' | Mesh name (used for output file) |
| `edge_groups` | Dict | None | Named boundary groups for BC application |

## Element Types and Orders

| element_type | order | HybriDFEM Element | Nodes | Description |
|--------------|-------|-------------------|-------|-------------|
| 'triangle' | 1 | Triangle3 | 3 | Linear triangle (CST) |
| 'triangle' | 2 | Triangle6 | 6 | Quadratic triangle (LST) |
| 'quad' | 1 | Quad4 | 4 | Bilinear quadrilateral |
| 'quad' | 2 | Quad8 | 8 | Serendipity quadrilateral |
| 'quad' | 2 | Quad9 | 9 | Lagrangian quadrilateral* |

*Quad9 requires `prefer_quad9=True` when calling `Structure_FEM.from_mesh()`

## Defining Boundary Groups

Edge groups allow you to name boundaries for easy BC application:

```python
# Points define a rectangle (counter-clockwise from bottom-left)
points = [(0, 0), (L, 0), (L, H), (0, H)]

# Edge indices correspond to edges between consecutive points:
#   Edge 0: (0,0) -> (L,0)   = bottom
#   Edge 1: (L,0) -> (L,H)   = right
#   Edge 2: (L,H) -> (0,H)   = top
#   Edge 3: (0,H) -> (0,0)   = left

edge_groups = {
    "bottom": [0],    # Edge index 0
    "right": [1],     # Edge index 1
    "top": [2],       # Edge index 2
    "left": [3],      # Edge index 3
}

mesh = Mesh(
    points=points,
    element_type='triangle',
    element_size=0.1,
    order=1,
    edge_groups=edge_groups
)
mesh.generate_mesh()

# Later, get nodes on a boundary:
left_nodes = mesh.get_boundary_nodes("left")
print(f"Nodes on left edge: {left_nodes}")
```

## Complete Example: Cantilever Beam

```python
"""Cantilever beam with GMSH mesh."""
import numpy as np
from Core.Objects.FEM.Mesh import Mesh
from Core.Structures.Structure_FEM import Structure_FEM
from Core.Objects.ConstitutiveLaw.Material import PlaneStress
from Core.Objects.FEM.Element2D import Geometry2D
from Core import Static, Visualizer

# === Geometry ===
L = 3.0   # Length [m]
H = 0.5   # Height [m]
t = 0.2   # Thickness [m]

# === Define mesh ===
points = [(0, 0), (L, 0), (L, H), (0, H)]
edge_groups = {
    "left": [3],    # Fixed support
    "right": [1],   # Load application
}

mesh = Mesh(
    points=points,
    element_type='triangle',
    element_size=0.1,
    order=2,  # Quadratic elements (Triangle6)
    edge_groups=edge_groups,
    name="cantilever"
)
mesh.generate_mesh()

# === Create structure ===
mat = PlaneStress(E=30e9, nu=0.2, rho=2400)
geom = Geometry2D(t=t)

St = Structure_FEM.from_mesh(mesh, mat, geom)
St.make_nodes()

print(f"Mesh: {len(St.list_nodes)} nodes, {len(St.list_fes)} elements")

# === Boundary conditions ===
# Fix left edge
left_nodes = mesh.get_boundary_nodes("left")
for node in left_nodes:
    St.fix_node(node, [0, 1])  # Fix ux, uy

# Apply distributed load on right edge
right_nodes = mesh.get_boundary_nodes("right")
total_load = -100e3  # 100 kN downward
load_per_node = total_load / len(right_nodes)
for node in right_nodes:
    St.load_node(node, [1], load_per_node)  # Fy

print(f"Fixed {len(left_nodes)} nodes, loaded {len(right_nodes)} nodes")

# === Solve ===
St = Static.solve_linear(St)

# === Results ===
# Find tip displacement (rightmost node)
max_x = max(n[0] for n in St.list_nodes)
tip_node = next(i for i, n in enumerate(St.list_nodes) if abs(n[0] - max_x) < 1e-6)
tip_dofs = St.get_dofs_from_node(tip_node)
print(f"\nTip displacement: uy = {St.U[tip_dofs[1]]:.6e} m")

# === Visualize ===
viz = Visualizer(St)
viz.plot_deformed_shape(scale=50)
```

## Advanced: Complex Geometries

### L-Shaped Domain

```python
# L-shaped domain (counter-clockwise)
points = [
    (0, 0),
    (2, 0),
    (2, 1),
    (1, 1),
    (1, 2),
    (0, 2),
]

mesh = Mesh(
    points=points,
    element_type='triangle',
    element_size=0.1,
    order=1
)
mesh.generate_mesh()
```

### Polygon with Hole

For domains with holes, you need to use GMSH directly or create the mesh externally and load it:

```python
# Load existing mesh file
mesh = Mesh(mesh_file="domain_with_hole.msh")
mesh.read_mesh()
```

## Batch Mesh Generation

For multiple disconnected domains (e.g., hybrid structures), use batch generation:

```python
from Core.Objects.FEM.Mesh import Mesh

# Define multiple surfaces
surfaces = [
    {
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'element_size': 0.05,
        'name': 'block_1'
    },
    {
        'points': [(1.5, 0), (2.5, 0), (2.5, 1), (1.5, 1)],
        'element_size': 0.1,
        'name': 'block_2'
    },
]

# Generate all meshes in one GMSH call (more efficient)
meshes = Mesh.generate_batch(
    surfaces=surfaces,
    element_type='triangle',
    order=1,
    verbose=True
)

# Each mesh can be used independently
for m in meshes:
    print(f"{m.name}: {len(m.nodes())} nodes, {len(m.elements())} elements")
```

## Mesh Quality and Refinement

### Controlling Element Size

```python
# Coarse mesh
mesh_coarse = Mesh(points=points, element_size=0.5, ...)

# Fine mesh
mesh_fine = Mesh(points=points, element_size=0.05, ...)
```

### Convergence Studies

For convergence studies, generate meshes with decreasing element sizes:

```python
sizes = [0.2, 0.1, 0.05, 0.025]
results = []

for size in sizes:
    mesh = Mesh(points=points, element_size=size, order=1, ...)
    mesh.generate_mesh()

    St = Structure_FEM.from_mesh(mesh, mat, geom)
    St.make_nodes()
    # ... apply BCs and solve ...

    results.append({
        'size': size,
        'nodes': len(St.list_nodes),
        'displacement': St.U[tip_dof]
    })
```

## Visualizing the Mesh

```python
# Plot mesh (opens matplotlib window)
mesh.plot()

# Save mesh plot to file
mesh.plot(save_path="mesh_plot.png", title="My FEM Mesh")
```

## Accessing Mesh Data

```python
# Get node coordinates
nodes = mesh.nodes()  # Shape: (n_nodes, 2)
print(f"Node 0 at: {nodes[0]}")

# Get element connectivity
elements = mesh.elements()  # Shape: (n_elements, nodes_per_element)
print(f"Element 0 connects nodes: {elements[0]}")

# Get boundary nodes
boundary_nodes = mesh.get_boundary_nodes("left")

# Get all available boundary groups
groups = mesh.get_all_boundary_groups()
print(f"Available groups: {list(groups.keys())}")
```

## Common Issues and Solutions

### GMSH Not Found

```
ImportError: GMSH is not available
```

**Solution**: Install GMSH with `pip install gmsh`. On some systems, you may also need:
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# macOS
brew install gmsh
```

### Mesh Generation Fails

If `generate_mesh()` fails, check:
1. Points are in counter-clockwise order
2. Polygon is simple (no self-intersection)
3. Element size is reasonable for the geometry

### Empty Elements Array

```python
elements = mesh.elements()
print(elements.shape)  # (0, 0) - empty!
```

**Solution**: Ensure mesh was generated successfully:
```python
mesh.generate_mesh()
assert mesh.generated, "Mesh generation failed"
```

## Alternative: Structured Meshes

For simple rectangular domains, you can use the built-in structured mesh generator instead of GMSH:

```python
# Structured mesh (no GMSH needed)
St = Structure_FEM.from_rectangular_grid(
    nx=10,              # Elements in x
    ny=5,               # Elements in y
    length=2.0,         # Domain length
    height=1.0,         # Domain height
    element_type='triangle',
    order=1,
    material=mat,
    geometry=geom
)
```

This is faster for rectangular domains but doesn't support complex geometries.

## References

- [GMSH Documentation](https://gmsh.info/doc/texinfo/gmsh.html)
- [Meshio Documentation](https://github.com/nschloe/meshio)

---

*Previous: [Extending HybriDFEM](08_extending.md) | Back to: [Documentation Index](../README.md)*
