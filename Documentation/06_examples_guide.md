# Examples Guide

This guide explains how to run existing examples and create new ones in HybriDFEM.

## Examples Directory Structure

```
Examples/
├── Structure_FEM/          # Pure FEM examples
│   ├── cantilever.py       # Cantilever beam
│   ├── fem_square.py       # Square plate
│   ├── fem_column.py       # Column under load
│   ├── cantilever_gmsh.py  # Using Gmsh meshes
│   └── convergence*.py     # Convergence studies
├── Structure_Block/        # Pure block examples
│   ├── block_square.py     # Single block
│   └── block_column.py     # Stacked blocks
├── Structure_Hybrid/       # Hybrid coupling examples
│   ├── hybrid_column.py    # Block + FEM column
│   ├── hybrid_beam.py      # Block + FEM beam
│   ├── hybrid_square.py    # Square with mixed discretization
│   └── fem_fem_mesh_*.py   # FEM-FEM coupling examples
├── Refactored_Legacy/      # Migrated legacy examples
│   ├── Modal_Analysis/     # Eigenvalue problems
│   ├── Linear_Dynamic/     # Time-history analyses
│   ├── Nonlinear_Dynamic/  # Rocking, impacts
│   └── ...                 # Research examples
├── convergence/            # Convergence studies
└── utils/                  # Shared utilities
    ├── model_builders.py   # High-level structure builders
    ├── mesh_generation.py  # Mesh creation helpers
    ├── solvers.py          # Solver runners
    ├── boundary_conditions.py  # BC helpers
    ├── visualization.py    # Plotting utilities
    └── runner.py           # Example execution framework
```

## Running Examples

### Basic Usage

```bash
# Run from the HybriDFEM root directory
python Examples/Structure_FEM/cantilever.py

# Or run a hybrid example
python Examples/Structure_Hybrid/hybrid_column.py
```

### Example Output

Most examples will:
1. Print progress information
2. Show numerical results
3. Display or save plots
4. Optionally save results to `Examples/*/Results/`

```bash
$ python Examples/Structure_FEM/cantilever.py

Building FEM: triangle Order 1 (4x8)
[OK] Generated 45 Nodes, 90 DOFs

Applying Conditions: CANTILEVER
  -> Fixed left edge: 9 nodes
  -> Applied Fy = -10.0 kN on 9 nodes

Solving Linear Static...
[OK] Solved in 0.05s

Results:
  Max displacement: 2.34e-04 m
  Control node displacement: 2.31e-04 m
```

## Understanding Example Structure

Most examples follow a consistent pattern:

```python
"""
Example Title
=============

Brief description of what this example demonstrates.

Structure Type: FEM / Block / Hybrid
Analysis: Linear static / Nonlinear / Dynamic / Modal
"""

import sys
from pathlib import Path

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Imports ---
from Core import Hybrid, Static, Visualizer
from Examples.utils.model_builders import create_hybrid_column
from Examples.utils.runner import run_example

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_CONFIG = {
    'geometry': {
        'width': 1.0,
        'height': 4.0,
        'thickness': 0.1,
        # ...
    },
    'material': {
        'E': 200e9,
        'nu': 0.3,
        'rho': 7850,
    },
    'contact': {
        'kn': 1e10,
        'ks': 1e9,
        # ...
    },
    'coupling': {
        'method': 'constraint',
        'tolerance': 1e-9,
    },
    'elements': {
        'type': 'triangle',
        'order': 1,
    },
    'loads': {
        'Fx': 1e5,
        'Fy': -1e5,
    },
    'bc': {
        'type': 'cantilever',
    },
    'solver': {
        'name': 'linear',
    },
    'io': {
        'filename': 'example_output',
        'dir': 'Results/',
        'scale': 100,
    }
}

# =============================================================================
# MODEL GENERATION
# =============================================================================

def create_model(config):
    """Create the structure from configuration."""
    return create_hybrid_column(config)

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

def apply_conditions(St, config):
    """Apply boundary conditions and loads."""
    # Fix base nodes
    bottom_nodes = [i for i, n in enumerate(St.list_nodes) if n[1] < 1e-6]
    for node in bottom_nodes:
        St.fix_node(node_ids=node, dofs=[0, 1, 2])

    # Apply loads
    top_nodes = [i for i, n in enumerate(St.list_nodes) if n[1] > 3.99]
    Fx = config['loads']['Fx']
    for node in top_nodes:
        St.load_node(node_ids=node, dofs=[0], force=Fx / len(top_nodes))

    control_node = top_nodes[len(top_nodes) // 2]
    return St, control_node

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_example(BASE_CONFIG, create_model, apply_conditions)
```

## Using the Utils Package

### Model Builders

The `Examples/utils/model_builders.py` module provides high-level builders:

```python
from Examples.utils.model_builders import (
    create_block_column,      # Pure block structure
    create_fem_column,        # Pure FEM structure
    create_hybrid_column,     # Hybrid (FEM on top or bottom)
    create_hybrid_column_slices,  # Alternating slices
    create_hybrid_beam_slices,    # Beam with vertical slices
)

# Example: Create hybrid column with config
config = {
    'geometry': {
        'width': 0.2,
        'thickness': 0.2,
        'n_slices': 4,
        'block_slice_height': 0.5,
        'fem_slice_height': 0.5,
        'nx': 2,
        'ny_block_slice': 2,
        'ny_fem_slice': 4,
        'start_with': 'block',
    },
    'material': {...},
    'contact': {...},
    'coupling': {...},
    'elements': {...},
}

St = create_hybrid_column_slices(config)
```

### Mesh Generation

```python
from Examples.utils.mesh_generation import (
    generate_node_grid,       # Regular node grid
    create_triangle_elements, # Triangle mesh from nodes
    create_quad_elements,     # Quad mesh from nodes
    generate_block_grid,      # Block grid
    generate_block_slice,     # Horizontal block layer
)

# Generate a 5x10 grid of nodes
nodes, nx, ny = generate_node_grid(
    nx_elem=5,      # Elements in x
    ny_elem=10,     # Elements in y
    width=1.0,
    height=2.0,
    order=1         # 1=linear, 2=quadratic
)
```

### Boundary Conditions

```python
from Examples.utils.boundary_conditions import find_node_sets

# Automatically find node sets by position
node_sets = find_node_sets(St, config)

bottom_nodes = node_sets['bottom']    # y = y_min
top_nodes = node_sets['top']          # y = y_max
left_nodes = node_sets['left']        # x = x_min
right_nodes = node_sets['right']      # x = x_max
center_x = node_sets['center_x']      # x = (x_min + x_max) / 2
center_y = node_sets['center_y']      # y = (y_min + y_max) / 2
```

### Solver Runner

```python
from Examples.utils.runner import run_example, run_examples

# Run single configuration
result = run_example(config, create_model, apply_conditions)

# Run multiple configurations (e.g., parameter study)
configs = [config1, config2, config3]
results = run_examples(configs, create_model, apply_conditions)
```

### Visualization

```python
from Examples.utils.visualization import (
    plot_structure,
    plot_displacement,
    plot_comparison,
)

# Plot structure
fig, ax = plot_structure(St, show_nodes=True, show_deformed=True, scale=100)

# Compare multiple results
results = [result1, result2, result3]
labels = ['Constraint', 'Penalty', 'Lagrange']
fig = plot_comparison(results, labels, metric='max_displacement')
```

## Creating New Examples

### Step 1: Choose a Location

Place your example in the appropriate directory:

- `Examples/Structure_FEM/` - Pure FEM analyses
- `Examples/Structure_Block/` - Pure block analyses
- `Examples/Structure_Hybrid/` - Hybrid coupling
- `Examples/Refactored_Legacy/<category>/` - Research-specific

### Step 2: Use the Template

Create a new file following this template:

```python
"""
My New Example
==============

Description of what this example demonstrates.

Structure Type: [FEM / Block / Hybrid]
Analysis: [Linear static / Nonlinear / Dynamic / Modal]
"""

import sys
from pathlib import Path

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Imports ---
from Core import Structure_FEM, Static, Visualizer
from Core.Objects.FEM import Triangle3
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.ConstitutiveLaw import PlaneStress

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define parameters
WIDTH = 1.0
HEIGHT = 2.0
THICKNESS = 0.01
NX, NY = 4, 8
E = 200e9
NU = 0.3
LOAD = 10000

# =============================================================================
# BUILD MODEL
# =============================================================================

def create_structure():
    """Build the structure."""
    St = Structure_FEM()

    # Material and geometry
    mat = PlaneStress(E=E, nu=NU, rho=7850)
    geom = Geometry2D(t=THICKNESS)

    # Create mesh (your mesh generation here)
    # ...

    St.make_nodes()
    return St


def apply_boundary_conditions(St):
    """Apply BCs and loads."""
    # Fix left edge
    for i, node in enumerate(St.list_nodes):
        if abs(node[0]) < 1e-9:
            St.fix_node(node_ids=i, dofs=[0, 1])

    # Apply load on right edge
    for i, node in enumerate(St.list_nodes):
        if abs(node[0] - WIDTH) < 1e-9:
            St.load_node(node_ids=i, dofs=[0], force=LOAD / NY)

    return St


# =============================================================================
# ANALYSIS
# =============================================================================

def run_analysis():
    """Run the analysis."""
    print("=" * 60)
    print("   MY NEW EXAMPLE")
    print("=" * 60)

    # Build
    St = create_structure()
    print(f"[OK] Created structure: {len(St.list_nodes)} nodes, {St.nb_dofs} DOFs")

    # Apply BCs
    St = apply_boundary_conditions(St)
    print(f"[OK] Applied boundary conditions")

    # Solve
    print("\nSolving...")
    St = Static.solve_linear(St)
    print(f"[OK] Solved")

    # Results
    print("\n--- Results ---")
    print(f"Max displacement: {max(abs(St.U)):.6e} m")

    # Visualize
    viz = Visualizer(St)
    viz.plot_deformed_shape(scale=100, title="My Example")

    return St


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    St = run_analysis()
```

### Step 3: Test Your Example

```bash
# Run your example
python Examples/Structure_FEM/my_new_example.py

# Ensure no errors
# Check output makes sense
```

### Step 4: Add Documentation

Add a docstring at the top explaining:
- What the example demonstrates
- Expected results
- Any special requirements

## Example Categories

### Pure FEM Examples

**Location:** `Examples/Structure_FEM/`

Demonstrate continuous finite element analysis:

```python
# Simple cantilever beam
from Core import Structure_FEM, Static

St = Structure_FEM()
# Add triangle/quad elements
St.make_nodes()
# Apply BCs
St = Static.solve_linear(St)
```

**Available examples:**
- `cantilever.py` - Classic cantilever beam
- `fem_square.py` - Square plate under load
- `fem_column.py` - Column compression
- `convergence*.py` - Mesh convergence studies

### Pure Block Examples

**Location:** `Examples/Structure_Block/`

Demonstrate discrete element analysis with rigid blocks:

```python
from Core import Structure_Block, Static
from Core.Objects.DFEM import Block_2D

St = Structure_Block()
St.add_block(block)
St.make_nodes()
St.detect_interfaces()
St.make_cfs(contact=contact_law)
St = Static.solve_linear(St)
```

**Available examples:**
- `block_square.py` - Single square block
- `block_column.py` - Stacked block column

### Hybrid Examples

**Location:** `Examples/Structure_Hybrid/`

Demonstrate block-FEM coupling:

```python
from Core import Hybrid, Static

St = Hybrid()
St.add_block(block)
St.add_fe(element)
St.make_nodes()
St.detect_interfaces()
St.make_cfs(contact=contact_law)
St.enable_block_fem_coupling(method='constraint')
St = Static.solve_linear(St)
```

**Available examples:**
- `hybrid_column.py` - Column with alternating slices
- `hybrid_beam.py` - Beam with vertical slices
- `hybrid_square.py` - Square with mixed discretization
- `square_benchmark.py` - Benchmark comparisons
- `fem_fem_mesh_match.py` - FEM-FEM with matching meshes
- `fem_fem_mesh_nonmatch.py` - FEM-FEM with non-matching meshes

### Research Examples (Refactored Legacy)

**Location:** `Examples/Refactored_Legacy/`

Research examples migrated from Legacy:

- `Modal_Analysis/` - Eigenvalue problems
- `Linear_Dynamic/` - Linear time-history
- `Nonlinear_Dynamic/` - Rocking, impact
- `COMPDYN_2023/`, `COMPDYN_2025/` - Conference examples
- `IABSE_2024/`, `EESD_2024/` - Publication examples

## Running Parameter Studies

For systematic parameter studies, use configuration variants:

```python
# Base configuration
BASE_CONFIG = {...}

# Create variants
from Examples.utils.helpers import create_config

configs = [
    create_config(BASE_CONFIG, 'T3', elements={'type': 'triangle', 'order': 1}),
    create_config(BASE_CONFIG, 'T6', elements={'type': 'triangle', 'order': 2}),
    create_config(BASE_CONFIG, 'Q4', elements={'type': 'quad', 'order': 1}),
    create_config(BASE_CONFIG, 'Q8', elements={'type': 'quad', 'order': 2}),
]

# Run all
from Examples.utils.runner import run_examples
results = run_examples(configs, create_model, apply_conditions)
```

## Saving Results

Results are typically saved to `Examples/*/Results/`:

```python
import h5py
from pathlib import Path

# Create results directory
results_dir = Path("Examples/Structure_FEM/Results")
results_dir.mkdir(parents=True, exist_ok=True)

# Save to HDF5
with h5py.File(results_dir / "results.h5", "w") as f:
    f.create_dataset("displacement", data=St.U)
    f.create_dataset("nodes", data=St.list_nodes)
    f.attrs["max_disp"] = float(max(abs(St.U)))

# Save plot
fig.savefig(results_dir / "deformed_shape.png", dpi=150, bbox_inches='tight')
```

Note: `Results/` directories are in `.gitignore` - don't commit result files.

---

*Previous: [Git Workflow](05_git_workflow.md) | Next: [API Reference](07_api_reference.md)*
