# Getting Started with HybriDFEM

This guide will help you set up your development environment and run your first HybriDFEM analysis.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** (3.11 or 3.12 recommended)
- **Git** for version control
- A code editor (VS Code, PyCharm, or similar)
- Basic knowledge of Python, NumPy, and structural mechanics

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url> HybriDFEM
cd HybriDFEM
```

### 2. Create a Virtual Environment

We strongly recommend using a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ~1.26.4 | Numerical computing |
| `scipy` | ~1.15.1 | Scientific computing, sparse matrices |
| `matplotlib` | ~3.10.0 | Visualization |
| `gmsh` | ~4.13.1 | Mesh generation |
| `meshio` | ~5.3.5 | Mesh I/O formats |
| `h5py` | ~3.13.0 | HDF5 file format for results |
| `pandas` | ~2.2.3 | Data analysis |
| `pytest` | ~8.4.2 | Testing framework |
| `pyqt6-sip` | ~13.10.2 | GUI framework |

### 4. Verify Installation

Run the test suite to verify everything is working:

```bash
pytest -v --tb=short
```

You should see all tests passing. If some tests fail, check the error messages for missing dependencies.

## Project Structure Overview

```
HybriDFEM/
├── Core/                   # Main framework library (READ THIS FIRST)
│   ├── __init__.py         # Public API exports
│   ├── Objects/            # Building blocks
│   │   ├── FEM/            # Continuous finite elements
│   │   ├── DFEM/           # Discrete rigid blocks
│   │   ├── Coupling/       # Block-FEM coupling methods
│   │   └── ConstitutiveLaw/# Material models
│   ├── Solvers/            # Analysis algorithms
│   └── Structures/         # High-level structure classes
├── Examples/               # Demonstration scripts
├── GUI/                    # Graphical interface
├── tests/                  # Test suite
└── Legacy/                 # Archived code (DO NOT USE)
```

## Your First Analysis

Let's run a simple FEM cantilever beam analysis:

### Option 1: Run an Existing Example

```bash
python Examples/Structure_FEM/cantilever.py
```

This will:
1. Create a mesh of triangular/quad elements
2. Apply boundary conditions (fixed support)
3. Apply loads
4. Solve the linear static problem
5. Display the deformed shape

### Option 2: Write Your Own Script

Create a new file `my_first_analysis.py`:

```python
"""My first HybriDFEM analysis."""
import numpy as np
from Core import Structure_FEM, Static, Visualizer
from Core.Objects.FEM import Triangle3
from Core.Objects.FEM.Element2D import Geometry2D
from Core.Objects.ConstitutiveLaw import PlaneStress

# 1. Create structure
St = Structure_FEM()

# 2. Define material and geometry
mat = PlaneStress(E=200e9, nu=0.3, rho=7850)  # Steel
geom = Geometry2D(t=0.01)  # 10mm thickness

# 3. Create a simple mesh (2x2 grid of triangles)
nodes = [
    (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
    (0.0, 0.5), (0.5, 0.5), (1.0, 0.5),
    (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
]

# Add triangular elements (counter-clockwise node ordering)
triangles = [
    [0, 1, 4], [0, 4, 3],  # Bottom-left quad as 2 triangles
    [1, 2, 5], [1, 5, 4],  # Bottom-right
    [3, 4, 7], [3, 7, 6],  # Top-left
    [4, 5, 8], [4, 8, 7],  # Top-right
]

for tri_nodes in triangles:
    elem_nodes = [nodes[i] for i in tri_nodes]
    elem = Triangle3(nodes=elem_nodes, mat=mat, geom=geom)
    St.list_fes.append(elem)

# 4. Build the structure
St.make_nodes()
print(f"Structure has {len(St.list_nodes)} nodes and {St.nb_dofs} DOFs")

# 5. Apply boundary conditions (fix left edge: nodes 0, 3, 6)
for node_id in [0, 3, 6]:
    St.fix_node(node_ids=node_id, dofs=[0, 1])  # Fix ux, uy

# 6. Apply load (force on right edge: nodes 2, 5, 8)
total_force = 10000  # 10 kN
for node_id in [2, 5, 8]:
    St.load_node(node_ids=node_id, dofs=[0], force=total_force/3)  # Fx

# 7. Solve
St = Static.solve_linear(St)

# 8. Post-process
print(f"\nMax displacement: {np.max(np.abs(St.U)):.6e} m")

# 9. Visualize
viz = Visualizer(St)
viz.plot_deformed_shape(scale=1000)
```

Run it:

```bash
python my_first_analysis.py
```

## Running Tests

HybriDFEM uses pytest for testing. Common commands:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_elements.py

# Run tests with specific marker
pytest -m fem           # FEM element tests
pytest -m dfem          # DFEM block tests
pytest -m hybrid        # Hybrid coupling tests
pytest -m solver        # Solver tests
pytest -m "not slow"    # Skip slow tests
```

## Running the GUI

HybriDFEM includes a PyQt6-based graphical interface:

```bash
python -m GUI.MainWindow
```

Note: The GUI is primarily for visualization and simple analyses. For complex workflows, use Python scripts.

## Common Issues

### ImportError: No module named 'Core'

Make sure you're running from the HybriDFEM root directory:

```bash
cd /path/to/HybriDFEM
python your_script.py
```

Or add the path in your script:

```python
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
```

### gmsh installation issues

On some systems, gmsh may require additional setup:

```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# macOS (if using brew)
brew install gmsh
```

### Matplotlib backend issues

If plots don't display, try setting the backend:

```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

## Next Steps

1. **Understand the Architecture**: Read [Architecture Overview](02_architecture.md)
2. **Learn Core Concepts**: Read [Core Concepts](03_core_concepts.md)
3. **Explore Examples**: Check the `Examples/` directory
4. **Run Tests**: Familiarize yourself with the test suite

## Quick Reference

| Task | Command |
|------|---------|
| Run all tests | `pytest` |
| Run specific tests | `pytest -m fem` |
| Run example | `python Examples/Structure_FEM/cantilever.py` |
| Run GUI | `python -m GUI.MainWindow` |
| Install deps | `pip install -r requirements.txt` |

---

*Next: [Architecture Overview](02_architecture.md)*
