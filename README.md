# HybriDFEM Documentation

Welcome to the HybriDFEM documentation. This guide is designed for new students and developers who will work on
extending and maintaining the HybriDFEM framework.

## What is HybriDFEM?

**HybriDFEM** is a Python framework for hybrid structural analysis developed at UCLouvain. It combines:

- **Discrete Finite Element Method (DFEM)**: Rigid block assemblies with contact mechanics
- **Continuous Finite Element Method (FEM)**: Triangular and quadrilateral plane stress/strain elements
- **Hybrid Coupling**: Four methods for block-FEM interaction

## Documentation Structure

| Document                                                      | Description                                         |
|---------------------------------------------------------------|-----------------------------------------------------|
| [1. Getting Started](Documentation/01_getting_started.md)     | Installation, dependencies, and first run           |
| [2. Architecture Overview](Documentation/02_architecture.md)  | Code organization, class hierarchy, design patterns |
| [3. Core Concepts](Documentation/03_core_concepts.md)         | DOF conventions, element types, coupling methods    |
| [4. Development Guide](Documentation/04_development_guide.md) | Coding standards, testing, debugging                |
| [5. Git Workflow](Documentation/05_git_workflow.md)           | Branching, rebasing, merging, pull requests         |
| [6. Examples Guide](Documentation/06_examples_guide.md)       | How to run and create examples                      |
| [7. API Reference](Documentation/07_api_reference.md)         | Quick reference for main classes and functions      |
| [8. Extending HybriDFEM](Documentation/08_extending.md)       | Adding new elements, solvers, materials             |
| [9. Mesh Generation](Documentation/09_mesh_generation.md)     | Creating FEM meshes with GMSH                       |

## Quick Links

### For New Users

1. Start with [Getting Started](Documentation/01_getting_started.md) to set up your environment
2. Read [Architecture Overview](Documentation/02_architecture.md) to understand the codebase
3. Follow [Examples Guide](Documentation/06_examples_guide.md) to run your first analysis

### For Developers

1. Read [Development Guide](Documentation/04_development_guide.md) for coding standards
2. Follow [Git Workflow](Documentation/05_git_workflow.md) before making changes
3. Check [Extending HybriDFEM](Documentation/08_extending.md) when adding new features

## Project Repository Structure

```
HybriDFEM/
├── Core/                   # Main framework library
│   ├── Objects/            # Building blocks (FEM, DFEM, Coupling, Materials)
│   ├── Solvers/            # Analysis algorithms
│   └── Structures/         # High-level structure classes
├── Examples/               # Demonstration scripts
│   ├── Structure_FEM/      # Pure FEM examples
│   ├── Structure_Block/    # Pure block examples
│   ├── Structure_Hybrid/   # Hybrid coupling examples
│   └── utils/              # Shared example utilities
├── GUI/                    # PyQt6-based graphical interface
├── tests/                  # Test suite
├── Legacy/                 # Archived code (reference only)
└── Documentation/          # This documentation
```

---

*This documentation was generated with the assistance of [Claude Code](https://claude.ai/code), Anthropic's AI coding
assistant.*
