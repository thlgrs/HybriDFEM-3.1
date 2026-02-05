"""
FEM Column Linear Example
==========================

Pure FEM cantilever column for comparison with:
- Hybrid examples (Examples/Structure_Hybrid/hybrid_column.py)
- Block example (Examples/Structure_Block/block_column.py)

Configuration:
- Geometry: 1m width x 4m height x 1m thickness
- Material: E = 30 GPa, nu = 0.2, rho = 2400 kg/m3
- Loading: Fx = 500 kN horizontal, Fy = -500 kN vertical at top
- BC: Fixed bottom edge (y = 0)
"""

import sys
from pathlib import Path

from Examples.utils.boundary_conditions import apply_cantilever_bc
# --- Imports ---
from Examples.utils.helpers import create_config
from Examples.utils.model_builders import create_fem_column
from Examples.utils.runner import run_example, run_examples

# =============================================================================
# CONFIGURATION
# =============================================================================
# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
EXAMPLE_ROOT = str(project_root / "Examples" / "Results" / "FEM" / "fem_column")

BASE_CONFIG = {
    "geometry": {
        "width": 1.0,
        "height": 4.0,
        "thickness": 1.0,
        "nx": 20,
        "ny": 80,
    },
    "elements": {
        "type": "quad",
        "order": 1,
    },
    "material": {
        "E": 30e9,
        "nu": 0.2,
        "rho": 2400,
    },
    "loads": {
        "Fx": 5e5,
        "Fy": -5e5,
    },
    "solver": {
        "name": "linear",
    },
    "io": {
        "filename": "fem_column",
        "dir": EXAMPLE_ROOT,
        "show_nodes": False,
        "scale": 50.0,
        "figsize": (8, 11),
        "sigma_x_range": [-5, 5],
        "sigma_y_range": [-10, 10],
        "tau_yx_range": [-2.5, 0],
        "tau_xy_range": [0, 2.5],
    },
}


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================


def apply_conditions(St, config):
    """Apply cantilever BCs: fixed base, load at top."""
    return apply_cantilever_bc(St, config, load_nodes="top", load_distribution="point")


# =============================================================================
# CONFIGURATION VARIANTS
# =============================================================================

T3 = create_config(BASE_CONFIG, "T3", elements={"type": "triangle", "order": 1})
T6 = create_config(BASE_CONFIG, "T6", elements={"type": "triangle", "order": 2})
Q4 = create_config(BASE_CONFIG, "Q4", elements={"type": "quad", "order": 1})
Q8 = create_config(BASE_CONFIG, "Q8", elements={"type": "quad", "order": 2})
Q9 = create_config(BASE_CONFIG, "Q9", elements={"type": "quad", "order": 3})

ALL_CONFIGS = [T3, T6, Q4, Q8, Q9]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def run_config(config):
    """Run a single configuration."""
    return run_example(config, create_fem_column, apply_conditions)


def run_configs(configs):
    """Run multiple configurations."""
    return run_examples(configs, create_fem_column, apply_conditions)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_configs(ALL_CONFIGS)
