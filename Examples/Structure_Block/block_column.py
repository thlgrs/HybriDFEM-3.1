"""
Block Column Example - Linear Static Analysis
==============================================

Demonstrates rigid block grid with contact mechanics under linear loading.

Structure Type: Pure DFEM (rigid blocks with 3 DOF: ux, uy, theta_z)
Analysis: Linear static

Configuration:
- Geometry: 1m x 4m x 1m (W x H x t)
- Material: E = 30 GPa, nu = 0.2, rho = 2400 kg/m3
- Contact: kn = 1e12 N/m, ks = 1e11 N/m
- Loading: Fx = 500 kN, Fy = -500 kN uniformly distributed at top
"""

import sys
from pathlib import Path

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Imports ---
from Examples.utils.helpers import create_config
from Examples.utils.runner import run_example, run_examples
from Examples.utils.model_builders import create_block_column
from Examples.utils.boundary_conditions import apply_cantilever_bc


# =============================================================================
# CONFIGURATION
# =============================================================================

EXAMPLE_ROOT = str(project_root / "Examples" / 'Results' / 'Block' / 'block_column')

BASE_CONFIG = {
    'geometry': {
        'nx': 20,
        'ny': 80,
        'width': 1.0,
        'height': 4.0,
        'thickness': 1.0,
    },
    'material': {
        'E': 30e9,
        'nu': 0.2,
        'rho': 2400,
    },
    'contact': {
        'kn': 1e12,
        'ks': 1e11,
        'LG': True,
        'nb_cps': 20,
    },
    'loads': {
        'Fx': 5e5,
        'Fy': -5e5,
        'Mz': 0,
    },
    'solver': {
        'name': 'linear',
    },
    'io': {
        'filename': 'block_column',
        'dir': EXAMPLE_ROOT,
        'show_nodes': False,
        'scale': 50,
        'figsize': (8, 14),
    }
}


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

def apply_conditions(St, config):
    """Apply cantilever BCs: fixed base, uniform load at top."""
    return apply_cantilever_bc(St, config, load_nodes='top', load_distribution='uniform')


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_config(config):
    """Run a single configuration."""
    return run_example(config, create_block_column, apply_conditions)


def run_configs(configs):
    """Run multiple configurations."""
    return run_examples(configs, create_block_column, apply_conditions)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_config(BASE_CONFIG)
