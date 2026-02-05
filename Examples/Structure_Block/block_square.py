"""
Block Square Example - Linear Static Analysis
==============================================

Demonstrates rigid block grid with contact mechanics under linear loading.

Structure Type: Pure DFEM (rigid blocks with 3 DOF: ux, uy, theta_z)
Analysis: Linear static

Configuration:
- Geometry: 1m x 1m x 0.25m square
- Material: E = 10 GPa, nu = 0.0
- Contact: kn = ks = 10e9 N/m
- Loading: Compression (Fy = -1000 kN) or Shear (Fx = 1000 kN)
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
from Examples.utils.boundary_conditions import apply_compression_bc, apply_shear_bc


# =============================================================================
# CONFIGURATION
# =============================================================================

EXAMPLE_ROOT = str(project_root / "Examples" / 'Results' / 'Block' / 'block_square')

BASE_CONFIG = {
    'geometry': {
        'nx': 40,
        'ny': 40,
        'width': 1.0,
        'height': 1.0,
        'thickness': 0.25,
    },
    'material': {
        'E': 10e9,
        'nu': 0.0,
        'rho': 0,
    },
    'contact': {
        'kn': 10e9,
        'ks': 10e9,
        'LG': True,
        'nb_cps': 20,
    },
    'loads': {
        'Fx': 0,
        'Fy': 0,
        'Mz': 0,
    },
    'solver': {
        'name': 'linear',
    },
    'io': {
        'filename': 'block_square',
        'dir': EXAMPLE_ROOT,
        'show_nodes': False,
        'scale': 50,
        'figsize': (4, 4),
    }
}


# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

def apply_conditions(St, config):
    """Apply BCs based on load type in config."""
    l = config.get('loads', {})
    bc_type = l.get('name', 'compression')

    if bc_type == 'shear':
        return apply_shear_bc(St, config)
    else:
        return apply_compression_bc(St, config)


# =============================================================================
# LOAD DEFINITIONS
# =============================================================================

COMPRESSION = {
    'name': 'compression',
    'Fx': 0,
    'Fy': -1e6,
    'Mz': 0,
}

SHEAR = {
    'name': 'shear',
    'Fx': 1e6,
    'Fy': 0,
    'Mz': 0,
}


# =============================================================================
# CONFIGURATION VARIANTS
# =============================================================================

COMPRESSION_BLOCK = create_config(BASE_CONFIG, 'Compression_BlockSquare', loads=COMPRESSION)
SHEAR_BLOCK = create_config(BASE_CONFIG, 'Shear_BlockSquare', loads=SHEAR)

ALL_CONFIGS = [SHEAR_BLOCK, COMPRESSION_BLOCK]


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
    run_configs(ALL_CONFIGS)
