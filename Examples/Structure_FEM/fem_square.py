"""
FEM Square Example
==================

Pure FEM square plate for comparison with:
- Hybrid examples (Examples/Structure_Hybrid/hybrid_square.py)
- Block example (Examples/Structure_Block/block_square.py)

Configuration:
- Geometry: 1m x 1m x 0.25m square plate
- Material: E = 10 GPa, nu = 0.0
- Loading: Compression (Fy = -1000 kN) or Shear (Fx = 1000 kN)
- BC: Fixed bottom edge
"""

import sys
from pathlib import Path

# --- Path Setup ---
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# --- Imports ---
from Examples.utils.helpers import create_config
from Examples.utils.runner import run_example, run_examples
from Examples.utils.model_builders import create_fem_column
from Examples.utils.boundary_conditions import apply_compression_bc, apply_shear_bc


# =============================================================================
# CONFIGURATION
# =============================================================================

EXAMPLE_ROOT = str(project_root / "Examples" / 'Results' / 'FEM' / 'illustration')

BASE_CONFIG = {
    'geometry': {
        'width': 1.0,
        'height': 1.0,
        'thickness': 0.25,
        'nx': 20,
        'ny': 40,
    },
    'elements': {
        'type': 'triangle',
        'order': 1,
    },
    'material': {
        'E': 10e9,
        'nu': 0.0,
        'rho': 0,
    },
    'loads': {
        'Fx': 0,
        'Fy': 0,
    },
    'solver': {
        'name': 'linear',
    },
    'io': {
        'filename': 'fem_square',
        'dir': EXAMPLE_ROOT,
        'show_nodes': False,
        'scale': 50.0,
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
}

SHEAR = {
    'name': 'shear',
    'Fx': 1e6,
    'Fy': 0,
}


# =============================================================================
# CONFIGURATION VARIANTS
# =============================================================================

# Compression configs
COMPRESSION_T3 = create_config(BASE_CONFIG, 'Compression_T3', loads=COMPRESSION,
                               elements={'type': 'triangle', 'order': 1})
COMPRESSION_T6 = create_config(BASE_CONFIG, 'Compression_T6', loads=COMPRESSION,
                               elements={'type': 'triangle', 'order': 2})
COMPRESSION_Q4 = create_config(BASE_CONFIG, 'Compression_Q4', loads=COMPRESSION,
                               elements={'type': 'quad', 'order': 1})
COMPRESSION_Q8 = create_config(BASE_CONFIG, 'Compression_Q8', loads=COMPRESSION,
                               elements={'type': 'quad', 'order': 2})
COMPRESSION_Q9 = create_config(BASE_CONFIG, 'Compression_Q9', loads=COMPRESSION,
                               elements={'type': 'quad', 'order': 3})

# Shear configs
SHEAR_T3 = create_config(BASE_CONFIG, 'Shear_T3', loads=SHEAR,
                         elements={'type': 'triangle', 'order': 1})
SHEAR_T6 = create_config(BASE_CONFIG, 'Shear_T6', loads=SHEAR,
                         elements={'type': 'triangle', 'order': 2})
SHEAR_Q4 = create_config(BASE_CONFIG, 'Shear_Q4', loads=SHEAR,
                         elements={'type': 'quad', 'order': 1})
SHEAR_Q8 = create_config(BASE_CONFIG, 'Shear_Q8', loads=SHEAR,
                         elements={'type': 'quad', 'order': 2})
SHEAR_Q9 = create_config(BASE_CONFIG, 'Shear_Q9', loads=SHEAR,
                         elements={'type': 'quad', 'order': 3})

COMPRESSION_CONFIGS = [COMPRESSION_T3, COMPRESSION_T6, COMPRESSION_Q4, COMPRESSION_Q8, COMPRESSION_Q9]
SHEAR_CONFIGS = [SHEAR_T3, SHEAR_T6, SHEAR_Q4, SHEAR_Q8, SHEAR_Q9]
ALL_CONFIGS = COMPRESSION_CONFIGS + SHEAR_CONFIGS


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
