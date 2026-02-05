"""
Helper Utilities
================

Miscellaneous helper functions for examples.
"""
import copy
from pathlib import Path

# Define central results directory: Examples/Results
EXAMPLES_DIR = Path(__file__).parent.parent
RESULTS_ROOT = EXAMPLES_DIR / 'Results'
ELEMENT_MAP = {
    ('triangle', 1): 't3',
    ('triangle', 2): 't6',
    ('quad', 1): 'q4',
    ('quad', 2): 'q8',
    ('quad', 3): 'q9',
}

def detect_structure_type(config):
    if 'coupling' in config:
        return 'Hybrid'
    elif 'element' in config and config['element']:
        return 'FEM'
    else:
        return 'Block'


def detect_analysis_type(config):
    # 1. Explicit solver name
    if 'solver' in config and 'name' in config['solver']:
        name = config['solver']['name'].lower()
        if 'disp' in name: return 'DispCtrl'
        if 'force' in name: return 'ForceCtrl'
        if 'linear' in name: return 'Linear'
        return name.capitalize()

    # 2. Infer from control parameters
    if 'control' in config:
        if 'target_disp' in config['control']:
            return 'DispCtrl'

    # 3. Infer from loading/solver steps
    if 'solver' in config:
        if config['solver'].get('steps', 1) > 1:
            return 'ForceCtrl'

    return 'Linear'


def detect_elem_type(config):
    if 'elements' in config:
        if hasattr(config['elements'], 'prefer_quad9'):
            if config['elements']['prefer_quad9']: return ELEMENT_MAP[('quad', 3)]
        else:
            return ELEMENT_MAP[(config['elements']['type'], config['elements']['order'])]
    else:
        return ""


def create_config(BASE_CONFIG, name='', **overrides):
    """
    Create a new independent config based on BASE_CONFIG with centralized result directory.
    
    Args:
        BASE_CONFIG: Dictionary with base configuration
        name: String with new name
        **overrides: Parameters to override in the config
        
    Returns:
        new_config: Updated configuration dictionary
    """
    new_config = copy.deepcopy(BASE_CONFIG)

    # Handle cases where filename is not set or needs suffix
    new_config['io']['filename'] = name

    # Apply overrides
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in new_config:
                new_config[key].update(value)
            else:
                new_config[key] = value

    an_type = detect_analysis_type(new_config)
    el_type = detect_elem_type(new_config)
    new_config['io']['dir'] += f"/{an_type}/{el_type}"

    return new_config
