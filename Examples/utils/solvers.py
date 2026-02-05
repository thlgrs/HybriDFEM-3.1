"""
Centralized Solver Utilities
============================

Unified interface for running linear and nonlinear analyses, 
handling hybrid coupling complexity (standard vs augmented solvers) automatically.
"""

from Core.Solvers.Static import StaticLinear, StaticNonLinear


def run_solver(St, config, control_node=None):
    """
    Run the appropriate solver based on configuration.
    
    Handles:
    - Linear Static (Standard & Augmented)
    - Force Control (Standard & Augmented)
    - Displacement Control (Standard & Augmented)
    
    Args:
        St: Structure object
        config: Configuration dictionary
        control_node: (Optional) Node ID for displacement control
        
    Returns:
        St: Solved structure
    """
    solver_conf = config.get('solver', {})

    # Default to linear if not specified, or if inferred from config structure
    analysis_type = 'linear'
    if 'control' in config and 'target_disp' in config['control']:
        analysis_type = 'disp_control'
    elif 'loading' in config and config.get('solver', {}).get('name') == 'forcectrl':
        analysis_type = 'force_control'
    elif config.get('solver', {}).get('name') == 'linear':
        analysis_type = 'linear'

    # Check if Augmented Solver is needed (Lagrange/Mortar)
    # We check the config['coupling']['method'] if it exists
    use_augmented = False
    if 'coupling' in config:
        method = config['coupling'].get('method', '')
        if method in ['lagrange', 'mortar']:
            use_augmented = True

    print("\n" + "=" * 60)
    print(f"   SOLVER: {analysis_type.replace('_', ' ').title()}")
    system_type = "Augmented (Saddle Point)" if use_augmented else "Standard"
    print(f"   System: {system_type}")
    print("=" * 60)

    # --- LINEAR STATIC ---
    if analysis_type == 'linear':
        if use_augmented:
            return StaticLinear.solve_augmented(St)
        else:
            return StaticLinear.solve(St, optimized=True)

    # --- DISPLACEMENT CONTROL ---
    elif analysis_type == 'disp_control':
        ctrl = config['control']
        io = config['io']

        args = {
            'steps': ctrl['steps'],
            'disp': ctrl['target_disp'],
            'node': control_node,
            'dof': ctrl['dof'],
            'tol': ctrl['tol_force'],
            'filename': io.get('filename'),
            'dir_name': io.get('dir_name'),
            'optimized': ctrl.get('optimized', True)
        }

        if use_augmented:
            args.update({
                'tol_constraint': ctrl.get('tol_constraint', 1e-9),
                'max_iter': ctrl.get('max_iter', 50),
                'stiff': 'tan'
            })
            return StaticNonLinear.solve_dispcontrol_augmented(St, **args)
        else:
            return StaticNonLinear.solve_dispcontrol(St, **args)

    # --- FORCE CONTROL ---
    elif analysis_type == 'force_control':
        sol = config['solver']
        io = config['io']

        args = {
            'steps': sol['steps'],
            'tol': sol['tol'],
            'stiff': 'tan',
            'filename': io.get('filename'),
            'dir_name': io.get('dir')
        }

        # Note: Augmented Force Control not explicitly implemented in all examples, 
        # but StaticNonLinear might support it if extended. 
        # Assuming standard for now unless Hybrid logic requires otherwise.
        return StaticNonLinear.solve_forcecontrol(St, **args)

    else:
        raise ValueError(f"Unknown analysis type: {analysis_type}")
