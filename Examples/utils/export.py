"""
Structure State Export Utilities
=================================

Export structure state (displacements, stresses, node positions) to CSV files
for easy analysis with pandas or other tools.

Usage:
    from Examples.utils.export import export_state, export_displacements, export_stresses

    # Export everything
    export_state(St, config)

    # Or export individually
    export_displacements(St, config)
    export_stresses(St, config)

    # Export linear system Ku = P
    from Examples.utils.export import export_system, export_stiffness_matrix, export_load_vector
    export_system(St, config)  # Exports both K and P
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np


def export_nodes(St, config, filename: Optional[str] = None) -> str:
    """
    Export node positions to CSV.

    Columns: node_id, x, y

    Args:
        St: Structure object
        config: Configuration dictionary with 'io' section
        filename: Optional custom filename (without extension)

    Returns:
        Path to saved file
    """
    io = config['io']
    os.makedirs(io['dir'], exist_ok=True)

    if filename is None:
        filename = io['filename'] + '_nodes'

    filepath = os.path.join(io['dir'], filename + '.csv')

    # Build data
    lines = ['node_id,x,y']
    for i, node in enumerate(St.list_nodes):
        lines.append(f'{i},{node[0]:.12e},{node[1]:.12e}')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Exported nodes to: {filepath}")
    return filepath


def export_displacements(St, config, filename: Optional[str] = None) -> str:
    """
    Export displacement field to CSV.

    For FEM (2 DOF/node): node_id, x, y, ux, uy, u_mag
    For Block (3 DOF/node): node_id, x, y, ux, uy, rz, u_mag

    Args:
        St: Structure object with U displacement vector
        config: Configuration dictionary with 'io' section
        filename: Optional custom filename (without extension)

    Returns:
        Path to saved file
    """
    io = config['io']
    os.makedirs(io['dir'], exist_ok=True)

    if filename is None:
        filename = io['filename'] + '_displacements'

    filepath = os.path.join(io['dir'], filename + '.csv')

    # Expand displacement if constraint coupling
    U = St.U
    if hasattr(St, 'coupling_T') and St.coupling_T is not None:
        U = St.coupling_T @ St.U

    # Detect DOFs per node
    has_blocks = hasattr(St, 'list_blocks') and St.list_blocks
    n_nodes = len(St.list_nodes)

    # Build data
    if has_blocks:
        # Block structure: 3 DOF per node
        lines = ['node_id,x,y,ux,uy,rz,u_mag']
        for i, node in enumerate(St.list_nodes):
            dofs = St.get_dofs_from_node(i)
            ux = U[dofs[0]]
            uy = U[dofs[1]]
            rz = U[dofs[2]] if len(dofs) > 2 else 0.0
            u_mag = np.sqrt(ux**2 + uy**2)
            lines.append(f'{i},{node[0]:.12e},{node[1]:.12e},{ux:.12e},{uy:.12e},{rz:.12e},{u_mag:.12e}')
    else:
        # FEM structure: 2 DOF per node
        lines = ['node_id,x,y,ux,uy,u_mag']
        for i, node in enumerate(St.list_nodes):
            dofs = St.get_dofs_from_node(i)
            ux = U[dofs[0]]
            uy = U[dofs[1]]
            u_mag = np.sqrt(ux**2 + uy**2)
            lines.append(f'{i},{node[0]:.12e},{node[1]:.12e},{ux:.12e},{uy:.12e},{u_mag:.12e}')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Exported displacements to: {filepath}")
    return filepath


def export_stresses(St, config, filename: Optional[str] = None) -> str:
    """
    Export stress field to CSV.

    For FEM elements: element_id, centroid_x, centroid_y, sigma_xx, sigma_yy, sigma_xy, von_mises
    For Block structures: exports contact face stresses if available

    Args:
        St: Structure object with computed stresses
        config: Configuration dictionary with 'io' section
        filename: Optional custom filename (without extension)

    Returns:
        Path to saved file
    """
    io = config['io']
    os.makedirs(io['dir'], exist_ok=True)

    if filename is None:
        filename = io['filename'] + '_stresses'

    filepath = os.path.join(io['dir'], filename + '.csv')

    has_fem = hasattr(St, 'list_fes') and St.list_fes
    has_blocks = hasattr(St, 'list_blocks') and St.list_blocks

    lines = []

    if has_fem:
        # Export FEM element stresses
        lines.append('element_id,centroid_x,centroid_y,sigma_xx,sigma_yy,sigma_xy,von_mises')

        # Expand displacement if needed
        U = St.U
        if hasattr(St, 'coupling_T') and St.coupling_T is not None:
            U = St.coupling_T @ St.U

        for elem_id, fe in enumerate(St.list_fes):
            # Get element centroid
            nodes = np.array(fe.nodes)
            centroid = nodes.mean(axis=0)

            # Get element stresses at centroid (natural coords 0,0)
            try:
                # Get element DOFs
                elem_dofs = []
                for node_coords in fe.nodes:
                    # Find node index
                    for i, n in enumerate(St.list_nodes):
                        if abs(n[0] - node_coords[0]) < 1e-10 and abs(n[1] - node_coords[1]) < 1e-10:
                            elem_dofs.extend(St.get_dofs_from_node(i))
                            break

                # Get element displacements
                u_elem = U[elem_dofs]

                # Compute stress at centroid
                stress = fe.get_stress(u_elem, xi=0, eta=0)
                sigma_xx = stress[0, 0]
                sigma_yy = stress[1, 0]
                sigma_xy = stress[2, 0]

                # Von Mises stress (plane stress)
                von_mises = np.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*sigma_xy**2)

                lines.append(f'{elem_id},{centroid[0]:.12e},{centroid[1]:.12e},'
                           f'{sigma_xx:.12e},{sigma_yy:.12e},{sigma_xy:.12e},{von_mises:.12e}')
            except Exception as e:
                # If stress computation fails, write NaN
                lines.append(f'{elem_id},{centroid[0]:.12e},{centroid[1]:.12e},nan,nan,nan,nan')

    if has_blocks and hasattr(St, 'list_cfs') and St.list_cfs:
        # Export contact face forces/stresses
        cf_filepath = os.path.join(io['dir'], filename + '_contact.csv')
        cf_lines = ['cf_id,x1,y1,x2,y2,normal_force,shear_force']

        for cf_id, cf in enumerate(St.list_cfs):
            try:
                # Get contact face endpoints
                x1, y1 = cf.x1, cf.y1
                x2, y2 = cf.x2, cf.y2

                # Sum forces from contact points
                fn_total = 0.0
                fs_total = 0.0
                if hasattr(cf, 'list_cps'):
                    for cp in cf.list_cps:
                        if hasattr(cp, 'fn'):
                            fn_total += cp.fn
                        if hasattr(cp, 'fs'):
                            fs_total += cp.fs

                cf_lines.append(f'{cf_id},{x1:.12e},{y1:.12e},{x2:.12e},{y2:.12e},{fn_total:.12e},{fs_total:.12e}')
            except Exception:
                pass

        if len(cf_lines) > 1:
            with open(cf_filepath, 'w') as f:
                f.write('\n'.join(cf_lines))
            print(f"  Exported contact stresses to: {cf_filepath}")

    if lines:
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  Exported stresses to: {filepath}")

    return filepath


def export_reactions(St, config, filename: Optional[str] = None) -> str:
    """
    Export reaction forces at fixed nodes to CSV.

    Columns: node_id, x, y, Rx, Ry, [Mz for blocks]

    Args:
        St: Structure object
        config: Configuration dictionary with 'io' section
        filename: Optional custom filename (without extension)

    Returns:
        Path to saved file
    """
    io = config['io']
    os.makedirs(io['dir'], exist_ok=True)

    if filename is None:
        filename = io['filename'] + '_reactions'

    filepath = os.path.join(io['dir'], filename + '.csv')

    # Check if we have fixed DOFs
    if not hasattr(St, 'dof_fix') or len(St.dof_fix) == 0:
        print("  No fixed DOFs found, skipping reactions export")
        return ""

    # Get internal forces (reactions at fixed DOFs)
    if not hasattr(St, 'P') or St.P is None:
        print("  No internal forces computed, skipping reactions export")
        return ""

    has_blocks = hasattr(St, 'list_blocks') and St.list_blocks

    # Group DOFs by node
    node_reactions = {}
    for dof in St.dof_fix:
        # Find which node this DOF belongs to
        for node_id in range(len(St.list_nodes)):
            dofs = St.get_dofs_from_node(node_id)
            dofs_list = list(dofs) if hasattr(dofs, '__iter__') else [dofs]
            if dof in dofs_list:
                if node_id not in node_reactions:
                    node_reactions[node_id] = {'Rx': 0.0, 'Ry': 0.0, 'Mz': 0.0}
                local_dof = dofs_list.index(dof)
                if local_dof == 0:
                    node_reactions[node_id]['Rx'] = St.P[dof]
                elif local_dof == 1:
                    node_reactions[node_id]['Ry'] = St.P[dof]
                elif local_dof == 2:
                    node_reactions[node_id]['Mz'] = St.P[dof]
                break

    # Build CSV
    if has_blocks:
        lines = ['node_id,x,y,Rx,Ry,Mz']
    else:
        lines = ['node_id,x,y,Rx,Ry']

    for node_id, reactions in sorted(node_reactions.items()):
        node = St.list_nodes[node_id]
        if has_blocks:
            lines.append(f'{node_id},{node[0]:.12e},{node[1]:.12e},'
                        f'{reactions["Rx"]:.12e},{reactions["Ry"]:.12e},{reactions["Mz"]:.12e}')
        else:
            lines.append(f'{node_id},{node[0]:.12e},{node[1]:.12e},'
                        f'{reactions["Rx"]:.12e},{reactions["Ry"]:.12e}')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Exported reactions to: {filepath}")
    return filepath


def export_elements(St, config, filename: Optional[str] = None) -> str:
    """
    Export element connectivity to CSV.

    Columns: element_id, type, node_ids (comma-separated), centroid_x, centroid_y

    Args:
        St: Structure object
        config: Configuration dictionary with 'io' section
        filename: Optional custom filename (without extension)

    Returns:
        Path to saved file
    """
    io = config['io']
    os.makedirs(io['dir'], exist_ok=True)

    if filename is None:
        filename = io['filename'] + '_elements'

    filepath = os.path.join(io['dir'], filename + '.csv')

    lines = ['element_id,type,n_nodes,centroid_x,centroid_y,node_ids']

    # Export FEM elements
    if hasattr(St, 'list_fes') and St.list_fes:
        for elem_id, fe in enumerate(St.list_fes):
            elem_type = fe.__class__.__name__
            nodes = np.array(fe.nodes)
            centroid = nodes.mean(axis=0)
            n_nodes = len(fe.nodes)

            # Find node IDs
            node_ids = []
            for node_coords in fe.nodes:
                for i, n in enumerate(St.list_nodes):
                    if abs(n[0] - node_coords[0]) < 1e-10 and abs(n[1] - node_coords[1]) < 1e-10:
                        node_ids.append(str(i))
                        break

            lines.append(f'{elem_id},{elem_type},{n_nodes},{centroid[0]:.12e},{centroid[1]:.12e},"{";".join(node_ids)}"')

    # Export blocks
    if hasattr(St, 'list_blocks') and St.list_blocks:
        for block_id, block in enumerate(St.list_blocks):
            elem_type = 'Block_2D'
            if hasattr(block, 'vertices'):
                vertices = np.array(block.vertices)
                centroid = vertices.mean(axis=0)
                n_nodes = 1  # Block has one node (ref point)

                # Find the block's node ID
                ref = block.ref_point if hasattr(block, 'ref_point') else centroid
                node_id = -1
                for i, n in enumerate(St.list_nodes):
                    if abs(n[0] - ref[0]) < 1e-10 and abs(n[1] - ref[1]) < 1e-10:
                        node_id = i
                        break

                lines.append(f'{block_id},{elem_type},{n_nodes},{centroid[0]:.12e},{centroid[1]:.12e},"{node_id}"')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Exported elements to: {filepath}")
    return filepath


def export_state(St, config, prefix: Optional[str] = None) -> Dict[str, str]:
    """
    Export complete structure state to multiple CSV files.

    Creates:
    - {prefix}_nodes.csv - Node positions
    - {prefix}_displacements.csv - Displacement field
    - {prefix}_stresses.csv - Stress field (FEM only)
    - {prefix}_reactions.csv - Reaction forces
    - {prefix}_elements.csv - Element connectivity

    Args:
        St: Solved structure object
        config: Configuration dictionary with 'io' section
        prefix: Optional prefix for filenames (defaults to config['io']['filename'])

    Returns:
        Dictionary mapping export type to file path
    """
    io = config['io']
    if prefix is None:
        prefix = io['filename']

    print(f"\nExporting structure state: {prefix}")

    paths = {}
    paths['nodes'] = export_nodes(St, config, prefix + '_nodes')
    paths['displacements'] = export_displacements(St, config, prefix + '_displacements')
    paths['elements'] = export_elements(St, config, prefix + '_elements')

    # Stresses only for FEM/Hybrid
    if hasattr(St, 'list_fes') and St.list_fes:
        paths['stresses'] = export_stresses(St, config, prefix + '_stresses')

    # Reactions if fixed DOFs exist
    if hasattr(St, 'dof_fix') and len(St.dof_fix) > 0:
        paths['reactions'] = export_reactions(St, config, prefix + '_reactions')

    print(f"  Export complete: {len(paths)} files")
    return paths


def load_displacements(filepath: str):
    """
    Load displacements from CSV file.

    Returns numpy arrays or can be used with pandas:
        import pandas as pd
        df = pd.read_csv(filepath)

    Args:
        filepath: Path to CSV file

    Returns:
        Dict with arrays: node_ids, x, y, ux, uy, [rz], u_mag
    """
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    return {name: data[name] for name in data.dtype.names}


def load_stresses(filepath: str):
    """
    Load stresses from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        Dict with arrays: element_id, centroid_x, centroid_y, sigma_xx, sigma_yy, sigma_xy, von_mises
    """
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    return {name: data[name] for name in data.dtype.names}


# =============================================================================
# LINEAR SYSTEM EXPORT (K, P, U)
# =============================================================================

def export_stiffness_matrix(St, config, filename: Optional[str] = None,
                            format: str = 'coo') -> str:
    """
    Export stiffness matrix K to file.

    Formats:
    - 'coo': Sparse COO format (row, col, value) - efficient for large matrices
    - 'dense': Full dense matrix as CSV - only for small systems
    - 'npz': NumPy compressed sparse format - fastest to load back

    Args:
        St: Structure object with assembled K matrix
        config: Configuration dictionary with 'io' section
        filename: Optional custom filename (without extension)
        format: Export format ('coo', 'dense', 'npz')

    Returns:
        Path to saved file
    """
    io = config['io']
    os.makedirs(io['dir'], exist_ok=True)

    if filename is None:
        filename = io['filename'] + '_K'

    # Get stiffness matrix (check K0 first, then K)
    K = None
    if hasattr(St, 'K0') and St.K0 is not None:
        K = St.K0
    elif hasattr(St, 'K') and St.K is not None:
        K = St.K

    if K is None:
        print("  No stiffness matrix found (K or K0), skipping K export")
        return ""

    if format == 'npz':
        # Save as compressed sparse
        from scipy import sparse
        filepath = os.path.join(io['dir'], filename + '.npz')
        if sparse.issparse(K):
            sparse.save_npz(filepath, K.tocsr())
        else:
            sparse.save_npz(filepath, sparse.csr_matrix(K))
        print(f"  Exported stiffness matrix to: {filepath}")
        return filepath

    elif format == 'dense':
        # Full dense matrix - warning for large systems
        filepath = os.path.join(io['dir'], filename + '.csv')
        if hasattr(K, 'toarray'):
            K_dense = K.toarray()
        else:
            K_dense = np.array(K)

        n_dof = K_dense.shape[0]
        if n_dof > 500:
            print(f"  Warning: Dense export of {n_dof}x{n_dof} matrix may be large")

        # Write with header showing DOF indices
        header = ','.join([f'dof_{i}' for i in range(n_dof)])
        np.savetxt(filepath, K_dense, delimiter=',', header=header, comments='')
        print(f"  Exported stiffness matrix (dense) to: {filepath}")
        return filepath

    else:  # 'coo' format - default
        filepath = os.path.join(io['dir'], filename + '.csv')

        # Convert to COO format
        from scipy import sparse
        if sparse.issparse(K):
            K_coo = K.tocoo()
        else:
            K_coo = sparse.coo_matrix(K)

        # Write COO format: row, col, value
        lines = ['row,col,value']
        for i, j, v in zip(K_coo.row, K_coo.col, K_coo.data):
            if abs(v) > 1e-16:  # Skip near-zero entries
                lines.append(f'{i},{j},{v:.15e}')

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        n_entries = len(lines) - 1
        n_dof = K_coo.shape[0]
        sparsity = 1.0 - n_entries / (n_dof * n_dof) if n_dof > 0 else 0
        print(f"  Exported stiffness matrix to: {filepath}")
        print(f"    Size: {n_dof}x{n_dof}, Non-zeros: {n_entries}, Sparsity: {sparsity:.1%}")
        return filepath


def export_load_vector(St, config, filename: Optional[str] = None) -> str:
    """
    Export load vector P to CSV.

    Columns: dof_id, force, [node_id, local_dof]

    Args:
        St: Structure object with load vector P
        config: Configuration dictionary with 'io' section
        filename: Optional custom filename (without extension)

    Returns:
        Path to saved file
    """
    io = config['io']
    os.makedirs(io['dir'], exist_ok=True)

    if filename is None:
        filename = io['filename'] + '_P'

    filepath = os.path.join(io['dir'], filename + '.csv')

    # Get load vector
    if not hasattr(St, 'P') or St.P is None:
        print("  No load vector found, skipping P export")
        return ""

    P = St.P

    # Build data with DOF-to-node mapping
    lines = ['dof_id,force,node_id,local_dof']

    # Build DOF to node mapping
    dof_to_node = {}
    for node_id in range(len(St.list_nodes)):
        dofs = St.get_dofs_from_node(node_id)
        dofs_list = list(dofs) if hasattr(dofs, '__iter__') else [dofs]
        for local_idx, global_dof in enumerate(dofs_list):
            dof_to_node[global_dof] = (node_id, local_idx)

    for dof_id, force in enumerate(P):
        node_id, local_dof = dof_to_node.get(dof_id, (-1, -1))
        lines.append(f'{dof_id},{force:.15e},{node_id},{local_dof}')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    n_nonzero = np.count_nonzero(P)
    print(f"  Exported load vector to: {filepath}")
    print(f"    DOFs: {len(P)}, Non-zero loads: {n_nonzero}")
    return filepath


def export_displacement_vector(St, config, filename: Optional[str] = None) -> str:
    """
    Export displacement vector U to CSV (raw DOF values).

    Columns: dof_id, displacement, node_id, local_dof

    Args:
        St: Structure object with displacement vector U
        config: Configuration dictionary with 'io' section
        filename: Optional custom filename (without extension)

    Returns:
        Path to saved file
    """
    io = config['io']
    os.makedirs(io['dir'], exist_ok=True)

    if filename is None:
        filename = io['filename'] + '_U'

    filepath = os.path.join(io['dir'], filename + '.csv')

    # Get displacement vector
    if not hasattr(St, 'U') or St.U is None:
        print("  No displacement vector found, skipping U export")
        return ""

    U = St.U

    # Expand if constraint coupling
    if hasattr(St, 'coupling_T') and St.coupling_T is not None:
        U = St.coupling_T @ St.U

    # Build DOF to node mapping
    dof_to_node = {}
    for node_id in range(len(St.list_nodes)):
        dofs = St.get_dofs_from_node(node_id)
        dofs_list = list(dofs) if hasattr(dofs, '__iter__') else [dofs]
        for local_idx, global_dof in enumerate(dofs_list):
            dof_to_node[global_dof] = (node_id, local_idx)

    lines = ['dof_id,displacement,node_id,local_dof']
    for dof_id, disp in enumerate(U):
        node_id, local_dof = dof_to_node.get(dof_id, (-1, -1))
        lines.append(f'{dof_id},{disp:.15e},{node_id},{local_dof}')

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))

    print(f"  Exported displacement vector to: {filepath}")
    return filepath


def export_system(St, config, prefix: Optional[str] = None,
                  k_format: str = 'coo') -> Dict[str, str]:
    """
    Export the complete linear system Ku = P.

    Creates:
    - {prefix}_K.csv (or .npz) - Stiffness matrix
    - {prefix}_P.csv - Load vector
    - {prefix}_U.csv - Displacement vector

    Args:
        St: Solved structure object
        config: Configuration dictionary with 'io' section
        prefix: Optional prefix for filenames
        k_format: Format for K matrix ('coo', 'dense', 'npz')

    Returns:
        Dictionary mapping export type to file path
    """
    io = config['io']
    if prefix is None:
        prefix = io['filename']

    print(f"\nExporting linear system: {prefix}")

    paths = {}
    paths['K'] = export_stiffness_matrix(St, config, prefix + '_K', format=k_format)
    paths['P'] = export_load_vector(St, config, prefix + '_P')
    paths['U'] = export_displacement_vector(St, config, prefix + '_U')

    print(f"  System export complete: {len([p for p in paths.values() if p])} files")
    return paths


def load_stiffness_matrix(filepath: str):
    """
    Load stiffness matrix from file.

    Supports:
    - .npz: NumPy compressed sparse format
    - .csv: COO format (row, col, value) or dense

    Args:
        filepath: Path to file

    Returns:
        scipy.sparse matrix or numpy array
    """
    from scipy import sparse

    if filepath.endswith('.npz'):
        return sparse.load_npz(filepath)

    # CSV format - check if COO or dense
    with open(filepath, 'r') as f:
        header = f.readline().strip()

    if header.startswith('row,col,value'):
        # COO format
        data = np.genfromtxt(filepath, delimiter=',', names=True)
        rows = data['row'].astype(int)
        cols = data['col'].astype(int)
        vals = data['value']
        n = max(rows.max(), cols.max()) + 1
        return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    else:
        # Dense format
        return np.loadtxt(filepath, delimiter=',', skiprows=1)


def load_load_vector(filepath: str):
    """
    Load load vector from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        numpy array of forces
    """
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    return data['force']


def load_system(directory: str, prefix: str) -> Dict[str, Any]:
    """
    Load complete linear system from files.

    Args:
        directory: Directory containing export files
        prefix: File prefix used during export

    Returns:
        Dict with 'K', 'P', 'U' arrays
    """
    system = {}

    # Try npz first, then csv for K
    k_npz = os.path.join(directory, prefix + '_K.npz')
    k_csv = os.path.join(directory, prefix + '_K.csv')
    if os.path.exists(k_npz):
        system['K'] = load_stiffness_matrix(k_npz)
    elif os.path.exists(k_csv):
        system['K'] = load_stiffness_matrix(k_csv)

    p_path = os.path.join(directory, prefix + '_P.csv')
    if os.path.exists(p_path):
        system['P'] = load_load_vector(p_path)

    u_path = os.path.join(directory, prefix + '_U.csv')
    if os.path.exists(u_path):
        data = np.genfromtxt(u_path, delimiter=',', names=True)
        system['U'] = data['displacement']

    return system
