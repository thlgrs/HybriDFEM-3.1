"""
VTK export utilities for HybriDFEM results.

This module provides functions for exporting structure geometry and results
to VTK format for visualization in ParaView, VisIt, or other VTK-compatible software.
"""

import numpy as np


class VTKExporter:
    """
    Export HybridFEM structures and results to VTK format.

    Supports:
    - Unstructured grids for FEM elements (triangles, quads)
    - Point data for block nodes
    - Cell data for stresses/strains
    - Vector fields for displacements
    """

    def __init__(self, structure):
        """
        Initialize VTK exporter.

        Parameters
        ----------
        structure : Structure_2D
            Structure to export (Structure_FEM, Structure_Block, or Hybrid)
        """
        self.structure = structure

    def _get_node_displacements(self, node_id):
        """
        Get displacements for a node (variable DOF compatible).

        Returns
        -------
        tuple
            (ux, uy, rz) - rz is 0.0 if node only has 2 DOFs
        """
        if self.structure.U is None:
            return (0.0, 0.0, 0.0)

        # Check if structure uses variable DOF system
        if hasattr(self.structure, 'node_dof_offsets') and hasattr(self.structure, 'node_dof_counts'):
            base_dof = self.structure.node_dof_offsets[node_id]
            dof_count = self.structure.node_dof_counts[node_id]

            ux = self.structure.U[base_dof + 0]
            uy = self.structure.U[base_dof + 1]
            rz = self.structure.U[base_dof + 2] if dof_count == 3 else 0.0

            return (ux, uy, rz)
        else:
            # Legacy 3-DOF system (fallback)
            ux = self.structure.U[3 * node_id]
            uy = self.structure.U[3 * node_id + 1]
            rz = self.structure.U[3 * node_id + 2]
            return (ux, uy, rz)

    def export_to_vtk(self, file_path: str, include_displacements: bool = True,
                     deformation_scale: float = 1.0):
        """
        Export structure to VTK legacy format (.vtk file).

        This format is widely supported and human-readable (ASCII).

        Parameters
        ----------
        file_path : str
            Output file path (.vtk extension)
        include_displacements : bool
            If True, include displacement vectors as point data
        deformation_scale : float
            Scale factor for deformed geometry (default: 1.0 = actual displacements)
        """
        file_path = str(file_path)
        if not file_path.endswith('.vtk'):
            file_path += '.vtk'

        with open(file_path, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"HybridFEM Analysis Results: {type(self.structure).__name__}\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n\n")

            # Points (nodes)
            num_nodes = len(self.structure.list_nodes)
            f.write(f"POINTS {num_nodes} float\n")

            for i, node in enumerate(self.structure.list_nodes):
                # Original position + scaled displacements
                if include_displacements and self.structure.U is not None:
                    ux, uy, _ = self._get_node_displacements(i)
                    x = node[0] + ux * deformation_scale
                    y = node[1] + uy * deformation_scale
                else:
                    x, y = node[0], node[1]

                f.write(f"{x:.6e} {y:.6e} 0.0\n")

            # Cells (FEM elements)
            if hasattr(self.structure, 'list_fes') and len(self.structure.list_fes) > 0:
                num_cells = len(self.structure.list_fes)

                # Count total connectivity size
                # Format: n_nodes node1 node2 ... (so n_nodes+1 integers per cell)
                total_size = sum(fe.nd + 1 for fe in self.structure.list_fes)

                f.write(f"\nCELLS {num_cells} {total_size}\n")

                for fe in self.structure.list_fes:
                    # VTK cell format: n_nodes node_id1 node_id2 ...
                    connectivity = fe.connect  # Global node indices
                    f.write(f"{fe.nd} ")
                    f.write(" ".join(str(idx) for idx in connectivity))
                    f.write("\n")

                # Cell types
                f.write(f"\nCELL_TYPES {num_cells}\n")
                for fe in self.structure.list_fes:
                    # VTK cell type codes:
                    # 5 = VTK_TRIANGLE (3 nodes)
                    # 22 = VTK_QUADRATIC_TRIANGLE (6 nodes)
                    # 9 = VTK_QUAD (4 nodes)
                    # 23 = VTK_QUADRATIC_QUAD (8 nodes)
                    if fe.nd == 3:
                        cell_type = 5  # Linear triangle
                    elif fe.nd == 6:
                        cell_type = 22  # Quadratic triangle
                    elif fe.nd == 4:
                        cell_type = 9  # Linear quad
                    elif fe.nd == 8:
                        cell_type = 23  # Quadratic quad
                    else:
                        cell_type = 1  # VTK_VERTEX (fallback)

                    f.write(f"{cell_type}\n")

            # Point data (nodal values)
            if include_displacements and self.structure.U is not None:
                f.write(f"\nPOINT_DATA {num_nodes}\n")

                # Displacement vectors
                f.write("VECTORS displacement float\n")
                for i in range(num_nodes):
                    ux, uy, rz = self._get_node_displacements(i)
                    f.write(f"{ux:.6e} {uy:.6e} 0.0\n")

                # Displacement magnitude
                f.write("\nSCALARS displacement_magnitude float 1\n")
                f.write("LOOKUP_TABLE default\n")
                for i in range(num_nodes):
                    ux, uy, _ = self._get_node_displacements(i)
                    mag = np.sqrt(ux**2 + uy**2)
                    f.write(f"{mag:.6e}\n")

    def export_to_vtu(self, file_path: str, include_displacements: bool = True,
                     deformation_scale: float = 1.0):
        """
        Export structure to VTK XML format (.vtu file).

        This is the modern VTK format, binary-compatible and efficient.
        Requires pyevtk library.

        Parameters
        ----------
        file_path : str
            Output file path (.vtu extension)
        include_displacements : bool
            If True, include displacement vectors as point data
        deformation_scale : float
            Scale factor for deformed geometry
        """
        try:
            from pyevtk.hl import unstructuredGridToVTK
        except ImportError:
            raise ImportError(
                "pyevtk library not found. Install with: pip install pyevtk"
            )

        # Prepare node coordinates
        num_nodes = len(self.structure.list_nodes)
        x = np.zeros(num_nodes)
        y = np.zeros(num_nodes)
        z = np.zeros(num_nodes)

        for i, node in enumerate(self.structure.list_nodes):
            if include_displacements and self.structure.U is not None:
                ux, uy, _ = self._get_node_displacements(i)
                x[i] = node[0] + ux * deformation_scale
                y[i] = node[1] + uy * deformation_scale
            else:
                x[i] = node[0]
                y[i] = node[1]
            z[i] = 0.0

        # Prepare connectivity and cell types
        if hasattr(self.structure, 'list_fes') and len(self.structure.list_fes) > 0:
            connectivity = []
            offsets = []
            cell_types = []

            current_offset = 0
            for fe in self.structure.list_fes:
                # Add connectivity
                connectivity.extend(fe.connect)
                current_offset += fe.nd
                offsets.append(current_offset)

                # Add cell type
                if fe.nd == 3:
                    cell_types.append(5)  # Linear triangle
                elif fe.nd == 6:
                    cell_types.append(22)  # Quadratic triangle
                elif fe.nd == 4:
                    cell_types.append(9)  # Linear quad
                elif fe.nd == 8:
                    cell_types.append(23)  # Quadratic quad
                else:
                    cell_types.append(1)  # Vertex

            connectivity = np.array(connectivity, dtype=np.int32)
            offsets = np.array(offsets, dtype=np.int32)
            cell_types = np.array(cell_types, dtype=np.uint8)
        else:
            # No FEM elements - create dummy connectivity
            connectivity = np.array([], dtype=np.int32)
            offsets = np.array([], dtype=np.int32)
            cell_types = np.array([], dtype=np.uint8)

        # Prepare point data
        pointData = {}
        if include_displacements and self.structure.U is not None:
            ux_arr = np.zeros(num_nodes)
            uy_arr = np.zeros(num_nodes)
            uz_arr = np.zeros(num_nodes)

            for i in range(num_nodes):
                ux, uy, rz = self._get_node_displacements(i)
                ux_arr[i] = ux
                uy_arr[i] = uy
                uz_arr[i] = 0.0

            pointData['displacement'] = (ux_arr, uy_arr, uz_arr)
            pointData['displacement_magnitude'] = np.sqrt(ux_arr**2 + uy_arr**2)

        # Remove .vtu extension if present (pyevtk adds it)
        file_path = str(file_path)
        if file_path.endswith('.vtu'):
            file_path = file_path[:-4]

        # Export
        unstructuredGridToVTK(
            file_path,
            x, y, z,
            connectivity=connectivity,
            offsets=offsets,
            cell_types=cell_types,
            pointData=pointData
        )


def export_vtk(structure, file_path: str, format: str = 'vtk',
               include_displacements: bool = True, deformation_scale: float = 1.0):
    """
    Convenience function to export structure to VTK format.

    Parameters
    ----------
    structure : Structure_2D
        Structure to export
    file_path : str
        Output file path
    format : str
        'vtk' for legacy ASCII format (always works)
        'vtu' for XML format (requires pyevtk)
    include_displacements : bool
        If True, include displacement field
    deformation_scale : float
        Scale factor for deformed geometry

    Returns
    -------
    str
        Path to created file

    Examples
    --------
    >>> # Export to legacy VTK (always works)
    >>> export_vtk(structure, 'results.vtk', format='vtk')

    >>> # Export to XML VTK (requires pyevtk)
    >>> export_vtk(structure, 'results.vtu', format='vtu')
    """
    exporter = VTKExporter(structure)

    if format.lower() == 'vtu':
        exporter.export_to_vtu(file_path, include_displacements, deformation_scale)
        if not file_path.endswith('.vtu'):
            file_path += '.vtu'
    else:
        exporter.export_to_vtk(file_path, include_displacements, deformation_scale)
        if not file_path.endswith('.vtk'):
            file_path += '.vtk'

    return file_path
