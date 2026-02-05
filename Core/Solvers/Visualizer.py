import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize


# =============================================================================
# Helper functions for higher-order element plotting
# =============================================================================

def _get_element_edges(element) -> List[List[int]]:
    """Return edge definitions for an element as list of node index lists."""
    element_type = type(element).__name__

    if element_type == 'Triangle3':
        return [[0, 1], [1, 2], [2, 0]]
    elif element_type == 'Triangle6':
        return [[0, 3, 1], [1, 4, 2], [2, 5, 0]]
    elif element_type == 'Quad4':
        return [[0, 1], [1, 2], [2, 3], [3, 0]]
    elif element_type in ['Quad8', 'Quad9']:
        return [[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]]
    else:
        nd = element.nd
        return [[i, (i + 1) % nd] for i in range(nd)]


def _interpolate_quadratic_edge(p0: np.ndarray, p_mid: np.ndarray, p1: np.ndarray,
                                n_points: int = 10) -> np.ndarray:
    """Interpolate a quadratic curve through three points."""
    t = np.linspace(0, 1, n_points)
    # Vectorized calculation
    L0 = (1 - t) * (1 - 2 * t)
    L1 = 4 * t * (1 - t)
    L2 = t * (2 * t - 1)

    # Broadcasting shapes for efficient calc
    x = np.outer(L0, p0[0]) + np.outer(L1, p_mid[0]) + np.outer(L2, p1[0])
    y = np.outer(L0, p0[1]) + np.outer(L1, p_mid[1]) + np.outer(L2, p1[1])

    return np.hstack([x, y])


def _get_element_boundary_coords(coords: np.ndarray, element, n_interp: int = 10) -> np.ndarray:
    """Get boundary coordinates for plotting an element."""
    edges = _get_element_edges(element)
    boundary_points = []

    for edge in edges:
        if len(edge) == 2:
            boundary_points.append(coords[edge[0]])
        elif len(edge) == 3:
            p0 = coords[edge[0]]
            p_mid = coords[edge[1]]
            p1 = coords[edge[2]]
            # Exclude last point to avoid duplication
            interp_pts = _interpolate_quadratic_edge(p0, p_mid, p1, n_interp)
            boundary_points.append(interp_pts[:-1])

    # Efficient flattening of the list of arrays
    if len(boundary_points) > 0:
        if len(boundary_points[0].shape) == 1:  # Single points
            boundary = np.array(boundary_points)
        else:  # Arrays of points
            boundary = np.vstack(boundary_points)
        return np.vstack([boundary, boundary[0]])  # Close polygon
    return coords


@dataclass
class PlotStyle:
    """Configuration for publication-ready structure plots."""
    # --- FEM (Line) Styling ---
    fem_linewidth: float = 0.8
    fem_color_orig: str = '#555555'
    fem_linestyle_orig: str = '--'
    fem_alpha_orig: float = 0.6

    fem_color_def: str = '#1f77b4'
    fem_linestyle_def: str = '-'
    fem_alpha_def: float = 1.0

    # --- Block (Polygon) Styling ---
    block_linewidth: float = 0.8
    block_edge_color: str = 'black'
    block_line_alpha_orig: float = 0.6

    block_fill_orig: str = 'white'
    block_linestyle_orig: str = '-'
    block_alpha_orig: float = 0.0

    block_fill_def: str = '#a6cee3'
    block_linestyle_def: str = '-'
    block_alpha_def: float = 0.7

    # --- Node Styling ---
    node_size: float = 2
    node_color_orig: str = '#555555'
    node_color_def: str = '#1f77b4'
    node_marker: str = 'o'
    node_alpha: float = 0.8

    # --- Figure Layout ---
    figsize: Tuple[float, float] = (8, 8)
    dpi: int = 600
    background_color: str = 'white'

    # --- Axes & Fonts ---
    axis_linewidth: float = 0.8
    tick_labelsize: int = 10
    label_fontsize: int = 12
    title_fontsize: int = 14
    grid: bool = True
    grid_alpha: float = 0.70
    font_family: str = 'serif'
    use_latex: bool = True

    def apply_global_config(self):
        if self.use_latex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "axes.unicode_minus": False,
            })
        else:
            plt.rcParams.update({"text.usetex": False})
            plt.rc('font', family=self.font_family)

    def apply_to_figure(self, fig):
        fig.patch.set_facecolor(self.background_color)

    def apply_to_axes(self, ax):
        for spine in ax.spines.values():
            spine.set_linewidth(self.axis_linewidth)
        ax.tick_params(labelsize=self.tick_labelsize)
        ax.set_facecolor(self.background_color)
        if self.grid:
            ax.grid(True, alpha=self.grid_alpha, linestyle='--', linewidth=0.5)

    @classmethod
    def default(cls):
        return cls()

    @classmethod
    def scientific(cls):
        return cls(
            fem_color_orig='black', fem_linestyle_orig='--', fem_alpha_orig=0.6,
            fem_color_def='black', fem_linestyle_def='-', fem_alpha_def=1.0,
            fem_linewidth=1.0, block_edge_color='black', block_line_alpha_orig=0.6,
            block_fill_orig='white', block_linestyle_orig=':', block_alpha_orig=1.0,
            block_fill_def='lightgray', block_linestyle_def='-', block_alpha_def=1.0,
            figsize=(7, 5), grid=False
        )

    @classmethod
    def presentation(cls):
        return cls(
            fem_linewidth=2.5, block_linewidth=2.5,
            fem_color_orig='gray', fem_linestyle_orig=':', fem_alpha_orig=0.6,
            fem_color_def='#d62728', fem_linestyle_def='-',
            block_edge_color='black', block_line_alpha_orig=0.6,
            block_fill_orig='white', block_fill_def='#17becf', block_alpha_def=0.5,
            label_fontsize=16, tick_labelsize=14, title_fontsize=18,
            figsize=(10, 7), grid=True, grid_alpha=0.3, use_latex=True
        )


class Visualizer:

    @staticmethod
    def _detect_structure_type(structure) -> str:
        has_blocks = hasattr(structure, 'list_blocks') and len(structure.list_blocks) > 0
        has_fes = hasattr(structure, 'list_fes') and len(structure.list_fes) > 0
        if has_blocks and has_fes:
            return 'hybrid'
        elif has_blocks:
            return 'block'
        elif has_fes:
            return 'fem'
        return 'unknown'

    @staticmethod
    def _build_node_map(structure) -> dict:
        """Builds a map of (x, y) -> node_index for fast lookup."""
        # Precision rounding to avoid float errors
        return {(round(n[0], 9), round(n[1], 9)): idx
                for idx, n in enumerate(structure.list_nodes)}

    @staticmethod
    def _detect_orientation(structure) -> str:
        """
        Detect structure orientation based on bounding box.

        Returns
        -------
        str
            'horizontal' if width >= height, 'vertical' otherwise.
        """
        all_x = []
        all_y = []

        # Collect coordinates from FEM elements
        if hasattr(structure, 'list_fes') and structure.list_fes:
            for fe in structure.list_fes:
                for node in fe.nodes:
                    all_x.append(node[0])
                    all_y.append(node[1])

        # Collect coordinates from blocks
        if hasattr(structure, 'list_blocks') and structure.list_blocks:
            for block in structure.list_blocks:
                for vertex in block.v:
                    all_x.append(vertex[0])
                    all_y.append(vertex[1])

        if not all_x or not all_y:
            return 'horizontal'  # Default fallback

        width = max(all_x) - min(all_x)
        height = max(all_y) - min(all_y)

        return 'horizontal' if width >= height else 'vertical'

    @staticmethod
    def _transform_points_rigid(points: np.ndarray, block, structure, scale: float = 1.0) -> np.ndarray:
        """Transforms points based on rigid body motion of the block."""
        if block is None:
            return points

        ref_x, ref_y = block.ref_point

        node_idx = block.connect
        dof_ux = structure._global_dof(node_idx, 0)
        dof_uy = structure._global_dof(node_idx, 1)
        dof_rot = structure._global_dof(node_idx, 2)

        ux = structure.U[dof_ux] * scale
        uy = structure.U[dof_uy] * scale
        theta = structure.U[dof_rot] * scale

        # Rigid body transformation
        dx = points[:, 0] - ref_x
        dy = points[:, 1] - ref_y

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        new_x = ref_x + ux + (dx * cos_t - dy * sin_t)
        new_y = ref_y + uy + (dx * sin_t + dy * cos_t)

        return np.column_stack([new_x, new_y])

    @staticmethod
    def _compute_block_deformed_poly(block, structure, scale=1.0):
        """Helper to compute deformed block polygon vertices."""
        orig_vertices = np.array(block.v)
        def_vertices = Visualizer._transform_points_rigid(orig_vertices, block, structure, scale)
        # Close polygon
        return np.vstack([def_vertices, def_vertices[0]])

    @staticmethod
    def _plot_blocks(ax, structure, style: PlotStyle, scale=1.0,
                     show_original=True, show_deformed=True, show_nodes=False):
        """Optimized block plotter using PolyCollections."""
        if not hasattr(structure, 'list_blocks'):
            return

        # Pre-allocate lists for collections
        orig_verts_list = []
        def_verts_list = []

        orig_nodes_x, orig_nodes_y = [], []
        def_nodes_x, def_nodes_y = [], []

        # Optimization: Process blocks
        for block in structure.list_blocks:
            # 1. Geometry Prep
            ref_x, ref_y = block.ref_point
            orig_vertices = np.array(block.v)

            # 2. Original Config
            if show_original:
                # Close polygon for plotting
                v_closed = np.vstack([orig_vertices, orig_vertices[0]])
                orig_verts_list.append(v_closed)
                if show_nodes:
                    orig_nodes_x.append(ref_x)
                    orig_nodes_y.append(ref_y)

            # 3. Deformed Config (Vectorized Math)
            if show_deformed:
                v_def_closed = Visualizer._compute_block_deformed_poly(block, structure, scale)
                def_verts_list.append(v_def_closed)

                if show_nodes:
                    # Re-calculate just for node (could optimize but this is minor)
                    node_idx = block.connect
                    dof_ux = structure._global_dof(node_idx, 0)
                    dof_uy = structure._global_dof(node_idx, 1)
                    ux = structure.U[dof_ux] * scale
                    uy = structure.U[dof_uy] * scale
                    def_nodes_x.append(ref_x + ux)
                    def_nodes_y.append(ref_y + uy)

        # Optimization: Use Collections instead of loop plotting
        if show_original and orig_verts_list:
            # Filled polygons
            pc_fill = PolyCollection(orig_verts_list,
                                     facecolors=style.block_fill_orig,
                                     edgecolors='none',  # Edges handled separately
                                     alpha=style.block_alpha_orig)
            ax.add_collection(pc_fill)

            # Edges (Lines)
            pc_edge = PolyCollection(orig_verts_list,
                                     facecolors='none',
                                     edgecolors='black',
                                     linestyles=style.block_linestyle_orig,
                                     linewidths=style.block_linewidth,
                                     alpha=style.block_line_alpha_orig)
            ax.add_collection(pc_edge)

        if show_deformed and def_verts_list:
            pc_fill = PolyCollection(def_verts_list,
                                     facecolors=style.block_fill_def,
                                     edgecolors='none',
                                     alpha=style.block_alpha_def)
            ax.add_collection(pc_fill)

            pc_edge = PolyCollection(def_verts_list,
                                     facecolors='none',
                                     edgecolors=style.block_edge_color,
                                     linestyles=style.block_linestyle_def,
                                     linewidths=style.block_linewidth)
            ax.add_collection(pc_edge)

        # Batch Plot Nodes
        if show_nodes:
            if show_original and orig_nodes_x:
                ax.scatter(orig_nodes_x, orig_nodes_y,
                           s=style.node_size, c=style.node_color_orig,
                           marker='s', alpha=style.node_alpha, zorder=10)
            if show_deformed and def_nodes_x:
                ax.scatter(def_nodes_x, def_nodes_y,
                           s=style.node_size, c=style.node_color_def,
                           marker='s', alpha=style.node_alpha, zorder=10)

    @staticmethod
    def _plot_fem_elements(ax, structure, style: PlotStyle, scale=1.0,
                           show_original=True, show_deformed=True, show_nodes=False):
        """Optimized FEM plotter using LineCollections and spatial hashing."""
        if not hasattr(structure, 'list_fes'):
            return

        # Optimization 1: Build a Coordinate -> DOF Map ONCE (O(N) instead of O(N^2))
        node_map = {}
        if show_deformed:
            node_map = Visualizer._build_node_map(structure)

        orig_segments = []
        def_segments = []

        # Use Sets to collect unique node coords for scatter plot
        unique_orig_nodes = set()
        unique_def_nodes = set()

        for fe in structure.list_fes:
            orig_coords = np.array(fe.nodes)

            # --- Original Configuration ---
            if show_original:
                orig_boundary = _get_element_boundary_coords(orig_coords, fe)
                orig_segments.append(orig_boundary)

                if show_nodes:
                    for coord in orig_coords:
                        unique_orig_nodes.add(tuple(coord))

            # --- Deformed Configuration ---
            if show_deformed:
                def_coords = orig_coords.copy()
                # Fast lookup
                for j, node_coord in enumerate(fe.nodes):
                    key = (round(node_coord[0], 9), round(node_coord[1], 9))
                    if key in node_map:
                        idx = node_map[key]
                        dof_x = structure.node_dof_offsets[idx]
                        dof_y = structure.node_dof_offsets[idx] + 1
                        def_coords[j, 0] += scale * structure.U[dof_x]
                        def_coords[j, 1] += scale * structure.U[dof_y]

                def_boundary = _get_element_boundary_coords(def_coords, fe)
                def_segments.append(def_boundary)

                if show_nodes:
                    for coord in def_coords:
                        unique_def_nodes.add(tuple(coord))

        # Optimization 2: Matplotlib Collections
        if show_original and orig_segments:
            lc_orig = LineCollection(orig_segments,
                                     colors=style.fem_color_orig,
                                     linestyles=style.fem_linestyle_orig,
                                     linewidths=style.fem_linewidth,
                                     alpha=style.fem_alpha_orig)
            ax.add_collection(lc_orig)

        if show_deformed and def_segments:
            lc_def = LineCollection(def_segments,
                                    colors=style.fem_color_def,
                                    linestyles=style.fem_linestyle_def,
                                    linewidths=style.fem_linewidth,
                                    alpha=style.fem_alpha_def)
            ax.add_collection(lc_def)

        # Batch Plot Nodes
        if show_nodes:
            if show_original and unique_orig_nodes:
                pts = np.array(list(unique_orig_nodes))
                ax.scatter(pts[:, 0], pts[:, 1],
                           s=style.node_size, c=style.node_color_orig,
                           marker=style.node_marker, alpha=style.node_alpha, zorder=10)

            if show_deformed and unique_def_nodes:
                pts = np.array(list(unique_def_nodes))
                ax.scatter(pts[:, 0], pts[:, 1],
                           s=style.node_size, c=style.node_color_def,
                           marker=style.node_marker, alpha=style.node_alpha, zorder=10)

    @staticmethod
    def _setup_figure(figsize, style):
        if style is None: style = PlotStyle()
        style.apply_global_config()
        size = figsize if figsize else style.figsize
        fig, ax = plt.subplots(1, 1, figsize=size)
        style.apply_to_figure(fig)
        style.apply_to_axes(ax)
        return fig, ax, style

    @staticmethod
    def _finalize_figure(fig, ax, figsize, style, save_path):
        #  - Handled automatically by autoscale_view
        ax.autoscale_view()  # Crucial when using Collections!

        if style.use_latex:
            ax.set_xlabel(r"$x$ [m]", fontsize=style.label_fontsize)
            ax.set_ylabel(r"$y$ [m]", fontsize=style.label_fontsize)
        else:
            ax.set_xlabel("x [m]", fontsize=style.label_fontsize)
            ax.set_ylabel("y [m]", fontsize=style.label_fontsize)

        ax.set_title(ax.get_title(), fontsize=style.title_fontsize)

        if style.background_color != 'white':
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('white')

        ax.set_aspect('equal', adjustable='box')

        if figsize is None:
            fig.canvas.draw()
            # Catch cases where limits are not yet set
            try:
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                data_w = x_max - x_min if x_max != x_min else 1.0
                data_h = y_max - y_min
                base_w = style.figsize[0]
                new_h = max(base_w * (data_h / data_w), 4.0)
                fig.set_size_inches(base_w, new_h)
            except:
                pass

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=style.dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"Saved: {save_path}")

    @staticmethod
    def plot_initial(structure, figsize=None, save_path=None, title=None, style=None, show_nodes=False):
        fig, ax, style = Visualizer._setup_figure(figsize, style)
        struct_type = Visualizer._detect_structure_type(structure)
        if struct_type in ['block', 'hybrid']:
            Visualizer._plot_blocks(ax, structure, style, show_deformed=False, show_nodes=show_nodes)
        if struct_type in ['fem', 'hybrid']:
            Visualizer._plot_fem_elements(ax, structure, style, show_deformed=False, show_nodes=show_nodes)
        Visualizer._finalize_figure(fig, ax, figsize, style, save_path)
        return fig

    @staticmethod
    def plot_deformed(structure, scale=1.0, figsize=None, save_path=None, title=None, style=None, show_nodes=False):
        if not hasattr(structure, 'U') or structure.U is None: raise ValueError("Structure not solved.")
        fig, ax, style = Visualizer._setup_figure(figsize, style)
        struct_type = Visualizer._detect_structure_type(structure)
        if struct_type in ['block', 'hybrid']:
            Visualizer._plot_blocks(ax, structure, style, scale=scale, show_original=False, show_nodes=show_nodes)
        if struct_type in ['fem', 'hybrid']:
            Visualizer._plot_fem_elements(ax, structure, style, scale=scale, show_original=False, show_nodes=show_nodes)
        if save_path is not None and scale != 1:
            root, ext = os.path.splitext(save_path)
            if not ext:  # No extension provided, default to .png
                ext = '.png'
            new_save_path = f"{root}_{scale}x{ext}"
        else:
            new_save_path = save_path
        Visualizer._finalize_figure(fig, ax, figsize, style, new_save_path)
        return fig

    @staticmethod
    def plot_comparison(structure, scale=1.0, figsize=None, save_path=None, title=None, style=None, show_nodes=False):
        if not hasattr(structure, 'U') or structure.U is None: raise ValueError("Structure not solved.")
        fig, ax, style = Visualizer._setup_figure(figsize, style)
        struct_type = Visualizer._detect_structure_type(structure)
        if struct_type in ['block', 'hybrid']:
            Visualizer._plot_blocks(ax, structure, style, scale=scale, show_nodes=show_nodes)
        if struct_type in ['fem', 'hybrid']:
            Visualizer._plot_fem_elements(ax, structure, style, scale=scale, show_nodes=show_nodes)
        if save_path is not None and scale != 1:
            root, ext = os.path.splitext(save_path)
            if not ext:  # No extension provided, default to .png
                ext = '.png'
            new_save_path = f"{root}_{scale}{ext}"
        else:
            new_save_path = save_path
        Visualizer._finalize_figure(fig, ax, figsize, style, new_save_path)
        return fig

    @staticmethod
    def plot_load_displacement(h5_files: Union[str, List[str]], dof_index: Union[int, List[int]] = None,
                               labels: List[str] = None, ref_loads: List[float] = None,
                               figsize: Tuple[float, float] = (8, 6), save_path: str = None,
                               title: str = None, style: Optional[PlotStyle] = None,
                               show_force: bool = False, normalize: bool = False,
                               colors: List[str] = None, linestyles: List[str] = None,
                               markers: List[str] = None) -> plt.Figure:

        # Normalize inputs
        if isinstance(h5_files, str): h5_files = [h5_files]
        n_files = len(h5_files)

        dof_indices = [dof_index] * n_files if isinstance(dof_index, int) else (
            dof_index if dof_index else [0] * n_files)
        labels = labels if labels else [os.path.splitext(os.path.basename(f))[0] for f in h5_files]
        ref_loads = [ref_loads] * n_files if isinstance(ref_loads, (int, float)) else (
            ref_loads if ref_loads else [1.0] * n_files)

        if colors is None:
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = [c['color'] for c in prop_cycle]
            colors = (colors * ((n_files // len(colors)) + 1))[:n_files]

        linestyles = linestyles if linestyles else ['-'] * n_files
        markers = markers if markers else ['o'] * n_files

        if style is None: style = PlotStyle()
        style.apply_global_config()
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
        style.apply_to_figure(fig)
        style.apply_to_axes(ax)

        for i, (h5_path, dof_idx, label, ref_load) in enumerate(zip(h5_files, dof_indices, labels, ref_loads)):
            if not os.path.exists(h5_path): continue
            with h5py.File(h5_path, 'r') as f:
                if 'U_conv' not in f or 'LoadFactor_conv' not in f: continue
                U_history = f['U_conv'][:]
                lambdas = f['LoadFactor_conv'][:]

                if dof_idx >= U_history.shape[0]: continue

                disp = U_history[dof_idx, :]
                y_values = lambdas * ref_load if show_force else lambdas

                if normalize:
                    disp = (disp - disp.min()) / (disp.max() - disp.min()) if disp.max() != disp.min() else disp
                    y_values = (y_values - y_values.min()) / (
                            y_values.max() - y_values.min()) if y_values.max() != y_values.min() else y_values

                ax.plot(disp, y_values, color=colors[i % len(colors)],
                        linestyle=linestyles[i % len(linestyles)], marker=markers[i % len(markers)],
                        markersize=1, label=label, linewidth=1)

        # Labels
        if normalize:
            ax.set_xlabel("Normalized Displacement [-]", fontsize=style.label_fontsize)
            y_lab = "Normalized Force [-]" if show_force else (
                r"Normalized Load Factor $\lambda$ [-]" if style.use_latex else "Normalized Load Factor [-]")
            ax.set_ylabel(y_lab, fontsize=style.label_fontsize)
        else:
            x_lab = r"Displacement [m]" if style.use_latex else "Displacement [m]"
            ax.set_xlabel(x_lab, fontsize=style.label_fontsize)
            if show_force:
                ax.set_ylabel(r"Force [N]" if style.use_latex else "Force [N]", fontsize=style.label_fontsize)
            else:
                ax.set_ylabel(r"Load Factor $\lambda$" if style.use_latex else "Load Factor (lambda)",
                              fontsize=style.label_fontsize)

        ax.set_title(title if title else "Load-Displacement Curve", fontsize=style.title_fontsize)
        if n_files > 1 or labels != [os.path.splitext(os.path.basename(f))[0] for f in h5_files]:
            ax.legend(fontsize=style.tick_labelsize)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=style.dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"  Saved: {save_path}")
        return fig

    @staticmethod
    def plot_stress(structure, scale: float = 0.0, figsize: tuple = None,
                    save_path: str = None, title: str = None,
                    style: PlotStyle = None, show_nodes: bool = False,
                    cmap: str = 'turbo', show_colorbar: bool = True,
                    vmin: float = None, vmax: float = None,
                    angle: float = None, component: str = None,
                    ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot stress field with element coloring.

        Visualizes stress fields for FEM elements and/or Block interfaces.

        Parameters
        ----------
        structure : Structure_FEM, Structure_Block, or Structure_Hybrid
            Solved structure.
        scale : float, default=0.0
            Deformation scale factor.
        figsize : tuple, optional
            Figure size (width, height).
        save_path : str, optional
            Path to save the figure.
        title : str, optional
            Custom title.
        style : PlotStyle, optional
            Plot styling.
        show_nodes : bool, default=False
            Overlay node markers.
        cmap : str, default='viridis'
            Colormap name.
        show_colorbar : bool = True
            Show colorbar.
        vmin, vmax : float, optional
            Color scale limits.
        angle : float, optional
            Filter/Direction angle in radians.
            - If provided: Shows directional stress on planes with this tangent angle.
            - If None (Default):
                - Blocks: Shows ALL interfaces.
                - FEM: Shows Principal Stresses (Max Principal for normal, Max Shear for shear).
        component : str, default='normal'
            Stress component to plot.
            Options: 'normal' (sigma), 'shear' (tau).
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot onto. If provided, figsize and save_path are ignored
            (unless handled externally).

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # Validation
        has_fem = hasattr(structure, 'list_fes') and structure.list_fes
        has_blocks = hasattr(structure, 'list_blocks') and structure.list_blocks

        if not has_fem and not (has_blocks and hasattr(structure, 'list_cfs') and structure.list_cfs):
            raise ValueError("Structure has neither FEM elements nor Block Contact Faces to plot.")
        if not hasattr(structure, 'U') or structure.U is None:
            raise ValueError("Structure not solved. Call solver first.")

        if component is None:
            component = 'normal'
        if component not in ['normal', 'shear', 'sigma', 's', 'tau', 't']:
            raise ValueError(f"Unknown component '{component}'. Use 'normal' or 'shear'.")

        # Setup figure
        if ax is None:
            fig, ax, style = Visualizer._setup_figure(figsize, style)
            is_subplot = False
        else:
            fig = ax.figure
            if style is None: style = PlotStyle()
            is_subplot = True

        all_stresses = []
        fem_verts_list = []
        fem_stress_vals = []
        block_verts_list = []
        block_stress_vals = []

        # Build node map only if necessary
        node_map = Visualizer._build_node_map(structure) if has_fem else {}

        # --- 1. FEM Processing ---
        if has_fem:
            if not hasattr(structure, 'compute_nodal_directional_stress'):
                print("Warning: Structure does not support stress computation. FEM stress skipped.")
            else:
                if angle is not None:
                    # Directional Stress
                    sigma_n, tau_nt = structure.compute_nodal_directional_stress(angle)
                    if component in ['normal', 'sigma', 's']:
                        nodal_values = sigma_n
                    else:
                        nodal_values = tau_nt
                else:
                    # Principal Stress (No angle specified)
                    sigma_1, sigma_2, tau_max = structure.compute_nodal_principal_stress()
                    if component in ['normal', 'sigma', 's']:
                        nodal_values = sigma_1  # Max Principal Stress
                    else:
                        nodal_values = tau_max  # Max Shear Stress

                for fe in structure.list_fes:
                    # Get global node indices for this element
                    node_indices = []
                    for node_coord in fe.nodes:
                        key = (round(node_coord[0], 9), round(node_coord[1], 9))
                        node_indices.append(node_map.get(key, 0))

                    # Build deformed/undeformed coordinates
                    coords = np.array(fe.nodes).copy()
                    if scale != 0.0:
                        for j, node_idx in enumerate(node_indices):
                            dof_offset = structure.node_dof_offsets[node_idx]
                            coords[j, 0] += structure.U[dof_offset] * scale
                            coords[j, 1] += structure.U[dof_offset + 1] * scale

                    # Get boundary with proper higher-order element handling
                    boundary = _get_element_boundary_coords(coords, fe, n_interp=10)
                    fem_verts_list.append(boundary)

                    # Average stress at element nodes
                    elem_val = np.mean([nodal_values[idx] for idx in node_indices]) / 1e6
                    fem_stress_vals.append(elem_val)
                    all_stresses.append(elem_val)

        # --- 2. Block Processing ---
        if has_blocks and hasattr(structure, 'list_cfs') and structure.list_cfs:
            comp_key = 's' if component in ['normal', 'sigma', 's'] else 't'

            for cf in structure.list_cfs:
                # Filter by angle ONLY if specified
                if angle is not None:
                    # Check orientation (periodic modulo pi) to handle anti-parallel vectors
                    # e.g. 0 and pi are the same physical line orientation
                    diff = abs(cf.angle - angle)
                    # Check distance to 0, +pi, or -pi
                    if min(diff, abs(diff - np.pi), abs(diff + np.pi)) > 1e-5:
                        continue

                for cp in cf.cps:
                    if hasattr(cp, 'to_ommit') and cp.to_ommit():
                        continue

                    val1 = cp.sp1.law.stress[comp_key] / 1e6
                    val2 = cp.sp2.law.stress[comp_key] / 1e6

                    # Get geometry
                    poly1 = cp.vertices_fibA
                    poly2 = cp.vertices_fibB

                    if scale != 0.0:
                        poly1 = Visualizer._transform_points_rigid(poly1, cp.bl_A, structure, scale)
                        poly2 = Visualizer._transform_points_rigid(poly2, cp.bl_B, structure, scale)

                    block_verts_list.append(poly1)
                    block_stress_vals.append(val1)
                    all_stresses.append(val1)

                    block_verts_list.append(poly2)
                    block_stress_vals.append(val2)
                    all_stresses.append(val2)

        # --- 3. Normalization ---
        if not all_stresses:
            print("Warning: No stress data found to plot.")
            Visualizer._finalize_figure(fig, ax, figsize, style, save_path)
            return fig

        all_stresses_np = np.array(all_stresses)
        calc_vmin, calc_vmax = all_stresses_np.min(), all_stresses_np.max()

        if vmin is None: vmin = calc_vmin
        if vmax is None: vmax = calc_vmax

        if vmax == vmin:
            vmax = vmin + 1e-10  # Avoid division by zero

        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap(cmap)

        # --- 4. Plotting ---

        # Draw blocks outlines (Neutral background) if present
        if has_blocks:
            Visualizer._plot_blocks(ax, structure, style, scale=scale, show_original=True, show_deformed=False,
                                    show_nodes=show_nodes)

        # Plot FEM Elements
        if fem_verts_list:
            colors = colormap(norm(fem_stress_vals))
            pc_fem = PolyCollection(fem_verts_list,
                                    facecolors=colors,
                                    edgecolors='black',
                                    linewidths=0.1,
                                    alpha=0.5)
            ax.add_collection(pc_fem)

        # Plot Block Interfaces
        if block_verts_list:
            colors = colormap(norm(block_stress_vals))
            pc_block = PolyCollection(block_verts_list,
                                      facecolors=colors,
                                      edgecolors='none',
                                      linewidths=0.1,
                                      alpha=0.5)
            ax.add_collection(pc_block)

        # Colorbar - orient based on structure shape
        if show_colorbar:
            sm = cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])

            # Detect orientation and set colorbar accordingly
            orientation = Visualizer._detect_orientation(structure)
            if orientation == 'horizontal':
                cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', location='bottom', pad=0.15)
            else:
                cbar = plt.colorbar(sm, ax=ax, orientation='vertical', location='right')

            label = "Stress (MPa)"
            cbar.set_label(label, fontsize=style.label_fontsize)

        # Node markers
        if show_nodes:
            unique_nodes = set()
            if has_fem:
                for fe in structure.list_fes:
                    for coord in fe.nodes:
                        key = (round(coord[0], 9), round(coord[1], 9))
                        if key in node_map:
                            node_idx = node_map[key]
                            x, y = coord[0], coord[1]
                            if scale != 0.0:
                                dof_offset = structure.node_dof_offsets[node_idx]
                                x += structure.U[dof_offset] * scale
                                y += structure.U[dof_offset + 1] * scale
                            unique_nodes.add((x, y))

            if unique_nodes:
                pts = np.array(list(unique_nodes))
                ax.scatter(pts[:, 0], pts[:, 1],
                           s=style.node_size, c='black',
                           marker=style.node_marker, alpha=style.node_alpha, zorder=10)

        # Title
        geom_label = fr"Deformed (Scale: {scale}\times)" if scale != 0.0 else "Undeformed"

        # Only set title/finalize if we own the figure or explicit title given
        if title:
            ax.set_title(title, fontsize=style.title_fontsize)

        # Ensure limits are updated to include all collections
        ax.autoscale(enable=True, axis='both', tight=True)

        if not is_subplot:
            Visualizer._finalize_figure(fig, ax, figsize, style, save_path)
        else:
            # For subplots, use 'box' to make the plot box fit the data tightly while maintaining aspect ratio
            ax.set_aspect('equal', adjustable='box')
            ax.set_anchor('C')
            
        return fig

    @staticmethod
    def plot_displacement(structure, scale: float = 0.0, component: str = 'magnitude',
                          figsize: tuple = None, save_path: str = None, title: str = None,
                          style: PlotStyle = None, show_nodes: bool = False,
                          cmap: str = 'viridis', show_colorbar: bool = True,
                          vmin: float = None, vmax: float = None,
                          ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Plot displacement field with element coloring.

        This method displays displacements as a color-coded mesh. Each element
        is colored based on the average displacement of its nodes.

        Parameters
        ----------
        structure : Structure_FEM or Hybrid
            Solved structure with FEM elements (structure.U must be populated)
        scale : float, default=0.0
            Deformation scale factor for displaying deformed geometry.
            Use 1.0 for true scale, higher values to exaggerate deformation.
        component : str, default='magnitude'
            Displacement component to plot:
            - 'magnitude': sqrt(ux^2 + uy^2) - total displacement
            - 'x' or 'ux': horizontal displacement
            - 'y' or 'uy': vertical displacement
        figsize : tuple, optional
            Figure size (width, height) in inches
        save_path : str, optional
            Path to save the figure
        title : str, optional
            Custom title for the plot
        style : PlotStyle, optional
            Plot styling configuration
        show_nodes : bool, default=False
            Whether to overlay node markers
        cmap : str, default='viridis'
            Matplotlib colormap name ('viridis', 'jet', 'coolwarm', 'RdBu', etc.)
        show_colorbar : bool, default=True
            Whether to show colorbar with displacement scale
        vmin : float, optional
            Minimum value for colormap scaling. If None, calculated from data.
        vmax : float, optional
            Maximum value for colormap scaling. If None, calculated from data.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot onto. If provided, figsize and save_path are ignored.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        # Validation
        has_fem = hasattr(structure, 'list_fes') and structure.list_fes
        has_blocks = hasattr(structure, 'list_blocks') and structure.list_blocks

        if not has_fem and not has_blocks:
            raise ValueError("Structure has neither FEM elements nor Blocks to plot.")
        if not hasattr(structure, 'U') or structure.U is None:
            raise ValueError("Structure not solved. Call solver first.")

        # Setup figure
        if ax is None:
            fig, ax, style = Visualizer._setup_figure(figsize, style)
            is_subplot = False
        else:
            fig = ax.figure
            if style is None: style = PlotStyle()
            is_subplot = True

        # Build node coordinate map for fast lookup
        node_map = Visualizer._build_node_map(structure)

        # Compute nodal displacements
        n_nodes = len(structure.list_nodes)
        nodal_disp = np.zeros(n_nodes)

        for idx in range(n_nodes):
            dof_offset = structure.node_dof_offsets[idx]
            ux = structure.U[dof_offset]
            uy = structure.U[dof_offset + 1]

            if component in ['magnitude', 'mag', 'norm']:
                nodal_disp[idx] = np.sqrt(ux ** 2 + uy ** 2)
            elif component in ['x', 'ux', 'X', 'horizontal']:
                nodal_disp[idx] = ux
            elif component in ['y', 'uy', 'Y', 'vertical']:
                nodal_disp[idx] = uy
            else:
                raise ValueError(f"Unknown component '{component}'. Use 'magnitude', 'x', or 'y'.")

        # Prepare element polygons and displacement values
        verts_list = []
        elem_disp = []

        # 1. FEM Elements
        if has_fem:
            for fe in structure.list_fes:
                # Get global node indices for this element
                node_indices = []
                for node_coord in fe.nodes:
                    key = (round(node_coord[0], 9), round(node_coord[1], 9))
                    node_indices.append(node_map.get(key, 0))

                # Build deformed/undeformed coordinates
                coords = np.array(fe.nodes).copy()
                if scale != 0.0:
                    for j, node_idx in enumerate(node_indices):
                        dof_offset = structure.node_dof_offsets[node_idx]
                        coords[j, 0] += structure.U[dof_offset] * scale
                        coords[j, 1] += structure.U[dof_offset + 1] * scale

                # Get boundary with proper higher-order element handling
                boundary = _get_element_boundary_coords(coords, fe, n_interp=10)
                verts_list.append(boundary)

                # Average displacement at element nodes
                elem_d = np.mean([nodal_disp[idx] for idx in node_indices])
                elem_disp.append(elem_d)

        # 2. Blocks
        if has_blocks:
            for block in structure.list_blocks:
                # Get displacement of block centroid/reference point
                node_idx = block.connect
                dof_offset = structure.node_dof_offsets[node_idx]
                ux = structure.U[dof_offset]
                uy = structure.U[dof_offset + 1]

                if component in ['magnitude', 'mag', 'norm']:
                    block_val = np.sqrt(ux ** 2 + uy ** 2)
                elif component in ['x', 'ux', 'X', 'horizontal']:
                    block_val = ux
                elif component in ['y', 'uy', 'Y', 'vertical']:
                    block_val = uy
                else:
                    block_val = 0.0

                # Compute deformed geometry
                poly = Visualizer._compute_block_deformed_poly(block, structure, scale)
                verts_list.append(poly)
                elem_disp.append(block_val)

        # Normalize colors
        if not elem_disp:
            print("Warning: No displacement data found to plot.")
            Visualizer._finalize_figure(fig, ax, figsize, style, save_path)
            return fig

        elem_disp = np.array(elem_disp)
        calc_vmin, calc_vmax = elem_disp.min(), elem_disp.max()

        # For signed components (x, y), use symmetric colormap centered at 0
        if component in ['x', 'ux', 'X', 'horizontal', 'y', 'uy', 'Y', 'vertical']:
            abs_max = max(abs(calc_vmin), abs(calc_vmax))
            if abs_max > 0:
                calc_vmin, calc_vmax = -abs_max, abs_max

        if vmin is None: vmin = calc_vmin
        if vmax is None: vmax = calc_vmax

        if vmax == vmin:
            vmax = vmin + 1e-10

        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap(cmap)
        colors = colormap(norm(elem_disp))

        # Draw filled polygons with displacement colors
        pc = PolyCollection(verts_list,
                            facecolors=colors,
                            edgecolors='black',
                            linewidths=0.1,
                            alpha=1)
        ax.add_collection(pc)

        # Colorbar - orient based on structure shape
        if show_colorbar:
            sm = cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])

            # Detect orientation and set colorbar accordingly
            orientation = Visualizer._detect_orientation(structure)
            if orientation == 'horizontal':
                cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', location='bottom', pad=0.15)
            else:
                cbar = plt.colorbar(sm, ax=ax, orientation='vertical', location='right')

            # Label based on component
            if component in ['magnitude', 'mag', 'norm']:
                cbar_label = 'Displacement (m)'
            elif component in ['x', 'ux', 'X', 'horizontal']:
                cbar_label = 'Horizontal Displacement (m)'
            elif component in ['y', 'uy', 'Y', 'vertical']:
                cbar_label = 'Vertical Displacement (m)'
            else:
                cbar_label = 'Displacement (m)'

            cbar.set_label(cbar_label, fontsize=style.label_fontsize)

        # Node markers
        if show_nodes:
            unique_nodes = set()
            # FEM Nodes
            if has_fem:
                for fe in structure.list_fes:
                    for coord in fe.nodes:
                        key = (round(coord[0], 9), round(coord[1], 9))
                        if key in node_map:
                            node_idx = node_map[key]
                            x, y = coord[0], coord[1]
                            dof_offset = structure.node_dof_offsets[node_idx]
                            x += structure.U[dof_offset] * scale
                            y += structure.U[dof_offset + 1] * scale
                            unique_nodes.add((x, y))

            # Block Nodes
            if has_blocks:
                for block in structure.list_blocks:
                    ref_x, ref_y = block.ref_point
                    node_idx = block.connect
                    if scale != 0.0:
                        dof_offset = structure.node_dof_offsets[node_idx]
                        ref_x += structure.U[dof_offset] * scale
                        ref_y += structure.U[dof_offset + 1] * scale
                    unique_nodes.add((ref_x, ref_y))

            if unique_nodes:
                pts = np.array(list(unique_nodes))
                ax.scatter(pts[:, 0], pts[:, 1],
                           s=style.node_size, c='black',
                           marker=style.node_marker, alpha=style.node_alpha, zorder=10)


        if title:
            ax.set_title(title, fontsize=style.title_fontsize)

        # Ensure limits are updated to include all collections
        ax.autoscale(enable=True, axis='both', tight=True)

        if not is_subplot:
            Visualizer._finalize_figure(fig, ax, figsize, style, save_path)
        else:
            # For subplots
            ax.set_aspect('equal', adjustable='box')
            ax.set_anchor('C')

        return fig
