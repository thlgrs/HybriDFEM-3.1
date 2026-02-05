"""
Matplotlib-based viewport widget for structure visualization.

This module provides the central viewport for displaying structures,
meshes, loads, and analysis results with interactive node selection.
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection, PolyCollection

from GUI.Dialogs.AddNodalLoadDialog import AddNodalLoadDialog
from GUI.Dialogs.AddSupportDialog import AddSupportDialog

# Import Visualizer for enhanced plotting
try:
    from Core.Solvers.Visualizer import Visualizer, PlotStyle
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False


class ViewportWidget(QWidget):
    """
    Central widget for matplotlib visualization with interactive node selection.
    """

    def __init__(self, project_state, parent=None):
        super().__init__(parent)
        self.project_state = project_state
        self.main_window = parent  # Reference to MainWindow for dialogs

        layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Signal connections
        self.project_state.structure_changed.connect(self.plot_geometry)
        self.project_state.results_ready.connect(self.plot_results)
        self.click_cid = self.canvas.mpl_connect(
            'button_press_event', self.on_canvas_click
        )
        self.project_state.selection_mode_changed.connect(self.on_selection_mode_changed)

        self.plot_geometry()

    def on_canvas_click(self, event):
        if self.project_state.selection_mode == "idle" or event.xdata is None or event.ydata is None:
            return
        try:
            node_id, node_coords = self.find_closest_node(event.xdata, event.ydata)
        except ValueError as e:
            self.project_state.log_message.emit(f"[Info] {e}")
            return
        self.project_state.log_message.emit(
            f"Click near ({event.xdata:.2f}, {event.ydata:.2f}). Closest node: {node_id}")
        if self.project_state.selection_mode == "select_support_node":
            self.handle_support_selection(node_id)
        elif self.project_state.selection_mode == "select_load_node":
            self.handle_load_selection(node_id)
        self.project_state.set_selection_mode("idle")

    def find_closest_node(self, x_click, y_click):
        structure = self.project_state.structure
        if not structure or not hasattr(structure, 'list_nodes') or not structure.list_nodes:
            raise ValueError("No structure or nodes to select.")
        nodes = np.array(structure.list_nodes)
        click_coord = np.array([x_click, y_click])
        distances = np.linalg.norm(nodes - click_coord, axis=1)
        closest_node_id = np.argmin(distances)
        return closest_node_id, nodes[closest_node_id]

    def handle_support_selection(self, node_id):
        dialog = AddSupportDialog(node_id, self.main_window)
        if dialog.exec():
            data = dialog.get_data()
            self.project_state.add_support_to_node(node_id=data["node_id"], dofs=data["dofs"])

    def handle_load_selection(self, node_id):
        dialog = AddNodalLoadDialog(node_id, self.main_window)
        if dialog.exec():
            data = dialog.get_data()
            if data["loads"]:
                self.project_state.add_load_to_node(node_id=data["node_id"], loads_list=data["loads"])
            else:
                self.project_state.log_message.emit("Load addition cancelled (zero values).")

    def on_selection_mode_changed(self, mode):
        from PyQt6.QtCore import Qt
        if mode == "idle":
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    def plot_geometry(self):
        """Plot the current structure geometry with enhanced visualization."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        structure = self.project_state.structure

        # Handle empty structure
        if structure is None:
            self._show_empty_state(ax, "No structure created yet")
            return

        # Check if structure has any geometry
        has_blocks = hasattr(structure, 'list_blocks') and len(structure.list_blocks) > 0
        has_fem = hasattr(structure, 'list_fes') and len(structure.list_fes) > 0
        has_nodes = hasattr(structure, 'list_nodes') and len(structure.list_nodes) > 0

        if not has_nodes:
            # Structure exists but no geometry added yet
            structure_type = self.project_state.structure_type.upper()
            self._show_empty_state(
                ax,
                f"{structure_type} structure ready\n\n"
                f"Add geometry using the Geometry panel:\n"
                f"• Blocks: Use 'Add Block' button\n"
                f"• FEM: Use 'Generate Mesh' button"
            )
            return

        try:
            # Get plot style from project state
            style = self.project_state.plot_style if self.project_state.plot_style else None

            # Use Visualizer if available, otherwise fallback to direct plotting
            if VISUALIZER_AVAILABLE:
                self._plot_with_visualizer(ax, structure, style, show_deformed=False)
            else:
                self._plot_fallback(ax, structure, style, show_deformed=False)

            # Draw boundary conditions on top
            self.draw_bcs(ax)

            # Build informative title
            title_parts = []
            if has_blocks:
                n_blocks = len(structure.list_blocks)
                title_parts.append(f"{n_blocks} block{'s' if n_blocks != 1 else ''}")
            if has_fem:
                n_elements = len(structure.list_fes)
                title_parts.append(f"{n_elements} FEM element{'s' if n_elements != 1 else ''}")
            if has_nodes:
                n_nodes = len(structure.list_nodes)
                n_dofs = structure.nb_dofs if hasattr(structure, 'nb_dofs') else 0
                title_parts.append(f"{n_nodes} nodes ({n_dofs} DOFs)")

            title = "Geometry: " + ", ".join(title_parts) if title_parts else "Geometry (Undeformed)"
            ax.set_title(title, fontsize=11, fontweight='bold')

            # Enhance axis appearance
            ax.set_xlabel("X [m]", fontsize=10)
            ax.set_ylabel("Y [m]", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_aspect('equal', adjustable='box')
            ax.autoscale_view()

            self.canvas.draw()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.project_state.log_message.emit(f"[ERROR] Geometry plot failed: {e}")
            self._show_error_state(ax, f"Plot error: {str(e)}")

    def _plot_with_visualizer(self, ax, structure, style, show_deformed=False, scale=1.0):
        """Use Visualizer class for plotting (robust, handles all structure types)."""
        if style is None:
            style = PlotStyle()

        # Detect structure type
        has_blocks = hasattr(structure, 'list_blocks') and len(structure.list_blocks) > 0
        has_fem = hasattr(structure, 'list_fes') and len(structure.list_fes) > 0

        # Plot blocks
        if has_blocks:
            Visualizer._plot_blocks(ax, structure, style, scale=scale,
                                   show_original=not show_deformed,
                                   show_deformed=show_deformed,
                                   show_nodes=True)

        # Plot FEM elements
        if has_fem:
            Visualizer._plot_fem_elements(ax, structure, style, scale=scale,
                                         show_original=not show_deformed,
                                         show_deformed=show_deformed,
                                         show_nodes=True)

    def _plot_fallback(self, ax, structure, style, show_deformed=False, scale=1.0):
        """Fallback plotting when Visualizer is not available."""
        has_blocks = hasattr(structure, 'list_blocks') and len(structure.list_blocks) > 0
        has_fem = hasattr(structure, 'list_fes') and len(structure.list_fes) > 0

        # Plot blocks as polygons
        if has_blocks:
            for block in structure.list_blocks:
                vertices = np.array(block.v)
                # Close polygon
                vertices = np.vstack([vertices, vertices[0]])
                ax.plot(vertices[:, 0], vertices[:, 1], 'b-', linewidth=1)
                ax.fill(vertices[:, 0], vertices[:, 1], alpha=0.2, color='lightblue')
                # Mark reference point
                ax.plot(block.ref_point[0], block.ref_point[1], 'bs', markersize=4)

        # Plot FEM elements as line segments
        if has_fem:
            segments = []
            for fe in structure.list_fes:
                coords = np.array(fe.nodes)
                # Create closed polygon
                for i in range(len(coords)):
                    p1 = coords[i]
                    p2 = coords[(i + 1) % len(coords)]
                    segments.append([p1, p2])

            if segments:
                lc = LineCollection(segments, colors='gray', linewidths=0.8, alpha=0.8)
                ax.add_collection(lc)

        # Plot nodes
        if hasattr(structure, 'list_nodes') and structure.list_nodes:
            nodes = np.array(structure.list_nodes)
            ax.scatter(nodes[:, 0], nodes[:, 1], s=10, c='black', marker='o', zorder=5)

    def _show_empty_state(self, ax, message):
        """Display an informative empty state message."""
        ax.text(0.5, 0.5, message,
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=12, color='gray',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        self.canvas.draw()

    def _show_error_state(self, ax, message):
        """Display an error message in the viewport."""
        ax.text(0.5, 0.5, f"⚠️ {message}",
                ha='center', va='center',
                transform=ax.transAxes,
                fontsize=11, color='red',
                bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.canvas.draw()

    def plot_results(self, solved_structure):
        """
        Slot triggered when 'results_ready' signal is emitted.
        Plots analysis results with deformed shape.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        if solved_structure is None:
            self.plot_geometry()  # Return to geometry plot
            return

        try:
            # Get deformation scale from results panel
            try:
                scale = self.main_window.results_panel.deformation_scale.value()
            except Exception as e:
                print(f"Error reading scale: {e}")
                scale = 50.0  # Default value

            # Get plot style from project state
            style = self.project_state.plot_style if self.project_state.plot_style else None

            # Use Visualizer if available
            if VISUALIZER_AVAILABLE:
                self._plot_with_visualizer(ax, solved_structure, style,
                                          show_deformed=True, scale=scale)
            else:
                self._plot_fallback(ax, solved_structure, style,
                                   show_deformed=True, scale=scale)

            # Redraw boundary conditions on top
            self.draw_bcs(ax)

            # Build informative title with structure info
            title_parts = [f"Results (Deformed, Scale {scale:.0f}x)"]
            if hasattr(solved_structure, 'U') and solved_structure.U is not None:
                max_disp = np.max(np.abs(solved_structure.U))
                title_parts.append(f"Max |u| = {max_disp:.4e} m")

            ax.set_title(" | ".join(title_parts), fontsize=11, fontweight='bold')

            # Enhance axis appearance
            ax.set_xlabel("X [m]", fontsize=10)
            ax.set_ylabel("Y [m]", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_aspect('equal', adjustable='box')
            ax.autoscale_view()

            self.canvas.draw()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.project_state.log_message.emit(f"[ERROR] Results plot failed: {e}")
            self._show_error_state(ax, f"Results plot error: {str(e)}")

    def draw_bcs(self, ax):
        """Draw boundary conditions (supports and loads) on the plot with enhanced visibility."""
        if not self.project_state.structure or not self.project_state.structure.list_nodes:
            return

        nodes = np.array(self.project_state.structure.list_nodes)
        lims_x = ax.get_xlim()
        lims_y = ax.get_ylim()
        diag = np.sqrt((lims_x[1] - lims_x[0]) ** 2 + (lims_y[1] - lims_y[0]) ** 2)
        if diag <= 1e-6:
            diag = 1.0
        symbol_size = diag * 0.05

        # Draw supports as blue triangles with better visibility
        for node_id, dofs in self.project_state.supports.items():
            if node_id >= len(nodes):
                continue
            x, y = nodes[node_id]
            # Larger marker for better visibility
            ax.plot(x, y, '^', color='blue', markersize=12,
                   markeredgecolor='darkblue', markeredgewidth=2, zorder=100)

            # Add text label showing constrained DOFs
            # Convert DOF indices [0,1,2] to strings ['x','y','z']
            dof_map = ['x', 'y', 'z']
            dof_text = "".join(dof_map[d] for d in dofs)  # e.g., "xy" or "xyz"
            ax.text(x, y - symbol_size * 0.3, f"Fixed: {dof_text}",
                   ha='center', va='top', fontsize=8,
                   color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='blue'))

        # Draw loads as red arrows with labels
        for node_id, load_list in self.project_state.loads.items():
            if node_id >= len(nodes):
                continue
            x, y = nodes[node_id]

            for dof_index, value in load_list:
                if value == 0:
                    continue

                hw = symbol_size * 0.4  # Larger head width
                hl = symbol_size * 0.4  # Larger head length

                if dof_index == 0:  # Fx
                    dx, dy = symbol_size * 0.9 * np.sign(value), 0
                    x_start = x - dx
                    ax.arrow(x_start, y, dx, dy, head_width=hw, head_length=hl,
                            fc='red', ec='darkred', linewidth=2, zorder=100)
                    # Add load magnitude label
                    label_x = x_start - symbol_size * 0.2
                    ax.text(label_x, y, f"{value:.1f} N",
                           ha='center', va='bottom', fontsize=8,
                           color='red', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'))

                elif dof_index == 1:  # Fy
                    dx, dy = 0, symbol_size * 0.9 * np.sign(value)
                    y_start = y - dy
                    ax.arrow(x, y_start, dx, dy, head_width=hw, head_length=hl,
                            fc='red', ec='darkred', linewidth=2, zorder=100)
                    # Add load magnitude label
                    label_y = y_start - symbol_size * 0.2
                    ax.text(x, label_y, f"{value:.1f} N",
                           ha='left', va='center', fontsize=8,
                           color='red', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'))
