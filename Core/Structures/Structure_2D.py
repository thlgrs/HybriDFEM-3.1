"""
Structure_2D - Abstract Base Class for 2D Structural Analysis
==============================================================

This module defines the base class for all 2D structures in HybriDFEM.
It provides core functionality shared by FEM, Block, and Hybrid structures:

1. **Node Management**: Adding nodes, finding nodes by coordinate, spatial indexing
2. **DOF Management**: Variable DOFs per node (2 for FEM, 3 for blocks)
3. **Boundary Conditions**: Fixing DOFs, applying loads
4. **Solution Storage**: Displacement (U), forces (P), internal forces (P_r)
5. **Matrix Storage**: Stiffness (K, K0, K_LG), Mass (M), Damping (C)

Class Hierarchy:
    Structure_2D (ABC)
    ├── Structure_FEM      # Pure FEM, 2 DOF/node [ux, uy]
    ├── Structure_Block    # Pure blocks, 3 DOF/node [ux, uy, rz]
    └── Hybrid             # Combined, variable DOF/node

For Students:
    - This class uses the Template Method pattern: abstract methods define
      what subclasses must implement (make_nodes, get_K_str, etc.)
    - DOF = Degree of Freedom = independent displacement component
    - The node_dof_offsets array tracks cumulative DOF counts for mixed structures

Example DOF layout for a hybrid structure with 2 blocks and 4 FEM nodes:
    Block 0: DOFs 0,1,2 (ux, uy, rz)
    Block 1: DOFs 3,4,5 (ux, uy, rz)
    FEM node 2: DOFs 6,7 (ux, uy)
    FEM node 3: DOFs 8,9 (ux, uy)
    ...
    node_dof_offsets = [0, 3, 6, 8, 10, ...]
"""

import math
import pickle
import warnings
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


class Structure_2D(ABC):
    """
    Abstract base class for 2D structures with variable DOF support.

    This class cannot be instantiated directly. Use Structure_FEM, Structure_Block,
    or Hybrid depending on your analysis type.

    Attributes
    ----------
    list_nodes : list
        Global node coordinates as list of [x, y] arrays
    node_dof_counts : list
        Number of DOFs at each node (2 for FEM nodes, 3 for block nodes)
    node_dof_offsets : list
        Cumulative DOF count: node i starts at DOF index node_dof_offsets[i]
    nb_dofs : int
        Total number of DOFs in the structure
    U : ndarray
        Global displacement vector
    P : ndarray
        External force vector (variable loads)
    P_fixed : ndarray
        External force vector (constant loads, e.g., self-weight)
    P_r : ndarray
        Internal force vector (from element/contact stresses)
    dof_fix : ndarray
        Indices of fixed (constrained) DOFs
    dof_free : ndarray
        Indices of free (unknown) DOFs
    K : ndarray
        Tangent stiffness matrix (current configuration)
    K0 : ndarray
        Initial stiffness matrix (reference configuration)
    K_LG : ndarray
        Geometric stiffness matrix (large deformation effects)
    M : ndarray
        Mass matrix

    Notes
    -----
    The equilibrium equation solved is: K * U = P - P_r
    where P is external load, P_r is internal resistance.
    """
    # Default DOFs per node (blocks have 3: ux, uy, rotation_z)
    # FEM elements override this to 2 (ux, uy only)
    DOF_PER_NODE = 3  # [ux, uy, rz]

    def __init__(self, structure_type: str = None):
        self.structure_type = structure_type

        # Geometry
        self.list_nodes = []

        # DOF Management
        self.node_dof_counts = []
        self.node_dof_offsets = [0]
        self.nb_dofs = None

        # Optimization (Spatial)
        self._kdtree = None
        self._kdtree_n = 0
        self._node_hash = None
        self._node_hash_decimals = None

        # Solution Vectors
        self.U = None
        self.P = None
        self.P_fixed = None
        self.P_r = None

        # Boundary Conditions
        self.dof_fix = np.array([], dtype=int)
        self.dof_free = np.array([], dtype=int)
        self.nb_dof_fix = 0
        self.nb_dof_free = 0

        # Matrices
        self.K = None
        self.K0 = None
        self.K_LG = None
        self.M = None

        # Damping
        self.xsi = [0.0, 0.0]
        self.damp_type = "RAYLEIGH"
        self.stiff_type = "INIT"

    # ==========================================================================
    # Abstract Methods
    # ==========================================================================
    @abstractmethod
    def make_nodes(self):
        pass

    @abstractmethod
    def get_P_r(self):
        pass

    @abstractmethod
    def get_M_str(self):
        pass

    @abstractmethod
    def get_K_str(self):
        pass

    @abstractmethod
    def get_K_str0(self):
        pass

    @abstractmethod
    def get_K_str_LG(self):
        pass

    @abstractmethod
    def set_lin_geom(self, lin_geom=True):
        pass

    # ==========================================================================
    # Node & Geometry Management
    # ==========================================================================

    def _validate_coord(self, node):
        """Helper to safely convert input to flat float array or return None."""
        try:
            arr = np.asanyarray(node, dtype=float).flatten()
            return arr if arr.size == 2 else None
        except Exception:
            return None

    def build_node_tree(self):
        """Builds KD-tree for fast spatial lookups."""

        nodes = np.array(self.list_nodes)
        self._kdtree = cKDTree(nodes)
        self._kdtree_n = len(nodes)

    def _refresh_node_tree(self):
        if (self._kdtree is None) or (self._kdtree_n != len(self.list_nodes)):
            self.build_node_tree()

    def _refresh_node_hash(self, tol: float):
        decimals = max(0, int(round(-math.log10(max(tol, 1e-15)))))
        if self._node_hash is None or self._node_hash_decimals != decimals:
            self._node_hash = {
                (round(n[0], decimals), round(n[1], decimals)): i
                for i, n in enumerate(self.list_nodes)
            }
            self._node_hash_decimals = decimals

    def get_node_id(self, node, tol: float = 1e-8, optimized: bool = False):
        """Find global node index by coordinates. Returns None if not found."""
        if isinstance(node, (int, np.integer)):
            return int(node)

        target = self._validate_coord(node)
        if target is None:
            return None

        # Strategy A: KD-Tree (O(log N))
        if optimized:
            try:
                self._refresh_node_tree()
                if self._kdtree:
                    dist, idx = self._kdtree.query(target)
                    if dist <= tol:
                        return int(idx)
            except ValueError:
                pass
            return None

        # Strategy B: Vectorized Euclidean Search (O(N))
        if not self.list_nodes:
            return None

        nodes = np.array(self.list_nodes)
        deltas = nodes - target
        dist_sq = np.sum(deltas ** 2, axis=1)
        matches = np.nonzero(dist_sq <= tol ** 2)[0]

        return int(matches[0]) if matches.size > 0 else None

    def _add_node_if_new(self, node, tol=1e-9, optimized=True, use_hash=False, dof_count=None, force_new=False):
        """Adds node if it doesn't exist. Returns node index.

        Parameters
        ----------
        force_new : bool
            If True, always create a new node even if one exists at same position.
            Useful for FEM-FEM coupling with split meshes.
        """
        target = self._validate_coord(node)
        if target is None:
            raise ValueError("Invalid node coordinates")

        dof_count = dof_count if dof_count is not None else self.DOF_PER_NODE
        idx = None

        # Skip duplicate check if force_new is True
        if not force_new:
            # Try finding existing node
            if use_hash:
                self._refresh_node_hash(tol)
                key = (round(target[0], self._node_hash_decimals), round(target[1], self._node_hash_decimals))
                idx = self._node_hash.get(key)

            if idx is None:
                # Fallback to standard search if hash failed or wasn't used
                idx = self.get_node_id(target, tol=tol, optimized=optimized)

            # Node Exists: Validate and Return
            if idx is not None:
                existing_dof_count = self.node_dof_counts[idx] if self.node_dof_counts else dof_count
                # Allow using a higher-DOF node for a lower-DOF element
                # (e.g., FEM element with 2 DOFs can use a 3-DOF block node)
                # But raise error if we need MORE DOFs than the existing node has
                if existing_dof_count < dof_count:
                    raise ValueError(f"DOF mismatch at node {idx}: existing={existing_dof_count}, requested={dof_count}")
                if use_hash: self._node_hash[key] = idx
                return idx

        # Node New: Create
        self.list_nodes.append(target)
        new_idx = len(self.list_nodes) - 1

        self.node_dof_counts.append(dof_count)
        self.node_dof_offsets.append(self.node_dof_offsets[-1] + dof_count)

        # Invalidate/Update Cache
        self._kdtree = None
        if use_hash and self._node_hash is not None:
            self._node_hash[key] = new_idx

        return new_idx

    # ==========================================================================
    # DOF Helpers
    # ==========================================================================

    def compute_nb_dofs(self):
        return self.node_dof_offsets[-1]

    def dofs_defined(self):
        if not self.nb_dofs:
            warnings.warn("Structure DOFs not defined.")

    def get_dofs_from_node(self, node_id: int) -> np.ndarray:
        if node_id + 1 < len(self.node_dof_offsets):
            return np.arange(self.node_dof_offsets[node_id], self.node_dof_offsets[node_id + 1])
        # Fallback for old/simple structures
        start = self.DOF_PER_NODE * node_id
        return np.arange(start, start + self.DOF_PER_NODE)

    def _global_dof(self, node_id: int, local_dof: int) -> int:
        if node_id + 1 < len(self.node_dof_offsets):
            return self.node_dof_offsets[node_id] + int(local_dof)
        return (self.DOF_PER_NODE * int(node_id)) + int(local_dof)

    def _iter_dofs(self, dofs):
        """Recursively yields integers from nested lists/arrays."""
        if isinstance(dofs, (int, np.integer)):
            yield int(dofs)
        elif isinstance(dofs, (list, tuple, np.ndarray)):
            for d in dofs:
                yield from self._iter_dofs(d)

    # ==========================================================================
    # Loading & Boundary Conditions
    # ==========================================================================

    def _resolve_targets(self, node_ids):
        """Generator to normalize inputs (int vs list vs coord) into Node IDs."""
        if isinstance(node_ids, (int, np.integer)):
            yield int(node_ids)
        elif isinstance(node_ids, list):
            for nid in node_ids:
                yield int(nid)
        elif isinstance(node_ids, np.ndarray) and node_ids.size == 2:
            # Coordinate check
            nid = self.get_node_id(node_ids)
            if nid is not None:
                yield nid
            else:
                warnings.warn("Node at coordinates not found.")
        else:
            warnings.warn("Invalid node identifier provided.")

    def load_node(self, node_ids, dofs, force, fixed: bool = False):
        """Apply loads to node(s)."""
        target_vector = self.P_fixed if fixed else self.P

        for nid in self._resolve_targets(node_ids):
            for dof in self._iter_dofs(dofs):
                gidx = self._global_dof(nid, dof)
                target_vector[gidx] += force

    def add_nodal_load(self, node_id: int, force_vector: np.ndarray):
        """Apply a force vector [fx, fy, mz] to a node."""
        nid = self.get_node_id(node_id)
        if nid is None: return

        # Determine expected vector length
        if nid < len(self.node_dof_counts):
            expected = self.node_dof_counts[nid]
        else:
            expected = self.DOF_PER_NODE

        if len(force_vector) != expected:
            warnings.warn(f"Force vector len {len(force_vector)} mismatch with node DOFs {expected}")
            return

        for local_dof, val in enumerate(force_vector):
            if val != 0:
                self.load_node(nid, local_dof, val)

    def reset_loading(self):
        self.P_fixed = np.zeros(self.nb_dofs)
        self.P = np.zeros(self.nb_dofs)

    def fix_node(self, node_ids, dofs):
        """Fix DOFs on node(s)."""
        new_fixed = []
        for nid in self._resolve_targets(node_ids):
            for dof in self._iter_dofs(dofs):
                gidx = self._global_dof(nid, dof)
                new_fixed.append(gidx)

        if new_fixed:
            # Update Fixed array
            self.dof_fix = np.unique(np.append(self.dof_fix, new_fixed))
            self.nb_dof_fix = len(self.dof_fix)

            # Update Free array (Efficient set difference)
            all_dofs = np.arange(self.nb_dofs) if self.nb_dofs else np.array([])
            self.dof_free = np.setdiff1d(all_dofs, self.dof_fix)
            self.nb_dof_free = len(self.dof_free)

    def set_damping_properties(self, xsi=0.0, damp_type="RAYLEIGH", stiff_type="INIT"):
        self.xsi = [xsi, xsi] if isinstance(xsi, float) else xsi
        self.damp_type = damp_type
        self.stiff_type = stiff_type

    # ==========================================================================
    # Analysis Configuration
    # ==========================================================================

    def ask_method(self, Meth=None):
        """
        Configure dynamic time integration.
        Returns: (method_name, parameters_dict)
        """
        # 1. Normalize Input
        params_list = []
        if Meth is None:
            # Interactive Prompt
            prompt = "Method? CDM, CAA, LA, NWK, WIL, HHT, WBZ, GEN (Default: CDM): "
            inp = input(prompt).strip()
            name = inp if inp else "CDM"
        elif isinstance(Meth, str):
            name = Meth
        elif isinstance(Meth, list):
            name = Meth[0]
            params_list = Meth[1:]
        else:
            return None, None

        # 2. Logic Dispatch
        if name == "CDM":
            return "CDM", {}

        elif name in ["CAA", "NWK", "LA"]:
            # Newmark Family
            g, b = 0.5, 0.25  # Defaults for CAA

            if name == "LA":
                b = 1 / 6
            elif name == "NWK":
                if params_list:
                    g, b = params_list[0], params_list[1]
                elif Meth is None:  # Interactive
                    g_in = input("Gamma? (Def 0.5): ")
                    b_in = input("Beta? (Def 0.25): ")
                    g = float(g_in) if g_in else 0.5
                    b = float(b_in) if b_in else 0.25

            # Return generalized form or specific dict
            if name == "NWK":
                return "GEN", {"am": 0, "af": 0, "g": g, "b": b}
            else:
                return "NWK", {"g": g, "b": b}

        elif name == "WIL":
            t = 1.5
            if params_list:
                t = params_list[0]
            elif Meth is None:
                t_in = input("Theta? (Def 1.5): ")
                t = float(t_in) if t_in else 1.5

            if t < 1.37:
                warnings.warn("Wilson Theta < 1.37 is unstable")
            return "WIL", {"t": t}

        elif name == "HHT":
            a = 1 / 4  # Default alpha
            if params_list:
                a = params_list[0]
                # Optional manual overrides for g, b
                g = params_list[1] if len(params_list) > 1 else (1 + 2 * a) / 2
                b = params_list[2] if len(params_list) > 2 else (1 + a) ** 2 / 4
            elif Meth is None:
                a_in = input("Alpha? (Def 0.25): ")
                a = float(a_in) if a_in else 0.25
                g = (1 + 2 * a) / 2
                b = (1 + a) ** 2 / 4

            if not (0 <= a <= 1 / 3): warnings.warn("HHT Alpha unstable (must be 0 to 1/3)")
            return "GEN", {"am": 0, "af": a, "g": g, "b": b}

        elif name == "WBZ":
            a = 0.5
            if params_list:
                a = params_list[0]
                g = params_list[1] if len(params_list) > 1 else (1 - 2 * a) / 2
                b = params_list[2] if len(params_list) > 2 else 0.25
            elif Meth is None:
                a_in = input("Alpha? (Def 0.5): ")
                a = float(a_in) if a_in else 0.5
                g = (1 - 2 * a) / 2
                b = 0.25

            return "GEN", {"am": a, "af": 0, "g": g, "b": b}

        elif name == "GEN":
            m = 1.0
            if params_list:
                m = params_list[0]
            elif Meth is None:
                m_in = input("Mu? (Def 1.0): ")
                m = float(m_in) if m_in else 1.0

            return "GEN", {
                "am": (2 * m - 1) / (m + 1),
                "af": m / (m + 1),
                "g": (3 * m - 1) / (2 * (m + 1)),
                "b": (m / (m + 1)) ** 2
            }

        return None, None

    # ==========================================================================
    # Plotting & IO
    # ==========================================================================

    def save_structure(self, filename):
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(self, f)

    def plot(self, ax=None, show_deformed=False, deformation_scale=1.0,
             show_nodes=True, show_blocks=True, show_elements=True,
             show_contact_faces=True, show_block_labels=False, title=None,
             node_size=50, element_linewidth=None, block_linewidth=None, block_alpha=None,
             **kwargs):
        """
        Plot the structure using Visualizer.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        show_deformed : bool
            Show deformed configuration if solved.
        deformation_scale : float
            Scale factor for deformed shape.
        show_nodes, show_blocks, show_elements, show_contact_faces : bool
            Visibility flags for structure components.
        title : str, optional
            Plot title.
        node_size, element_linewidth, block_linewidth, block_alpha : float
            Style parameters.
        """
        from Core.Solvers.Visualizer import Visualizer, PlotStyle

        # Create or use provided axes
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        else:
            fig = ax.figure

        # Build a PlotStyle with GUI-provided parameters
        style = PlotStyle()
        style.node_size = node_size
        if element_linewidth is not None:
            style.fem_linewidth = element_linewidth
        if block_linewidth is not None:
            style.block_linewidth = block_linewidth
        if block_alpha is not None:
            style.block_alpha_orig = block_alpha
            style.block_alpha_def = block_alpha

        # Disable LaTeX for GUI (faster rendering)
        style.use_latex = False

        # Determine if we can show deformed
        can_show_deformed = (show_deformed and
                            hasattr(self, 'U') and
                            self.U is not None and
                            np.any(self.U != 0))

        # Detect structure type
        has_blocks = hasattr(self, 'list_blocks') and len(getattr(self, 'list_blocks', [])) > 0
        has_fes = hasattr(self, 'list_fes') and len(getattr(self, 'list_fes', [])) > 0

        # Plot blocks
        if show_blocks and has_blocks:
            Visualizer._plot_blocks(
                ax, self, style,
                scale=deformation_scale if can_show_deformed else 0,
                show_original=not can_show_deformed,
                show_deformed=can_show_deformed,
                show_nodes=show_nodes
            )

        # Plot FEM elements
        if show_elements and has_fes:
            Visualizer._plot_fem_elements(
                ax, self, style,
                scale=deformation_scale if can_show_deformed else 0,
                show_original=not can_show_deformed,
                show_deformed=can_show_deformed,
                show_nodes=show_nodes and not has_blocks  # Only show FEM nodes if no blocks
            )

        # Plot contact faces (simple line segments)
        if show_contact_faces and hasattr(self, 'list_cfs') and self.list_cfs:
            for cf in self.list_cfs:
                x1, y1 = cf.xe1
                x2, y2 = cf.xe2
                ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1, alpha=0.5)

        # Show nodes for block structures
        if show_nodes and has_blocks and not has_fes:
            for block in self.list_blocks:
                ref = block.ref_point
                if can_show_deformed:
                    dof_ux = self._global_dof(block.connect, 0)
                    dof_uy = self._global_dof(block.connect, 1)
                    ref = ref + np.array([self.U[dof_ux], self.U[dof_uy]]) * deformation_scale
                ax.scatter(ref[0], ref[1], s=node_size, c='blue', marker='s', zorder=10)

        # Autoscale and set aspect
        ax.autoscale_view()
        ax.set_aspect('equal', adjustable='box')

        # Set title if provided
        if title:
            ax.set_title(title)

        return ax
