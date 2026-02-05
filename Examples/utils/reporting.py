"""
Analysis Reporting Utilities
============================

Generates standardized Markdown reports and plots for analysis results.
Includes timing, configuration details, DOF counts, and comprehensive metrics.
"""

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from Examples.utils.model_builders import find_nodes_at_base


class AnalysisReport:
    """
    Comprehensive analysis report generator.

    Creates standardized markdown reports with:
    - Full configuration details
    - Timing information
    - DOF counts and structure metrics
    - Displacement results
    - Reaction forces and equilibrium checks
    - Error diagnostics
    """

    def __init__(self, St, config, start_time=None):
        """
        Initialize report with structure and config.

        Args:
            St: Solved structure object
            config: Configuration dictionary
            start_time: Optional start time for timing (time.time())
        """
        self.St = St
        self.config = config
        self.start_time = start_time
        self.end_time = None
        self.lines = []
        self.errors = []
        self.warnings = []

        # Auto-detect structure type
        self.structure_type = self._detect_structure_type()
        self.dofs_per_node = 3 if self.structure_type == 'Block' else 2

        # Initialize report sections
        self._init_report()

    def _detect_structure_type(self):
        """Detect structure type from object attributes."""
        has_blocks = hasattr(self.St, 'list_blocks') and self.St.list_blocks
        has_fem = hasattr(self.St, 'list_fes') and self.St.list_fes

        if has_blocks and has_fem:
            return 'Hybrid'
        elif has_blocks:
            return 'Block'
        elif has_fem:
            return 'FEM'
        return 'Unknown'

    def _init_report(self):
        """Initialize report with header."""
        name = self.config['io'].get('filename', 'analysis')
        self.lines.append(f"# Analysis Report: {name}")
        self.lines.append("")
        self.lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        self.lines.append("")

    def log(self, text="", console=True):
        """Add text to report and optionally print to console."""
        if console:
            print(text)
        self.lines.append(text)

    def section(self, title):
        """Add a section header."""
        self.log(f"\n## {title}")

    def subsection(self, title):
        """Add a subsection header."""
        self.log(f"\n### {title}")

    # =========================================================================
    # CONFIGURATION SECTION
    # =========================================================================

    def add_configuration(self):
        """Add full configuration details to report."""
        self.section("Configuration")

        g = self.config.get('geometry', {})
        m = self.config.get('material', {})
        e = self.config.get('elements', {})
        c = self.config.get('contact', {})
        cp = self.config.get('coupling', {})
        s = self.config.get('solver', {})
        l = self.config.get('loads', self.config.get('loading', {}))

        # Structure type
        self.log(f"- **Structure Type**: {self.structure_type}")

        # Geometry
        self.subsection("Geometry")
        width = g.get('width', g.get('x_dim', 'N/A'))
        height = g.get('height', g.get('y_dim', 'N/A'))
        thickness = g.get('thickness', 'N/A')
        self.log(f"- **Dimensions**: {width} x {height} x {thickness} m (W x H x t)")

        nx = g.get('nx', 'N/A')
        ny = g.get('ny', 'N/A')
        self.log(f"- **Mesh**: {nx} x {ny} elements/blocks")

        if 'n_slices' in g:
            self.log(f"- **Slices**: {g['n_slices']} (start: {g.get('start_with', 'block')})")

        # Elements (FEM/Hybrid only)
        if e:
            self.subsection("Elements")
            elem_type = e.get('type', 'N/A')
            elem_order = e.get('order', 'N/A')
            elem_names = {
                ('triangle', 1): 'T3 (3-node triangle)',
                ('triangle', 2): 'T6 (6-node triangle)',
                ('quad', 1): 'Q4 (4-node quad)',
                ('quad', 2): 'Q8 (8-node quad)',
                ('quad', 3): 'Q9 (9-node quad)',
            }
            elem_name = elem_names.get((elem_type, elem_order), f"{elem_type} (order {elem_order})")
            self.log(f"- **Element Type**: {elem_name}")

        # Material
        self.subsection("Material")
        if isinstance(m, dict):
            # Check for block/fem split
            if 'block' in m or 'fem' in m:
                if 'block' in m:
                    mb = m['block']
                    self.log(f"- **Block**: E = {mb.get('E', 0)/1e9:.1f} GPa, nu = {mb.get('nu', 0)}")
                if 'fem' in m:
                    mf = m['fem']
                    self.log(f"- **FEM**: E = {mf.get('E', 0)/1e9:.1f} GPa, nu = {mf.get('nu', 0)}")
            else:
                E = m.get('E', 0)
                nu = m.get('nu', 0)
                rho = m.get('rho', 0)
                self.log(f"- **E** = {E/1e9:.1f} GPa")
                self.log(f"- **nu** = {nu}")
                if rho > 0:
                    self.log(f"- **rho** = {rho} kg/m3")

        # Contact (Block/Hybrid only)
        if c:
            self.subsection("Contact")
            self.log(f"- **kn** = {c.get('kn', 0):.2e} N/m")
            self.log(f"- **ks** = {c.get('ks', 0):.2e} N/m")
            self.log(f"- **Linear Geometry**: {c.get('LG', True)}")
            self.log(f"- **Contact Points/Face**: {c.get('nb_cps', 20)}")

        # Coupling (Hybrid only)
        if cp:
            self.subsection("Coupling")
            method = cp.get('method', 'N/A').upper()
            self.log(f"- **Method**: {method}")
            self.log(f"- **Tolerance**: {cp.get('tolerance', 'N/A')}")
            if method == 'PENALTY':
                self.log(f"- **Penalty Stiffness**: {cp.get('penalty_stiffness', 0):.2e}")
            if method == 'MORTAR':
                self.log(f"- **Integration Order**: {cp.get('integration_order', 2)}")

        # Loading
        self.subsection("Loading")
        Fx = l.get('Fx', 0)
        Fy = l.get('Fy', 0)
        Mz = l.get('Mz', 0)
        if Fx != 0:
            self.log(f"- **Fx** = {Fx/1e3:.1f} kN")
        if Fy != 0:
            self.log(f"- **Fy** = {Fy/1e3:.1f} kN")
        if Mz != 0:
            self.log(f"- **Mz** = {Mz/1e3:.1f} kNm")

        # Solver
        if s:
            self.subsection("Solver")
            solver_name = s.get('name', 'linear')
            self.log(f"- **Type**: {solver_name}")
            if 'steps' in s:
                self.log(f"- **Steps**: {s['steps']}")
            if 'tol' in s:
                self.log(f"- **Tolerance**: {s['tol']}")

        return self

    # =========================================================================
    # MODEL INFORMATION SECTION
    # =========================================================================

    def add_model_info(self):
        """Add model statistics."""
        self.section("Model Statistics")

        n_nodes = len(self.St.list_nodes)
        n_dofs = self.St.nb_dofs
        n_fixed = len(self.St.dof_fix) if hasattr(self.St, 'dof_fix') else 0

        self.log(f"- **Nodes**: {n_nodes:,}")
        self.log(f"- **Total DOFs**: {n_dofs:,}")
        self.log(f"- **Fixed DOFs**: {n_fixed:,}")
        self.log(f"- **Free DOFs**: {n_dofs - n_fixed:,}")

        # Structure-specific info
        if self.structure_type == 'Block' or self.structure_type == 'Hybrid':
            if hasattr(self.St, 'list_blocks'):
                n_blocks = len(self.St.list_blocks)
                self.log(f"- **Blocks**: {n_blocks:,}")
            if hasattr(self.St, 'list_cfs'):
                n_cf = len(self.St.list_cfs)
                self.log(f"- **Contact Faces**: {n_cf:,}")

        if self.structure_type == 'FEM' or self.structure_type == 'Hybrid':
            if hasattr(self.St, 'list_fes'):
                n_fes = len(self.St.list_fes)
                self.log(f"- **Finite Elements**: {n_fes:,}")

        # Coupling info for hybrid
        if self.structure_type == 'Hybrid':
            if hasattr(self.St, 'coupling_T') and self.St.coupling_T is not None:
                self.log(f"- **Coupling**: Constraint (reduced system)")
            if hasattr(self.St, 'coupling_pairs'):
                n_pairs = len(self.St.coupling_pairs) if self.St.coupling_pairs else 0
                self.log(f"- **Coupling Pairs**: {n_pairs}")

        return self

    # =========================================================================
    # DISPLACEMENT SECTION
    # =========================================================================

    def add_displacements(self, control_node=None, control_dof=None, target=None):
        """Add displacement results."""
        self.section("Displacements")

        # Expand if constraint coupling
        if hasattr(self.St, 'coupling_T') and self.St.coupling_T is not None:
            self.St.U = self.St.coupling_T @ self.St.U
            self.log("*(Expanded from reduced DOF system)*")

        # Control node displacements
        if control_node is not None:
            self.subsection(f"Control Node {control_node}")
            dofs = self.St.get_dofs_from_node(control_node)

            ux = self.St.U[dofs[0]]
            uy = self.St.U[dofs[1]]

            self.log(f"- **ux** = {ux:.6e} m ({ux*1000:.4f} mm)")
            self.log(f"- **uy** = {uy:.6e} m ({uy*1000:.4f} mm)")

            if len(dofs) > 2:  # Block node with rotation
                rz = self.St.U[dofs[2]]
                self.log(f"- **rz** = {rz:.6e} rad ({np.degrees(rz):.4f} deg)")

            # Target check for displacement control
            if target is not None and control_dof is not None:
                achieved = self.St.U[dofs[control_dof]]
                error = abs(achieved - target) / (abs(target) + 1e-12) * 100
                self.log(f"- **Target**: {target:.6e} m, **Achieved**: {achieved:.6e} m")
                self.log(f"- **Error**: {error:.4f}%")

        # Maximum displacements
        self.subsection("Maximum Values")

        if self.dofs_per_node == 3:  # Block structure
            ux_all = self.St.U[0::3]
            uy_all = self.St.U[1::3]
            rz_all = self.St.U[2::3]

            self.log(f"- **Max |ux|** = {np.max(np.abs(ux_all)):.6e} m")
            self.log(f"- **Max |uy|** = {np.max(np.abs(uy_all)):.6e} m")
            self.log(f"- **Max |rz|** = {np.max(np.abs(rz_all)):.6e} rad")
        else:  # FEM structure
            ux_all = self.St.U[0::2]
            uy_all = self.St.U[1::2]

            self.log(f"- **Max |ux|** = {np.max(np.abs(ux_all)):.6e} m")
            self.log(f"- **Max |uy|** = {np.max(np.abs(uy_all)):.6e} m")

        self.log(f"- **Max |U|** = {np.max(np.abs(self.St.U)):.6e} m")

        return self

    # =========================================================================
    # REACTION FORCES SECTION
    # =========================================================================

    def add_reactions(self):
        """Add reaction force results."""
        self.section("Reaction Forces")

        # Compute reactions if not present
        if not hasattr(self.St, 'P_r') or self.St.P_r is None:
            if hasattr(self.St, 'coupling_T') and self.St.coupling_T is not None:
                self.St.P_r = np.zeros(len(self.St.U), dtype=float)
                if hasattr(self.St, '_get_P_r_block'):
                    self.St._get_P_r_block()
                if hasattr(self.St, '_get_P_r_fem'):
                    self.St._get_P_r_fem()
                if hasattr(self.St, '_get_P_r_hybrid'):
                    self.St._get_P_r_hybrid()
            elif hasattr(self.St, 'get_P_r'):
                self.St.get_P_r()

        # Find base nodes
        base_nodes = find_nodes_at_base(self.St, tolerance=1e-5)

        total_Rx = 0.0
        total_Ry = 0.0
        total_Mz = 0.0

        for node_id in base_nodes:
            dofs = self.St.get_dofs_from_node(node_id)
            coords = self.St.list_nodes[node_id]

            if hasattr(self.St, 'P_r') and self.St.P_r is not None:
                Rx = self.St.P_r[dofs[0]]
                Ry = self.St.P_r[dofs[1]]
            elif hasattr(self.St, 'P'):
                Rx = self.St.P[dofs[0]]
                Ry = self.St.P[dofs[1]]
            else:
                Rx = Ry = 0.0

            total_Rx += Rx
            total_Ry += Ry

            # Moment contribution
            total_Mz += Ry * coords[0] - Rx * coords[1]

            # Direct moment DOF for blocks
            if len(dofs) > 2:
                if hasattr(self.St, 'P_r') and self.St.P_r is not None:
                    total_Mz += self.St.P_r[dofs[2]]
                elif hasattr(self.St, 'P'):
                    total_Mz += self.St.P[dofs[2]]

        # Store for equilibrium check
        self.total_Rx = total_Rx
        self.total_Ry = total_Ry
        self.total_Mz = total_Mz

        self.log(f"**Total Base Reactions:**")
        self.log(f"- **Rx** = {total_Rx:+.2f} N ({total_Rx/1e3:+.3f} kN)")
        self.log(f"- **Ry** = {total_Ry:+.2f} N ({total_Ry/1e3:+.3f} kN)")
        self.log(f"- **Mz** = {total_Mz:+.2f} Nm ({total_Mz/1e3:+.3f} kNm)")

        return self

    # =========================================================================
    # EQUILIBRIUM CHECK SECTION
    # =========================================================================

    def check_equilibrium(self, applied_Fx=None, applied_Fy=None, applied_Mz=0):
        """Check force equilibrium."""
        self.section("Equilibrium Check")

        # Get applied loads from config if not provided
        l = self.config.get('loads', self.config.get('loading', {}))
        if applied_Fx is None:
            applied_Fx = l.get('Fx', 0)
        if applied_Fy is None:
            applied_Fy = l.get('Fy', 0)
        if applied_Mz == 0:
            applied_Mz = l.get('Mz', 0)

        # Get reaction totals
        Rx = getattr(self, 'total_Rx', 0)
        Ry = getattr(self, 'total_Ry', 0)

        # Calculate errors
        err_x = abs(applied_Fx + Rx)
        err_y = abs(applied_Fy + Ry)

        # Relative errors
        rel_err_x = err_x / (abs(applied_Fx) + 1e-12) if abs(applied_Fx) > 1e-6 else err_x
        rel_err_y = err_y / (abs(applied_Fy) + 1e-12) if abs(applied_Fy) > 1e-6 else err_y

        self.log(f"| Direction | Applied | Reaction | Abs Error | Rel Error | Status |")
        self.log(f"|-----------|---------|----------|-----------|-----------|--------|")

        status_x = "OK" if rel_err_x < 1e-3 else "WARN"
        status_y = "OK" if rel_err_y < 1e-3 else "WARN"

        self.log(f"| Fx | {applied_Fx/1e3:+.2f} kN | {Rx/1e3:+.2f} kN | {err_x:.2e} N | {rel_err_x:.2e} | {status_x} |")
        self.log(f"| Fy | {applied_Fy/1e3:+.2f} kN | {Ry/1e3:+.2f} kN | {err_y:.2e} N | {rel_err_y:.2e} | {status_y} |")

        if status_x != "OK":
            self.warnings.append(f"Equilibrium error in X: {rel_err_x:.2e}")
        if status_y != "OK":
            self.warnings.append(f"Equilibrium error in Y: {rel_err_y:.2e}")

        return self

    # =========================================================================
    # TIMING SECTION
    # =========================================================================

    # =========================================================================
    # STRESS SECTION (FEM ONLY)
    # =========================================================================

    def add_stresses(self):
        """Add maximum stress results for FEM structures."""
        # Only applicable to FEM structures
        if self.structure_type not in ['FEM', 'Hybrid']:
            return self

        if not hasattr(self.St, 'compute_nodal_directional_stress'):
            self.warnings.append("Stress computation not available for this structure type")
            return self

        self.section("Stresses")

        try:
            # Horizontal stress (sigma_x): plane with vertical tangent (angle = pi/2)
            sigma_x, _ = self.St.compute_nodal_directional_stress(np.pi / 2)

            # Vertical stress (sigma_y) and shear (tau_xy): plane with horizontal tangent (angle = 0)
            sigma_y, tau_xy = self.St.compute_nodal_directional_stress(0)

            # Convert to MPa for display
            sigma_x_MPa = sigma_x / 1e6
            sigma_y_MPa = sigma_y / 1e6
            tau_xy_MPa = tau_xy / 1e6

            # Compute extrema
            max_sigma_x = np.max(sigma_x_MPa)
            min_sigma_x = np.min(sigma_x_MPa)
            max_sigma_y = np.max(sigma_y_MPa)
            min_sigma_y = np.min(sigma_y_MPa)
            max_tau_xy = np.max(tau_xy_MPa)
            min_tau_xy = np.min(tau_xy_MPa)

            self.subsection("Maximum Stress Values")
            self.log(f"| Component | Min (MPa) | Max (MPa) |")
            self.log(f"|-----------|-----------|-----------|")
            self.log(f"| sigma_x (horizontal) | {min_sigma_x:+.4f} | {max_sigma_x:+.4f} |")
            self.log(f"| sigma_y (vertical) | {min_sigma_y:+.4f} | {max_sigma_y:+.4f} |")
            self.log(f"| tau_xy (shear) | {min_tau_xy:+.4f} | {max_tau_xy:+.4f} |")

        except Exception as e:
            self.warnings.append(f"Stress computation failed: {str(e)}")

        return self

    # =========================================================================
    # TIMING SECTION
    # =========================================================================

    def add_timing(self, phase_times=None):
        """Add timing information."""
        self.section("Performance")

        if self.start_time is not None:
            if self.end_time is None:
                self.end_time = time.time()
            total_time = self.end_time - self.start_time

            self.log(f"- **Total Time**: {total_time:.2f} s")

        if phase_times:
            self.log("")
            self.log("| Phase | Time (s) | % of Total |")
            self.log("|-------|----------|------------|")
            total = sum(phase_times.values())
            for phase, t in phase_times.items():
                pct = 100 * t / total if total > 0 else 0
                self.log(f"| {phase} | {t:.3f} | {pct:.1f}% |")

        return self

    # =========================================================================
    # ERRORS/WARNINGS SECTION
    # =========================================================================

    def add_diagnostics(self):
        """Add error and warning diagnostics."""
        if self.errors or self.warnings:
            self.section("Diagnostics")

            if self.errors:
                self.log("**Errors:**")
                for err in self.errors:
                    self.log(f"- {err}")

            if self.warnings:
                self.log("**Warnings:**")
                for warn in self.warnings:
                    self.log(f"- {warn}")

        return self

    # =========================================================================
    # SAVE
    # =========================================================================

    def save(self):
        """Save report to markdown file."""
        io_conf = self.config['io']
        os.makedirs(io_conf['dir'], exist_ok=True)
        md_path = os.path.join(io_conf['dir'], io_conf['filename'] + ".md")

        # Add diagnostics at end if any
        self.add_diagnostics()

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))

        print(f"\n[OK] Report saved to: {md_path}")
        return md_path

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def full_report(self, control_node=None, control_dof=None, target=None,
                    applied_Fx=None, applied_Fy=None, phase_times=None):
        """Generate a full report with all sections."""
        self.add_configuration()
        self.add_model_info()
        self.add_displacements(control_node, control_dof, target)
        self.add_stresses()
        self.add_reactions()
        self.check_equilibrium(applied_Fx, applied_Fy)
        self.add_timing(phase_times)
        return self


def plot_history(St, config, control_node):
    """Standard history plotter for nonlinear analysis."""
    if not HAS_H5PY or not HAS_MATPLOTLIB:
        return  # Required libraries not available

    io = config['io']
    if 'control' not in config:
        return  # Only for disp/force control

    ctrl = config.get('control', {})
    dof = ctrl.get('dof', 0)

    h5_path = os.path.join(io['dir'], io['filename'] + '.h5')
    if not os.path.exists(h5_path):
        return

    dof_idx = St._global_dof(control_node, dof)

    with h5py.File(h5_path, 'r') as f:
        if 'U_conv' not in f:
            return

        U_hist = f['U_conv'][:]
        lambdas = f['LoadFactor_conv'][:]

        disp = U_hist[dof_idx, :]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(disp, lambdas, '-o')
        ax.set_xlabel("Displacement (m)")
        ax.set_ylabel("Load Factor")
        ax.set_title("Load-Displacement Curve")
        ax.grid(True)

        save_path = os.path.join(io['dir'], io['filename'] + '_history.png')
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"  Saved history plot: {save_path}")
