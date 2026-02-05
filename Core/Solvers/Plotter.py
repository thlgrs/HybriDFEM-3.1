import os

from Core.Structures.Structure_2D import *


class Plotter:
    """
    Advanced plotting utilities for HybriDFEM-3 structures.

    This class provides detailed visualization capabilities for stress analysis,
    modal analysis, and advanced deformed shape rendering. It supports structures
    with variable DOFs per node (blocks with 3 DOFs, triangles with 2 DOFs).

    For most use cases, consider using:
    - Visualizer class for simple visualization
    - structure.plot() methods for direct plotting

    This class is best suited for:
    - Stress contour visualization
    - Modal analysis results
    - Advanced customization needs

    Example:
        from Core.Solvers.Plotter import Plotter
        
        # Plot stress distribution
        Plotter.plot_stresses(St_solved)
        
        # Plot modal shapes
        Plotter.plot_modes(St_solved, modes=5, scale=10)
        
        # Plot deformed structure with loads and supports
        Plotter.plot_def_structure(St_solved, scale=10, plot_forces=True)
    """
    def __init__(self, save=None, angle=None, tag=None, cf_index=0, scale=0, plot_cf=True,
                 plot_forces=True, plot_supp=True, lighter=False, modes=None, lims=None, folder=None, show=True):
        self.save = save
        self.angle = angle
        self.tag = tag
        self.cf_index = cf_index
        self.scale = scale
        self.plot_cf = plot_cf
        self.plot_forces = plot_forces
        self.plot_supp = plot_supp
        self.lighter = lighter
        self.modes = modes
        self.lims = lims
        self.folder = folder
        self.show = show

    @staticmethod
    def _get_node_info(structure, global_dof):
        """
        Extract node ID and local DOF from global DOF index (variable-DOF aware).

        This helper method correctly handles structures with variable DOFs per node
        (e.g., blocks with 3 DOFs, triangles with 2 DOFs).

        Args:
            structure: Structure_2D instance
            global_dof: Global DOF index

        Returns:
            tuple: (node_id, local_dof_index)
                - node_id: Index into structure.list_nodes
                - local_dof_index: 0=ux, 1=uy, 2=rotation (if applicable)

        Example:
            node_id, local_dof = Plotter._get_node_info(St, 15)
            print(f"DOF 15 belongs to node {node_id}, local DOF {local_dof}")
        """
        # Use binary search to find which node this DOF belongs to
        node_id = np.searchsorted(structure.node_dof_offsets[1:], global_dof, side='right')
        local_dof = global_dof - structure.node_dof_offsets[node_id]
        return node_id, local_dof

    @staticmethod
    def _get_node_displacement(structure, node_id, scale=1.0):
        """
        Get [ux, uy] displacement for a node (variable-DOF aware).

        This method extracts the translational displacements for any node,
        regardless of whether it has 2 or 3 DOFs.

        Args:
            structure: Structure_2D instance
            node_id: Index into structure.list_nodes
            scale: Scale factor for displacement (default: 1.0)

        Returns:
            np.ndarray: [ux, uy] displacement array (scaled)

        Note:
            Always returns first 2 DOFs (ux, uy) since these are translation
            components for both 2-DOF and 3-DOF nodes.
        """
        base_dof = structure.node_dof_offsets[node_id]
        ux = structure.U[base_dof + 0] * scale
        uy = structure.U[base_dof + 1] * scale
        return np.array([ux, uy])

    def stiffness(self, structure):
        E = []
        vertices = []

        for j, CF in enumerate(structure.list_cfs):
            for i, CP in enumerate(CF.cps):
                E.append(np.around(CP.sp1.law.stiff["E"], 3))
                E.append(np.around(CP.sp2.law.stiff["E"], 3))
                vertices.append(CP.vertices_fibA)
                vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):
            if (smax - smin) == 0 and smax < 0:
                return Normalize(
                    vmin=1.1 * smin / 1e9, vmax=0.9 * smax / 1e9, clip=False
                )
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(
                    vmin=0.9 * smin / 1e9, vmax=1.1 * smax / 1e9, clip=False
                )
            else:
                return Normalize(vmin=smin / 1e9, vmax=smax / 1e9, clip=False)

        def plot(stiff, vertex):
            smax = np.max(stiff)
            smin = np.min(stiff)

            plt.axis("equal")
            plt.axis("off")
            plt.title("Axial stiffness [GPa]")

            norm = normalize(smax, smin)
            cmap = cm.get_cmap("coolwarm", 200)

            for i in range(len(stiff)):
                if smax - smin == 0:
                    index = norm(np.around(stiff[i], 6) / 1e9)
                else:
                    index = norm(np.around(stiff[i], 6) / 1e9)
                color = cmap(index)
                vertices_x = np.append(vertex[i][:, 0], vertex[i][0, 0])
                vertices_y = np.append(vertex[i][:, 1], vertex[i][0, 1])
                plt.fill(vertices_x, vertices_y, color=color)

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(plt.gca())

            cax = divider.append_axes("right", size="10%", pad=0.2)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        plt.figure()

        plot(E, vertices)

        if self.save is not None:
            plt.savefig(self.save)

    def get_stresses(self, structure):
        # Compute maximal stress and minimal stress:

        eps = np.array([])
        sigma = np.array([])
        x_s = np.array([])

        for j, CF in enumerate(structure.list_cfs):
            if (self.angle is None) or (abs(CF.angle - self.angle) < 1e-6):
                for i, CP in enumerate(CF.cps):
                    if not CP.to_ommit():
                        if self.tag is None or CP.sp1.law.tag == self.tag:
                            eps = np.append(eps, np.around(CP.sp1.law.strain["e"], 12))
                            sigma = np.append(
                                sigma, np.around(CP.sp1.law.stress["s"], 12)
                            )
                            x_s = np.append(x_s, CP.x_cp[0])
        return sigma, eps, x_s

    def stresses(self, structure):
        # Compute maximal stress and minimal stress:

        tau = []
        sigma = []
        vertices = []

        for j, CF in enumerate(structure.list_cfs):
            if (self.angle is None) or (abs(CF.angle - self.angle) < 1e-6):
                for i, CP in enumerate(CF.cps):
                    if not CP.to_ommit():
                        if self.tag is None or CP.sp1.law.tag == self.tag:
                            tau.append(np.around(CP.sp1.law.stress["t"], 12))
                            tau.append(np.around(CP.sp2.law.stress["t"], 12))
                            sigma.append(np.around(CP.sp1.law.stress["s"], 12))
                            sigma.append(np.around(CP.sp2.law.stress["s"], 12))
                            vertices.append(CP.vertices_fibA)
                            vertices.append(CP.vertices_fibB)

        from matplotlib.colors import Normalize
        from matplotlib import cm

        def normalize(smax, smin):
            if (smax - smin) == 0 and smax < 0:
                return Normalize(
                    vmin=1.1 * smin / 1e6, vmax=0.9 * smax / 1e6, clip=False
                )
            elif (smax - smin) == 0 and smax == 0:
                return Normalize(vmin=-1e-6, vmax=1e-6, clip=False)
            elif (smax - smin) == 0:
                return Normalize(
                    vmin=0.9 * smin / 1e6, vmax=1.1 * smax / 1e6, clip=False
                )
            else:
                return Normalize(vmin=smin / 1e6, vmax=smax / 1e6, clip=False)

        def plot(stress, vertex, name_stress=None):
            smax = np.max(stress)
            smin = np.min(stress)

            print(
                f"Maximal {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smax / 1e6, 3)} MPa"
            )
            print(
                f"Minimum {'axial' if name_stress == 'sigma' else 'shear'} stress is {np.around(smin / 1e6, 3)} MPa"
            )
            # Plot sigmas

            plt.axis("equal")
            plt.axis("off")
            plt.title(
                f"{'Axial' if name_stress == 'sigma' else 'Shear'} stresses [MPa]"
            )

            norm = normalize(smax, smin)
            cmap = cm.get_cmap("viridis", 200)

            for i in range(len(sigma)):
                if smax - smin == 0:
                    index = norm(np.around(stress[i], 6) / 1e6)
                else:
                    index = norm(np.around(stress[i], 6) / 1e6)
                color = cmap(index)
                vertices_x = np.append(vertex[i][:, 0], vertex[i][0, 0])
                vertices_y = np.append(vertex[i][:, 1], vertex[i][0, 1])
                plt.fill(vertices_x, vertices_y, color=color)

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(plt.gca())

            cax = divider.append_axes("right", size="10%", pad=0.2)
            plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

        plt.figure()

        plt.subplot(2, 1, 1)
        plot(sigma, vertices, name_stress="sigma")
        plt.subplot(2, 1, 2)
        plot(tau, vertices, name_stress="tau")

        if self.save is not None:
            plt.savefig(self.save)

    def stress_profile(self, structure):
        stresses = []
        x = []
        counter = 0
        for cp in structure.list_cfs[self.cf_index].cps:
            counter += 1
            if not cp.to_ommit():
                stresses.append(cp.sp1.law.stress["s"] / 1e6)
                x.append(cp.x_cp[1] * 100)

        offset = 0.5 / (2 * len(stresses))
        # x = np.linspace(-.25+offset,0.25-offset,len(stresses))

        # x2 = np.linspace(-.25,0.25,100)
        # y_sigma = np.linspace(-36,36,100)
        # y_tau = 6 * 100e3 * (0.25**2 - (x2)**2) / (0.5**3 * 0.2)
        # # y_tau = - (5*100e3) / (0.5**2 * 0.5 * 0.2) * x2**2 + 5 * 100e3 / (4*0.5*0.2)
        plt.figure(None, figsize=(5, 5), dpi=600)
        # plt.scatter(x*100, stresses, label='HybriDFEM', marker='.', color='blue')
        plt.bar(
            x,
            stresses,
            label="HybriDFEM",
            facecolor="white",
            edgecolor="blue",
            linewidth=1,
            width=50 / counter,
        )
        plt.legend(fontsize=12)
        plt.ylabel(r"Stress [MPa]")
        plt.xlabel(r"Height [cm]")
        plt.grid(True, linestyle="--", linewidth=0.3)

        if self.save:
            plt.savefig(self.save)

    def def_structure(self, structure):
        # structure.get_P_r()

        # Plot blocks if structure has them (Structure_Block or Hybrid)
        if hasattr(structure, 'list_blocks'):
            for block in structure.list_blocks:
                block.disps = structure.U[block.dofs]
                block.plot_block(scale=self.scale, lighter=self.lighter)

        # Plot FEM elements if structure has them (Structure_FEM or Hybrid)
        if hasattr(structure, 'list_fes'):
            for fe in structure.list_fes:
                if self.scale == 0:
                    fe.PlotUndefShapeElem()
                else:
                    defs = structure.U[fe.dofs]
                    fe.PlotDefShapeElem(defs, scale=self.scale)

        # for cf in structure.list_cfs:
        #     if cf.cps[0].sp1.law.tag == 'CTC':
        #         if cf.cps[0].sp1.law.cracked:
        #             disp1 = structure.U[cf.bl_A.dofs[0]]
        #             disp2 = structure.U[cf.bl_B.dofs[0]]
        #             cf.plot_cf(scale, disp1, disp2)

        # Plot contact faces if structure has them (Structure_Block or Hybrid)
        if self.plot_cf and hasattr(structure, 'list_cfs'):
            for cf in structure.list_cfs:
                cf.plot_cf(self.scale)

        if self.plot_forces:
            for i in structure.dof_free:
                if structure.P[i] != 0:
                    # Variable DOF-aware extraction
                    node_id, dof = self._get_node_info(structure, i)

                    # Get node position with displacement
                    start = structure.list_nodes[node_id] + self._get_node_displacement(structure, node_id, self.scale)
                    arr_len = 0.3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(structure.P[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="green",
                            ec="green",
                        )
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(structure.P[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="green",
                            ec="green",
                        )
                    else:
                        if np.sign(structure.P[i]) == 1:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="green",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker=".",
                                markerfacecolor="green",
                                markeredgecolor="green",
                                markersize=5,
                            )
                        else:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="green",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker="x",
                                markerfacecolor="None",
                                markeredgecolor="green",
                                markersize=10,
                            )

                if structure.P_fixed[i] != 0:
                    # Variable DOF-aware extraction
                    node_id, dof = self._get_node_info(structure, i)

                    # Get node position with displacement
                    start = structure.list_nodes[node_id] + self._get_node_displacement(structure, node_id, self.scale)
                    arr_len = 0.3

                    if dof == 0:
                        end = arr_len * np.array([1, 0]) * np.sign(structure.P_fixed[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="red",
                            ec="red",
                        )
                    elif dof == 1:
                        end = arr_len * np.array([0, 1]) * np.sign(structure.P_fixed[i])
                        plt.arrow(
                            start[0],
                            start[1],
                            end[0],
                            end[1],
                            head_width=0.05,
                            head_length=0.075,
                            fc="red",
                            ec="red",
                        )
                    else:
                        if np.sign(structure.P_fixed[i]) == 1:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="red",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker=".",
                                markerfacecolor="red",
                                markeredgecolor="red",
                                markersize=5,
                            )
                        else:
                            plt.plot(
                                start[0],
                                start[1],
                                marker="o",
                                markerfacecolor="None",
                                markeredgecolor="red",
                                markersize=10,
                            )
                            plt.plot(
                                start[0],
                                start[1],
                                marker="x",
                                markerfacecolor="None",
                                markeredgecolor="red",
                                markersize=10,
                            )

        if self.plot_supp:
            for fix in structure.dof_fix:
                # Variable DOF-aware extraction
                node_id, dof = self._get_node_info(structure, fix)

                # Get node position with displacement
                node = structure.list_nodes[node_id] + self._get_node_displacement(structure, node_id, self.scale)

                import matplotlib as mpl

                if dof == 0:
                    mark = mpl.markers.MarkerStyle(marker=5)
                elif dof == 1:
                    mark = mpl.markers.MarkerStyle(marker=6)
                else:
                    mark = mpl.markers.MarkerStyle(marker="x")

                plt.plot(node[0], node[1], marker=mark, color="blue", markersize=8)

    def modes(self, structure):
        if not hasattr(structure, "eig_modes"):
            warnings.warn("Eigen modes were not determined yet")

        if self.modes is None:
            modes = structure.nb_dof_free

        if len(structure.eig_vals) < modes:
            warnings.warn("Asking for too many modes, fewer were computed")

        for i in range(modes):
            structure.U[structure.dof_free] = structure.eig_modes.T[i]

            if self.lims is None:
                plt.figure(None, dpi=400, figsize=(6, 6))
            else:
                x_len = self.lims[0][1] - self.lims[0][0]
                y_len = self.lims[1][1] - self.lims[1][0]
                if x_len > y_len:
                    plt.figure(None, dpi=400, figsize=(6, 6 * y_len / x_len))
                else:
                    plt.figure(None, dpi=400, figsize=(6 * x_len / y_len, 6))

            plt.axis("equal")
            plt.axis("off")

            self.def_structure(structure)

            if self.lims is not None:
                plt.xlim(self.lims[0][0], self.lims[0][1])
                plt.ylim(self.lims[1][0], self.lims[1][1])

            w = np.around(structure.eig_vals[i], 3)
            f = np.around(structure.eig_vals[i] / (2 * np.pi), 3)
            if not w == 0:
                T = np.around(2 * np.pi / w, 3)
            else:
                T = float("inf")
            plt.title(
                rf"$\omega_{{{i + 1}}} = {w}$ rad/s - $T_{{{i + 1}}} = {T}$ s - $f_{{{i + 1}}} = {f}$ "
            )
            if self.save:
                if self.folder is not None:
                    if not os.path.exists(self.folder):
                        os.makedirs(self.folder)

                    plt.savefig(self.folder + f"/Mode_{i + 1}.eps")
                else:
                    plt.savefig(f"Mode_{i + 1}.eps")

            if not self.show:
                plt.close()
            else:
                plt.show()

    def structure(self, structure):
        desired_aspect = 1.0

        if self.lims is not None:
            x0, x1 = self.lims[0][0], self.lims[0][1]
            xrange = x1 - x0
            y0, y1 = self.lims[1][0], self.lims[1][1]
            yrange = y1 - y0
            aspect = xrange / yrange

            if aspect > desired_aspect:
                center_y = (y0 + y1) / 2
                yrange_new = xrange
                y0 = center_y - yrange_new / 2
                y1 = center_y + yrange_new / 2
            else:
                center_x = (x0 + x1) / 2
                xrange_new = yrange
                x0 = center_x - xrange_new / 2
                x1 = center_x + xrange_new / 2

        plt.figure(None, dpi=400, figsize=(6, 6))

        # plt.axis('equal')
        plt.axis("off")

        self.def_structure(structure)

        if self.lims is not None:
            plt.xlim((x0, x1))
            plt.ylim((y0, y1))

        if self.save is not None:
            plt.savefig(self.save)

        if not self.show:
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_stiffness(structure, save=None):
        plotter = Plotter(save=save)
        return plotter.stiffness(structure)

    @staticmethod
    def get_stresses(structure, angle=None, tag=None):
        plotter = Plotter(angle=angle, tag=tag)
        return plotter.get_stresses(structure)

    @staticmethod
    def plot_stresses(structure, angle=None, save=None, tag=None):
        plotter = Plotter(angle=angle, tag=tag)
        return plotter.stresses(structure)

    @staticmethod
    def plot_stress_profile(structure, cf_index=0, save=None):
        plotter = Plotter(cf_index=cf_index, save=save)
        return plotter.stress_profile(structure)

    @staticmethod
    def plot_def_structure(structure, scale=0, plot_cf=True, plot_forces=True,
                           plot_supp=True, lighter=False):
        plotter = Plotter(scale=scale, plot_cf=plot_cf, plot_forces=plot_forces, plot_supp=plot_supp, lighter=lighter)
        return plotter.def_structure(structure)

    @staticmethod
    def plot_modes(structure, modes=None, scale=1, save=False, lims=None, folder=None,
                   show=True):
        plotter = Plotter(modes=modes, scale=scale, save=save, lims=lims, folder=folder, show=show)
        return plotter.modes(structure)

    @staticmethod
    def plot_structure(structure, scale=0, plot_cf=True, plot_forces=True,
                       plot_supp=True, show=True, save=None, lims=None):
        plotter = Plotter(scale=scale, plot_cf=plot_cf, plot_forces=plot_forces, show=show)
        return plotter.structure(structure)
