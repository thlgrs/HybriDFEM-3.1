import matplotlib.pyplot as plt
import h5py
import pickle
import numpy as np
import pathlib
import sys
import matplotlib as mpl


# ============================================================================
# FIXED: Removed hard-coded paths - use relative imports from Legacy package
# Original code (kept for reference):
# folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
# sys.path.append(str(folder))
# ============================================================================


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'
plt.rcParams['font.size'] = 15

from Legacy.Objects import Structure as st

FC = 32
FCT = 0.3 * FC ** (2 / 3) * 1e6
EC = 33e9

L = 1.26

blocks = [50, 60, 70, 80, 90, 100, 110, 150]
data_space = []
data_opening = []

for BL in blocks:

    file = f'TC_Danilo_{BL}BL.pkl'

    with open(file, 'rb') as f:
        St = pickle.load(f)

    file = f'TC_Danilo_{BL}BL.h5'
    with h5py.File(file, 'r') as hf:

        # Import what you need
        print(file)
        U = hf['U_conv'][()]
        P = hf['P_r_conv'][()]

    # St.plot_structure(scale=500, plot_cf=False, plot_supp=False)
    St.U = U[:, -1]
    # St.plot_structure(scale=2, plot_cf=False, plot_supp=False)
    St.get_P_r()

    eps_c_max = FCT / EC

    sigma_c, eps_c, x_c = St.get_stresses(angle=np.pi / 2, tag='CTC')
    sigma_s, eps_s, x_s = St.get_stresses(angle=np.pi / 2, tag='STC')

    mask = eps_c > eps_c_max

    location = x_c[mask]
    location = np.append(0, location)
    location = np.append(location, 1.26)

    openings = eps_c[mask] * (L / BL) * 1e3

    spacing = np.zeros(len(location) - 1)

    for i in range(len(location) - 1):
        spacing[i] = location[i + 1] - location[i]

    data_space.append(spacing * 100)
    data_opening.append(openings)
    # print(f'A total of {len(location)-2} cracks')
    # print(np.average(spacing)*100)
    # print(f'Average crack opening {np.average(openings)} mm')
    # print(f'Maximal crack opening {np.max(openings)} mm')
    # print(f'Minimal crack opening {np.min(openings)} mm')

# %% Plotting
plt.figure(None, figsize=(5, 5), dpi=600)

plt.axhspan(11.36, 22.7, facecolor='lightgrey', alpha=0.5, hatch=None)

plt.axhline(y=11.36, color='black', linestyle='--', linewidth=.5)
plt.axhline(y=17, color='black', linestyle='--', linewidth=.5)
plt.axhline(y=22.7, color='black', linestyle='--', linewidth=.5)

flier_style = dict(marker='x', markerfacecolor='black', markersize=5,
                   linestyle=None, markeredgecolor='black', linewidth=0, )
median_style = dict(marker=None, markerfacecolor=None, linewidth=2,
                    linestyle='-', markeredgecolor=None, color='black')
mean_style = dict(marker='o',  # diamond marker
                  markerfacecolor='black',
                  markeredgecolor='black',
                  markersize=5)

box = plt.boxplot(data_space,
                  vert=True,  # vertical (False = horizontal)
                  patch_artist=True,  # fill the box with color
                  showmeans=True,
                  flierprops=flier_style,
                  medianprops=median_style,
                  meanprops=mean_style)  # show the mean as a marker

for patch in box['boxes']:
    patch.set_facecolor('darkgrey')

plt.grid(linewidth=.25, color='lightgrey')
plt.xlabel('Number of blocks')
plt.ylabel('Crack spacing [cm]')
plt.ylim((0, 25))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], labels=blocks)

plt.savefig('boxplot_spacing.eps')

# %% Plotting
plt.figure(None, figsize=(5, 5), dpi=600)

# plt.axhspan(11.36, 22.7, facecolor='lightgrey', alpha=0.5, hatch=None)

plt.axhline(y=4.7, color='black', linestyle='--', linewidth=.5)
# plt.axhline(y=17, color='black', linestyle='--', linewidth=.5)
# plt.axhline(y=22.7, color='black', linestyle='--', linewidth=.5)

flier_style = dict(marker='x', markerfacecolor='black', markersize=5,
                   linestyle=None, markeredgecolor='black', linewidth=0, )
median_style = dict(marker=None, markerfacecolor=None, linewidth=2,
                    linestyle='-', markeredgecolor=None, color='black')
mean_style = dict(marker='o',  # diamond marker
                  markerfacecolor='black',
                  markeredgecolor='black',
                  markersize=5)

box = plt.boxplot(data_opening,
                  vert=True,  # vertical (False = horizontal)
                  patch_artist=True,  # fill the box with color
                  showmeans=True,
                  flierprops=flier_style,
                  medianprops=median_style,
                  meanprops=mean_style)  # show the mean as a marker

for patch in box['boxes']:
    patch.set_facecolor('darkgrey')

plt.grid(linewidth=.25, color='lightgrey')
plt.xlabel('Number of blocks')
plt.ylabel('Crack opening [mm]')
plt.ylim((0, 5))
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], labels=blocks)

plt.savefig('boxplot_opening.eps')
