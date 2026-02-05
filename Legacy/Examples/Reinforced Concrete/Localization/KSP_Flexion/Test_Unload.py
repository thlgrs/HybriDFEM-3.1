# %% -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""
import numpy as np
import os
import h5py
import sys
import pathlib
import importlib


# ============================================================================
# FIXED: Removed hard-coded paths - use relative imports from Legacy package
# Original code (kept for reference):
# folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
# sys.path.append(str(folder))
# ============================================================================


from Legacy.Objects import Structure as st
from Legacy.Objects import Contact as ct
from Legacy.Objects import Surface as surf
from Legacy.Objects import ContactPair as cp
from Legacy.Objects import Material as mat
import pandas as pd

# From Tanaka and Park 1990

CPS = 2
BLOCKS = 2

N1 = np.array([0, 0], dtype=float)
N2 = np.array([1, 0], dtype=float)

ft = 0
et = 0
etu = 5 * et
b = 0.0

fc = 39e6
ec = 0.0024
ecu = 0.0248

Gc = 180e3
l_elem = (N2 - N1)[1] / (2 * BLOCKS)
gc = Gc / l_elem
reg = False
gt = None
# gc=None

d_max = 20e-3

if not reg:
    gc = None
    gt = None

# file = f'{BLOCKS}Bl_{'R_' if reg else ''}Coupled'

CONCR = mat.KSP_concrete(fc, ec, .2 * fc, ecu, ft, et, etu, b, gc=gc, gt=gt)
# CONCR.plot_stress_strain()


H = 1
B = 1
N = 1  # P = 0

St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 0, B, material=CONCR)

St.make_nodes()
St.make_cfs(False, nb_cps=CPS)

F = 1e3
Node_base = St.get_node_id(N1)
St.fixNode(Node_base, [0, 1, 2])
Node = St.get_node_id(N2)

St.loadNode(Node, 0, F)

St.plot_structure(scale=1, plot_cf=True, plot_forces=True, plot_supp=True)

LIST = np.linspace(0, -d_max, 100)
LIST = np.append(LIST, np.linspace(LIST[-1], 0, 100))

# St.solve_dispcontrol(List, d_max, Node, 0, filename=file,tol=10,max_iter=25)
St.solve_dispcontrol(LIST.tolist(), d_max, Node, 0, tol=1, max_iter=100)

St.plot_stresses(tag=None)
# St.plot_strains(tag=None)
# St.plot_stress_profile(save='stress_prof.eps')
St.plot_structure(scale=1, plot_cf=False, plot_forces=True, plot_supp=True)

# %% Plot Pushover

import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

file1 = 'Results_DispControl.h5'
with h5py.File(file1, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    P_c = hf['P_r_conv'][3 * Node, :last_conv] / 1000
    U_c = hf['U_conv'][3 * Node, :last_conv] * 1000

import matplotlib.pyplot as plt

# print(max(P_c))
plt.figure(figsize=(6, 6), dpi=600)
plt.grid(True)
plt.xlabel(r'Control displacement [mm]')
plt.ylabel(r'Applied horizonal force')
plt.plot(U_c, P_c, color='black')

# plt.plot(U_e,P_e,label='Elastic')
# # plt.legend()
# plt.xlim((0,80))
# plt.ylim((0, 650))

# plt.savefig('Dispcontrol_arch.eps')
