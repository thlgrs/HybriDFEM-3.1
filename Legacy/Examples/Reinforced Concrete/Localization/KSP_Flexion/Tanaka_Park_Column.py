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

CPS = 100
BLOCKS = 3

N1 = np.array([0, 0], dtype=float)
N2 = np.array([0, 1.65 / 2], dtype=float)
N3 = np.array([0, 1.65], dtype=float)

# ft = 0.33*np.sqrt(39)*1e6
# et = 2e-4
# etu = 5*et
# b = 0.4

ft = 0 
et = 0
etu = 5 * et
b = 0.0

fc = 39e6
ec = 0.0024
ecu = 0.0248

Gc = 180e3
l_elem = (N2 - N1)[1] / (BLOCKS)
gc = Gc / l_elem
reg = True

gt = None
# gc=None

d_max = 80e-3

if not reg: 
    gc = None
    gt = None

file = f'{BLOCKS}Bl_{'R_' if reg else ''}Coupled'

CONCR = mat.KSP_concrete(fc, ec, .2 * fc, ecu, ft, et, etu, b, gc=gc, gt=gt)
CONCR.plot_stress_strain()


Es = 200e9
NU = 0.2
fy = 474e6
STEEL = mat.steel_EC(Es, fy, alpha=1e-2)
# STEEL.plot_stress_strain()
r = 10e-3
A = np.pi * r **2


H = .55
B = .55
P = 0.3 * fc * H *B
# P = 0 


St = st.Structure_2D()

St.add_beam(N1, N2, BLOCKS, H, 0, B, material=CONCR)


h1 = 0.217
h2 = 0.072
EIc = CONCR.stiff0['E'] * B * H ** 3 / (12)
EI1 = 2 * Es * 4 * A * h1 ** 2
EI2 = 2 * Es * 2 * A * h2 ** 2
EItot = EIc + EI1 + EI2

EAtot = (CONCR.stiff0['E'] * H * B + Es * 12 * A)
Hb = np.sqrt(12 * EItot / EAtot)
Eb = EAtot / H
Bb = 1
NU = 0.

St.add_fe(N2, N3, Eb, NU, Hb, Bb, lin_geom=False)

St.make_nodes()
St.make_cfs(False, nb_cps=CPS)

r = 10e-3
A = np.pi * r **2
# print(A)
for cf in St.list_cfs:
    cf.add_reinforcement([-.789], 4 * A, material=STEEL, height=r)
    cf.add_reinforcement([.789], 4 * A, material=STEEL, height=r)
    cf.add_reinforcement([.262], 2 * A, material=STEEL, height=r)
    cf.add_reinforcement([-.262], 2 * A, material=STEEL, height=r)
    
# for i in range(1, BLOCKS-1): 
#     St.fixNode(i, [1,2])

F = 1e3
Node_base = St.get_node_id(N1)
St.fixNode(Node_base, [0, 1, 2])
Node = St.get_node_id(N3)
# St.fixNode(BLOCKS-1, [0,1,2])

St.loadNode(Node, 0, F)
St.loadNode(Node, 1, -P, fixed=True)
St.plot_structure(scale=1, plot_cf=True, plot_forces=True, plot_supp=True)

LIST = np.linspace(0, d_max, 1000)
# LIST = np.append(LIST, np.linspace(LIST[-1],35.5e-3,1000))
# LIST = np.append(LIST, np.linspace(LIST[-1],d_max,1000))


# St.solve_dispcontrol(List, d_max, Node, 0, filename=file,tol=10,max_iter=25)
St.solve_dispcontrol(LIST.tolist(), d_max, Node, 0, filename=file, tol=1, max_iter=100)

St.plot_stresses(tag=None)
# St.plot_strains(tag=None)
# St.plot_stress_profile(save='stress_prof.eps')
St.plot_structure(scale=1, plot_cf=False, plot_forces=True, plot_supp=True)

# %% Plot Pushover
file = f'{BLOCKS}Bl__Coupled'
file1 = file + '.h5'
import matplotlib as mpl
# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

with h5py.File(file1, 'r') as hf:
    #Import what you need
    last_conv = hf['Last_conv'][()]
    P_c = hf['P_r_conv'][3 * Node, :last_conv] / 1000
    U_c = hf['U_conv'][3 * Node, :last_conv] * 1000

file = f'{BLOCKS}Bl__R_Coupled'
file2 = file + '.h5'
with h5py.File(file2, 'r') as hf:
    #Import what you need
    last_conv = hf['Last_conv'][()]
    P_e = hf['P_r_conv'][3 * Node, :last_conv] / 1000
    U_e = hf['U_conv'][3 * Node, :last_conv] * 1000

import matplotlib.pyplot as plt
# print(max(P_c))
plt.figure(figsize=(6, 3), dpi=600)
plt.grid(True)
plt.xlabel(r'Control displacement [mm]')
plt.ylabel(r'Applied horizonal force')
plt.plot(U_c, P_c, color='black')
plt.plot(U_e, P_e, color='blue')


# plt.plot(U_e,P_e,label='Elastic')
# # plt.legend()
plt.xlim((0, 80))
plt.ylim((0, 650))

# plt.savefig('Dispcontrol_arch.eps')