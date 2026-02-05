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
from copy import deepcopy

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as ct
import Surface as surf
import ContactPair as cp
import Material as mat
import pandas as pd

N1 = np.array([0, 0], dtype=float)
N2 = np.array([1, 0], dtype=float)

CPS = 1
BLOCKS = 2

ft = 2e6
et = 1e-4
etu = 5 * et
b = 0.2

fc = 30e6
ec = 0.002
ecu = 0.01

Gc = 8.8 * np.sqrt(fc / 1e6) * 1e3
Gt = Gc / 250

l_block = 1 / (BLOCKS - 1)
gc = 2 * Gc / l_block
gt = 2 * Gt / l_block
reg = False
# gt=None
# gc=None

d_max = -5e-3

if not reg: 
    gc = None
    gt = None

file = f'{BLOCKS}Bl_{'C' if d_max < 0 else 'T'}{'_R' if reg else ''}'

CONCR = mat.KSP_concrete(fc, ec, 0.2 * fc, ecu, ft, et, etu, b, gc=gc, gt=gt)
# CONCR = mat.popovics_concrete(fc, ec, ecu, 1.5, ft, et, etu, b)
# CONCR2 = mat.KSP_concrete(.997*fc, .95*ec, 0.2*fc, ecu, ft, et, etu, b, gc=gc, gt=None)
CONCR.plot_stress_strain()

H = .2
B = .2

St = st.Structure_2D()

x = np.array([.5, 0])
y = np.array([0, .5])

vertices = np.array([N1, N1, N1, N1])
vertices[0] += -.95 * H * y + l_block * x
vertices[1] += .95 * H * y + l_block * x
vertices[2] += .95 * H * y
vertices[3] += -.95 * H * y
St.add_block(vertices, 0, B, material=CONCR, ref_point=N1)

c = N1.copy()
for i in range(1, BLOCKS):
    c += l_block * x * 2
    vertices = np.array([c, c, c, c])

    if i == BLOCKS - 1:
        vertices[0] += -H * y
        vertices[1] += H * y
        vertices[2] += H * y - l_block * x
        vertices[3] += -H * y - l_block * x
        St.add_block(vertices, 0, B, material=CONCR, ref_point=N2)
    else:
        vertices[0] += -H * y + l_block * x
        vertices[1] += H * y + l_block * x
        vertices[2] += H * y - l_block * x
        vertices[3] += -H * y - l_block * x

        St.add_block(vertices, 0, B, material=CONCR, ref_point=c)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

for i in range(1, BLOCKS - 1):
    St.fixNode(i, [1, 2])

F = 1 * np.sign(d_max)
St.fixNode(0, [0, 1, 2])
St.fixNode(BLOCKS - 1, [1, 2])

St.plot_structure(scale=1, plot_cf=True, plot_forces=True, plot_supp=True)

for i in range(1, BLOCKS - 1):
    St.fixNode(i, [1, 2])

St.loadNode(BLOCKS - 1, 0, F)

St.solve_dispcontrol(1000, d_max, BLOCKS - 1, 0, filename=file, tol=1e-4, max_iter=100)

St.plot_stresses(tag=None)
# St.plot_strains(tag=None)
# St.plot_stress_profile(save='stress_prof.eps')
St.plot_structure(scale=0, plot_cf=True, plot_forces=True, plot_supp=True)

#%% Plot Pushover
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
    P_c = -hf['P_r_conv'][3 * BLOCKS - 3, :last_conv] / 1000
    U_c = -hf['U_conv'][3 * BLOCKS - 3, :last_conv] * 1000

# file2 = 'Elastic_DispControl.h5'

# with h5py.File(file2, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     P_e = hf['P_r_conv'][3*Node,:last_conv] / W
#     U_e = hf['U_conv'][3*Node,:last_conv]*1000
import matplotlib.pyplot as plt
print(max(P_c))
plt.figure(figsize=(4.5, 4.5), dpi=600)
plt.grid(True)
plt.xlabel(r'Control displacement [mm]')
plt.ylabel(r'Load multiplier [-]')
plt.plot(-U_c, -P_c, color='black')
# plt.plot(U_e,P_e,label='Elastic')
# plt.legend()
# plt.xlim((0,40))
# plt.ylim((0, 0.08))

# plt.savefig('Dispcontrol_arch.eps')