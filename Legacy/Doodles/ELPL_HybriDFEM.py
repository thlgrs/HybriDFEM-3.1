# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 15:59:50 2025

@author: ibouckaert
"""

import numpy as np
import os
import h5py
import sys
import pathlib
import importlib


def reload_modules():
    importlib.reload(st)
    importlib.reload(ct)
    importlib.reload(cp)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Surface as surf
import ContactPair as cp

# %% Blocks
N1 = np.array([0, 0])

H = 1
B = 1

# %%
RHO = 1 / (H * B * 10)

PATTERN = [[1, 1]]

St = st.Structure_2D()

# St.add_block(vertices, RHO, b=1)
St.add_wall(N1, B, H, PATTERN, RHO, b=1., material=None)

# %% Surface law and CFs
knn = 100.;
knt = 0.;
ktn = 0.;
ktt = 50.
c = 1.;
phi = .8;
psi = 0.2
ft = 0.1
Cnn = 1.0;
Css = 9.0;
Cn = 0.;
fc = 1e10

Surf = surf.ElastoPlastic(knn * 2, ktt * 2, knt * 2, ktn * 2, c, phi, psi, ft, Cnn, Css, Cn, fc)

St.make_nodes()

# nb_cps = [0]
St.make_cfs(True, nb_cps=[-1, 0, 1], surface=Surf, offset=-1)

# %% BCs and Forces

Total_mass = 0
St.fixNode(0, [0, 1, 2])
# St.fixNode(1,[0])

W = 1.

St.loadNode(1, 0, -W, fixed=True)
St.loadNode(1, 1, W)

St.plot_structure(1)

Node = 1
d_end = -1

LIST = np.array([])
LIST = np.append(LIST, np.linspace(0, d_end, 100))
# sigma = np.array([-1, 0])
# d_tot = np.array([sigma[0]/knn, sigma[1]/ktt])
St.solve_dispcontrol(LIST.tolist(), 0, Node, 1, tol=1e-8, filename=f'Trial_ElPl', max_iter=1000)

# St.U[St.dof_free] = d_tot
# St.get_P_r()

St.plot_structure(1)

import matplotlib.pyplot as plt

file = 'Trial_ElPl.h5'
with h5py.File(file, 'r') as hf:
    U = -hf['U_conv'][-2] * 1000
    P = -hf['P_r_conv'][-2]

plt.figure(None, figsize=(6, 6))
# plt.xlim(0,-d_end*1100)
# plt.ylim(0,1)
plt.plot(U, P, '-*')

# St.commit()
