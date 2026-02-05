# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 17:24:23 2024

@author: ibouckaert
"""
# %% Library imports

import numpy as np
import os
import h5py
import sys
import pathlib
import importlib
from copy import deepcopy
import pickle


def reload_modules():
    importlib.reload(st)
    importlib.reload(mat)
    importlib.reload(ct)
    importlib.reload(surf)


folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Material as mat
import Contact as ct
import Surface as surf

reload_modules()

# %% Structure parameters


N4 = np.array([3, 6], dtype=float)
N5 = np.array([3, 3], dtype=float)

BLOCKS = 15

RHO_b = 2000.

k = 2e9
phi = .8
# mu=10
# SURF = surf.Coulomb(k, k, 0.7)

SURF = surf.Coulomb(k, k, phi)
# %% Building structure

St = st.Structure_2D()

St.add_beam(N4, N5, BLOCKS, .2, rho=RHO_b, b=1)

St.make_nodes()
St.make_cfs(False, offset=-1, nb_cps=[-1, -.5, 0, .5, 1.], surface=SURF)

St.plot_structure(scale=0, plot_cf=False, save='build_model.eps')

# %% BCs and Forces

St.fixNode(N4, [0, 2])
St.fixNode(N5, [0, 1, 2])

St.get_M_str()

for i in range(len(St.list_blocks)):
    W = St.list_blocks[i].m * 9.81
    N = St.list_blocks[i].ref_point
    Node = St.get_node_id(N)
    St.loadNode(Node, 1, -W, fixed=True)

St.loadNode(St.get_node_id(N4), 1, -0.4 * 0.4 * 1.5 * 2500 * 9.81, fixed=True)

St.solve_forcecontrol(10)
# %% Dynamic analysis
import pandas as pd


def read_accelerogram(filename):
    df = pd.read_csv(filename, sep='\s+', header=1)
    values = df.to_numpy()

    a = values[:, :6]
    a = a.reshape(-1, 1)
    a = a[~np.isnan(a)]

    file = open(filename)
    # get the first line of the file
    line1 = file.readline()
    line2 = file.readline()
    items = line2.split(' ')
    items = np.asarray(items)
    items = items[items != '']
    dt = float(items[1])

    return (dt, a)


dt, lmbda = read_accelerogram('Earthquakes/NF13')
time = np.arange(len(lmbda)) * dt
pga = .8
lmbda = pga * lmbda / np.max(abs(lmbda))

dt_new = 2e-4
new_time = np.arange(time[0], time[-1], dt_new)

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

interpolator = interp1d(time, lmbda, kind="linear", fill_value="extrapolate")
new_lmbda = interpolator(new_time)

plt.plot(new_time, new_lmbda)
# 
v_trap = np.zeros(len(new_lmbda))
u_trap = np.zeros(len(new_lmbda))

for i in range(len(new_lmbda)):
    v_trap[i] = np.trapezoid(new_lmbda[:i] * 9.81, dx=dt_new)
    u_trap[i] = np.trapezoid(v_trap[:i], dx=dt_new)

plt.plot(new_time, u_trap)

Meth = 'CDM'

St.impose_dyn_excitation(St.get_node_id(N4), 0, u_trap, dt_new)
St.impose_dyn_excitation(St.get_node_id(N5), 0, u_trap, dt_new)

St.set_damping_properties(xsi=0.05, damp_type='STIFF', stiff_type='TAN_LG')

# St.solve_dyn_nonlinear(20, dt_new, Meth=Meth, filename=f'Response_NF13_{pga}_Isol')
# %% Plot Structure at end of simulation

St.plot_structure(scale=1, plot_cf=False, plot_forces=False)
# %%
St.save_structure(filename='Isolated_Frame')
