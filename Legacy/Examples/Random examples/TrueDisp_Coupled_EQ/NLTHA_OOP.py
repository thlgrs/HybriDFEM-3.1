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

N1 = np.array([0, 0], dtype=float)
N12_1 = np.array([0, 1], dtype=float)
N12_2 = np.array([0, 2], dtype=float)
N2 = np.array([0, 3], dtype=float)
N23_1 = np.array([0, 4], dtype=float)
N23_2 = np.array([0, 5], dtype=float)
N3 = np.array([0, 6], dtype=float)
N34_1 = np.array([1, 6], dtype=float)
N34_2 = np.array([2, 6], dtype=float)
N4 = np.array([3, 6], dtype=float)
N5 = np.array([3, 3], dtype=float)
N56_1 = np.array([3, 2], dtype=float)
N56_2 = np.array([3, 1], dtype=float)
N6 = np.array([3, 0], dtype=float)
N25_1 = np.array([1, 3], dtype=float)
N25_2 = np.array([2, 3], dtype=float)

B = .4
H = .4

BLOCKS = 15

E = 30e9
NU = 0.0

RHO_s = 2500.
RHO_b = 2000.

k = 2e9
# mu=10
# SURF = surf.Coulomb(k, k, 0.7)

knn = k;
knt = 0.;
ktn = 0.;
ktt = k
c = .2e6;
phi = .8;
psi = 0.0
ft = 0.0
Cnn = 1;
Css = 9.;
Cn = 0.;
fc = np.inf

# SURF = surf.ElastoPlastic(knn, ktt, 0, 0, c, phi, psi, ft, Cnn, Css, Cn, fc)
SURF = surf.Coulomb(k, k, phi)
# %% Building structure

St = st.Structure_2D()

St.add_fe(N1, N12_1, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N12_1, N12_2, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N12_2, N2, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_beam(N4, N5, BLOCKS, .2, rho=RHO_b, b=1)
St.add_fe(N3, N34_1, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N34_1, N34_2, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N34_2, N4, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N2, N23_1, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N23_1, N23_2, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N23_2, N3, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N5, N56_1, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N56_1, N56_2, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N56_2, N6, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N2, N25_1, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N25_1, N25_2, E, NU, H, b=B, lin_geom=False, rho=RHO_s)
St.add_fe(N25_2, N5, E, NU, H, b=B, lin_geom=False, rho=RHO_s)

St.make_nodes()
St.make_cfs(False, offset=-1, nb_cps=[-1, -.5, 0, .5, 1.], surface=SURF)

St.plot_structure(scale=0, plot_cf=False, save='build_model.eps')

# %% BCs and Forces

St.fixNode(N1, [0, 1, 2])
St.fixNode(N6, [0, 1, 2])

St.get_M_str()

for i in range(len(St.list_blocks)):
    W = St.list_blocks[i].m * 9.81
    N = St.list_blocks[i].ref_point
    Node = St.get_node_id(N)
    St.loadNode(Node, 1, -W, fixed=True)

for i in range(len(St.list_fes)):
    W = St.list_fes[i].mass[0, 0] * 9.81
    N_1 = St.list_fes[i].N1
    N_2 = St.list_fes[i].N2
    Node1 = St.get_node_id(N_1)
    Node2 = St.get_node_id(N_2)
    St.loadNode(Node1, 1, -W, fixed=True)
    St.loadNode(Node2, 1, -W, fixed=True)

St.solve_forcecontrol(10)
# print(St.eig_vals)

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

dt_new = 2.5e-3
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

Meth = 'NWK'

St.impose_dyn_excitation(St.get_node_id(N1), 0, u_trap, dt_new)
St.impose_dyn_excitation(St.get_node_id(N6), 0, u_trap, dt_new)

St.set_damping_properties(xsi=0.05, damp_type='STIFF', stiff_type='TAN_LG')

St.solve_dyn_nonlinear(20, dt_new, Meth=Meth, filename=f'Response_NF13_{pga}')
# %% Plot Structure at end of simulation

St.plot_structure(scale=1, plot_cf=False, plot_forces=False)
# %%
St.save_structure(filename='Composite_Frame')
# %%
