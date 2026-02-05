import numpy as np
import os
import h5py
import sys
import pathlib
import matplotlib.pyplot as plt

folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
sys.path.append(str(folder))

import Structure as st
import Contact as cont
import ContactPair as cp
import Surface as surf

save_path = os.path.dirname(os.path.abspath(__file__))

# Meth =['HHT', 0.3]
# Meth = 'CAA'
# Meth = 'CDM'

kn = 1e9
ks = 1e9

B = 1.
H = 1
L = 0.5

nb_blocks = 2

rho = 2700

St1 = st.Structure_2D()

H_base = .5
L_base = 1

N1 = np.array([0, -H_base / 2], dtype=float)
N2 = np.array([0, H / 2], dtype=float)
x = np.array([.5, 0])
y = np.array([0, .5])

vertices = np.array([N1, N1, N1, N1])
vertices[0] += L_base * x - H_base * y
vertices[1] += L_base * x + H_base * y
vertices[2] += - L_base * x + H_base * y
vertices[3] += - L_base * x - H_base * y

St1.add_block(vertices, rho)

vertices = np.array([N2, N2, N2, N2])
vertices[0] += L * x - H * y
vertices[1] += L * x + H * y
vertices[2] += - L * x + H * y
vertices[3] += - L * x - H * y

St1.add_block(vertices, rho)

St1.make_nodes()

St1.make_cfs(False, nb_cps=20, offset=-1, surface=surf.Coulomb(kn, ks, 1))
St1.fixNode(0, [0, 1, 2])

M = St1.list_blocks[1].m
W = M * 9.81
St1.loadNode(1, [1], -W)

# # Gravity loads applied statically
Node = 1

# St1.plot_structure(scale=10, plot_forces=True, plot_cf=True)
St1.solve_forcecontrol(2, tol=.1)

# St2.solve_forcecontrol(10, tol=.1)

St1.reset_loading()

St1.loadNode(1, [1], -W, fixed=True)
St1.loadNode(1, [0], -W)

St1.plot_structure(scale=0, plot_forces=True, plot_cf=True)

# %% Excitation function and damping
t_p = .3
a = 0.5

lag = 0.0

print(f'Period is {t_p}s and amplitude is {a}g')


def lmbda(x):
    if x < lag: return 0
    if x < t_p + lag: return a * x ** 2 / 2
    if x < 3 * t_p + lag: return - a * x ** 2 / 4 + 3 * a * t_p * x / 2 - 3 * a * t_p ** 2 / 4
    return 3 * a * t_p ** 2 /2


t_end = 8
dt = 5e-4
time = np.arange(0, t_end, dt)
time = np.append(time, t_end)

Disp = np.zeros(len(time))  # Displacement

for i, t in enumerate(time):
    Disp[i] = lmbda(t) * 9.81

St1.impose_dyn_excitation(0, 0, Disp, dt)
damp = 0.05
stiff_type = 'TAN'
St1.set_damping_properties(xsi=damp, damp_type='STIFF', stiff_type=stiff_type)

# # #%% Computation 
Meth = 'CDM'
St1.solve_dyn_nonlinear(5, dt, Meth=Meth, lmbda=None, filename=f'TwoBlocks_{a}g_Rock')
St1.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St1.save_structure(filename='TwoBlocks_Pulse')
