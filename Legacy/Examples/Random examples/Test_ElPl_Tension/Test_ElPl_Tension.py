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

kn = 1e9
ks = 1e9
mu = 10

B = 1.
H = 1.

RHO = 1000

N1 = np.array([0, 0], dtype=float)
x = np.array([.5, 0])
y = np.array([0, 1])

PATTERN = [[1], [1]]

St1 = st.Structure_2D()

# St.add_block(vertices, RHO, b=1)
St1.add_wall(N1, B, H, PATTERN, RHO, b=1., material=None)
St1.make_nodes()

nb_cps = 1
# nb_cps = nb_cps.tolist()

SURF = surf.ElastoPlastic(knn=kn, ktt=ks, knt=0, ktn=0, c=0.0, phi=mu, psi=0.0, ft=0.0, Cnn=2e16, Css=2e16, Cn=0,
                          fc=2e16)
# SURF.plot_failure_dom()

SURF = surf.Coulomb(kn, ks, mu)
# SURF.plot_failure_dom()

St1.make_cfs(False, nb_cps=nb_cps, offset=-1, surface=SURF)

St1.plot_structure(scale=0, plot_forces=False, plot_cf=True)
St1.fixNode(0, [0, 1, 2])
St1.fixNode(1, [0, 2])

g = 9.81
M = St1.list_blocks[0].m
W = M * g
St1.loadNode(1, [1], 1.1 * W, fixed=False)
St1.loadNode(1, [1], -W, fixed=True)
# St1.loadNode(1, [0], -W)

# %% Excitation function and damping
t_p = .5
w_s = np.pi / t_p
a = 1.0
lag = 0


# print(f'Period is {t_p}s and amplitude is {a}g')

def lmbda(x):
    if x < lag: return 0
    if x < t_p + lag: return a
    # if x < 2*t_p + lag: return -a
    return 0


# St1.impose_dyn_excitation(0,0,Disp,dt)
damp = 0.0
stiff_type = 'TAN'
St1.set_damping_properties(xsi=damp, damp_type='STIFF', stiff_type=stiff_type)

# #%% Computation
Meth = 'NWK'
St1.solve_dyn_nonlinear(2, 5e-4, Meth=Meth, lmbda=lmbda, filename=stiff_type + f'_NoTension_{-a}g_{damp}')
St1.plot_structure(scale=1, plot_forces=False, plot_cf=True)

St1.save_structure(filename='Lemos_Arch_Coulomb')

print(St1.damp_coeff)

# %% Plot results
files = [stiff_type + f'_NoTension_{-a}g_{damp}' + '_CDM_.h5']
files = ['TAN_NoTension_-1.0g_0.0_NWK_g=0.5_b=0.25.h5']

for i, file in enumerate(files):
    with h5py.File(file, 'r') as hf:
        # Import what you need
        last_conv = hf['Last_conv'][()]
        U_conv = hf['U_conv'][4, :last_conv]
        # P_conv = hf['P_r_conv'][:last_conv,5]
        Time = hf['Time'][:last_conv]

plt.plot(Time, U_conv)
plt.ylim((0, 0.15))
