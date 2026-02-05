# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:49:37 2025

@author: ibouckaert
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'
plt.rcParams['font.size'] = 15

discs = [2, 3, 4, 5, 6]

colors = ['red', 'blue', 'green', 'orange', 'purple']

plt.figure(None, figsize=(5, 5), dpi=400)
for i, disc in enumerate(discs):
    filename = f'{disc}Bl_C.h5'
    
    with h5py.File(filename, 'r') as hf:
        # Import what you need
        last_conv = hf['Last_conv'][()]
        P_c = hf['P_r_conv'][3 * disc - 3, :last_conv] / 1000
        U_c = hf['U_conv'][3 * disc - 3, :last_conv] * 1000

    indexes = np.where(P_c != 0)
    P_c = np.append(np.zeros(1), P_c[indexes])
    U_c = np.append(np.zeros(1), U_c[indexes])

    # P_c = np.append(P_c, np.zeros(1))
    # U_c = np.append(U_c, U_c[-1])

    plt.plot(-U_c, -P_c, label=f'{disc} Blocks', color=colors[i])

# plt.title('Regularized')
plt.grid(True)
plt.legend(fontsize=12)
plt.xlabel('Compression force [kN]')
plt.ylabel('Top displacement [mm]')
plt.xlim((0, 5))
plt.ylim(0, 1500)

plt.savefig('NonReg_C.eps')

#%% Regularized plotplt.figure(None,figsize=(6,6),dpi=400)

# plt.figure(None,figsize=(6,6),dpi=400)

# for disc in discs: 

#     filename = f'Results_{disc}_R.h5'

#     with h5py.File(filename, 'r') as hf:

#         #Import what you need
#         last_conv = hf['Last_conv'][()]
#         P_c = -hf['P_r_conv'][3*disc-3,:last_conv] / 1000
#         U_c = -hf['U_conv'][3*disc-3,:last_conv] * 1000

#         plt.plot(U_c, P_c, label=f'{disc} Blocks')

# plt.title('Regularized with fracture energy')
# plt.grid(True)
# plt.legend()
# plt.xlim((0,5))
# plt.ylim(0,1300)
