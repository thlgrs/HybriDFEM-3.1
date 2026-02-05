
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

discs = [3, 4, 5, 10, 15, 20]

colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
plt.figure(None, figsize=(6, 3), dpi=400)
for i, disc in enumerate(discs):

    filename = f'{disc}Bl_R_Coupled.h5'
    
    with h5py.File(filename, 'r') as hf:

        # Import what you need
        last_conv = hf['Last_conv'][()]
        P_c = hf['P_r_conv'][3 * disc, :last_conv] / 1000
        U_c = hf['U_conv'][3 * disc, :last_conv] * 1000

    indexes = np.where(P_c != 0)
    P_c = np.append(np.zeros(1), P_c[indexes])
    U_c = np.append(np.zeros(1), U_c[indexes])

    # plt.plot(U_c, P_c, label=f'{disc} Blocks', color=colors[i])

    filename = f'{disc}Bl_Coupled.h5'

    try:

        with h5py.File(filename, 'r') as hf:

            # Import what you need
            last_conv = hf['Last_conv'][()]
            P_c = hf['P_r_conv'][3 * disc, :last_conv] / 1000
            U_c = hf['U_conv'][3 * disc, :last_conv] * 1000

        indexes = np.where(P_c != 0)

        P_c = np.append(np.zeros(1), P_c[indexes])
        U_c = np.append(np.zeros(1), U_c[indexes])
        # plt.plot(U_c, P_c, label=None, color=colors[i], linestyle='--', linewidth=.75)

    except:
        pass

x = [0, 0.418848167539267, 0.7539267015706806, 1.5078534031413613, 2.429319371727749, 4.020942408376964,
     6.366492146596859, 8.209424083769633, 9.214659685863875, 10.219895287958115, 11.30890052356021, 12.481675392670159,
     16.25130890052356, 21.10994764397906, 25.130890052356023, 30.24083769633508, 34.345549738219894,
     38.952879581151834, 43.56020942408377, 47.91623036649215, 52.439790575916234, 63.497382198952884,
     65.34031413612566, 67.35078534031415, 71.12041884816755, 74.30366492146597, 75.47643979057592, 77.23560209424085,
     79.49738219895289, 82.51308900523561, 84.27225130890052]
y = [0, 98.9795918367347, 157.14285714285714, 252.04081632653063, 334.6938775510204, 424.48979591836735,
     511.2244897959184, 564.2857142857143, 571.4285714285714, 573.469387755102, 574.4897959183673, 574.4897959183673,
     570.4081632653061, 564.2857142857143, 555.1020408163265, 541.8367346938776, 533.6734693877551, 522.4489795918367,
     511.2244897959184, 500, 485.7142857142857, 444.8979591836735, 419.38775510204084, 400, 393.8775510204082,
     372.44897959183675, 361.2244897959184, 356.12244897959187, 355.10204081632656, 339.7959183673469,
     328.57142857142856]
plt.plot(x, y, label='FE', color='black', linestyle='--', linewidth=.75)

plt.ylabel(r'F [kN]')
plt.xlabel(r'Top horizontal displacement [mm]')
plt.grid(True)
plt.legend()
plt.xlim((0, 80))
plt.ylim(0, 700)

plt.savefig('reg_flexion_bis.eps')

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
