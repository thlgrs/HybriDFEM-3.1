# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import h5py
import os

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 15  # Adjust font size
})

# %% Plot Pushover
Node = 9
W = 8517.20


# %% Results With Initial Stiffness

def plot_response(filename, label, color, linestyle=None):
    if os.path.exists(filename):

        with h5py.File(filename, 'r') as hf:

            last_conv = hf['Last_conv'][()]
            U_conv = hf['U_conv'][3 * Node, :last_conv]
            T_conv = hf['U_conv'][3 * Node + 2, :last_conv]

            U_conv -= 0.25 * np.sin(T_conv)
            where_max = np.argmax(U_conv)
            where_min = np.argmin(U_conv)
            Time = hf['Time'][:last_conv]

            print(Time[where_max])
            print(min(U_conv) * 1000)

            if 'Simulation_Time' in hf.attrs:
                sim_time = hf.attrs['Simulation_Time']
                print(label + f' took {sim_time} seconds')

        plt.plot(Time, U_conv * 1000, color=color, label=label, linestyle=linestyle)

    else:
        print('The file ' + filename + ' does not exist')


plt.figure(None, (9, 6), dpi=600)

file = 'TAN_NoTension_0.15g_0.05_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{tan} - 5\%$', color='orange')

file = 'TAN_NoTension_0.15g_0.01_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{tan} - 1\%$', color='orange', linestyle='--')

file = 'TAN_LG_NoTension_0.15g_0.05_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{tan,lg} - 5\%$', color='lightblue')

file = 'TAN_LG_NoTension_0.15g_0.01_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{tan,lg} - 1\%$', color='lightblue', linestyle='--')

file = 'INIT_NoTension_0.15g_0.05_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{0} - 5\%$', color='blue')

file = 'INIT_NoTension_0.15g_0.01_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{0} - 1\%$', color='blue', linestyle='--')

file = 'Simulation Results/SW_TAN_NoTension_0.15g_0.05_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{SW} - 5\%$', color='red')

file = 'Simulation Results/SW_TAN_NoTension_0.15g_0.01_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{SW} - 1\%$', color='red', linestyle='--')

file = 'TAN_NoTension_0.15g_0.05_CDM_.h5'
plot_response(file, '$\mathbf{K}^{tan} - 5\% - CDM$', color='purple')

data = np.loadtxt("Figures Lemos/fig-12/acn103-pp15-s5-h7.txt", skiprows=2)
# Extract columns as separate arrays
t_ls, d_ls = data[:, 0], data[:, 1] * 1000
plt.plot(t_ls, d_ls, color='green', label='3DEC, $5\%$ [50]', linestyle='-')
print(min(d_ls))

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)

plt.xlabel('Time [s]', fontsize=17)
plt.ylabel('Horizontal displacement [mm]', fontsize=17)
plt.xlim((0, 2))
plt.ylim((-15, 20))
plt.savefig('Response_all.eps')

# %% Comparison of tangent stiffness
plt.figure(None, (9, 6), dpi=600)
plt.xlabel('Time [s]', fontsize=17)
plt.ylabel('Horizontal displacement [mm]', fontsize=17)
# plt.title('Initial stiffness proportional damping',fontsize=17)


file = 'TAN_LG_NoTension_0.15g_0.05_Sing_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{tan,lg} - 5\%$', color='blue')

with h5py.File(file, 'r') as hf:
    steps_singular = hf['Singular_steps'][()]

from itertools import groupby
from operator import itemgetter

existing = set(steps_singular)
missing = [i for i in range(2501) if i not in existing]


def extract_consecutive_suites(lst):
    suites = []
    for _, group in groupby(enumerate(lst), key=lambda x: x[0] - x[1]):
        suites.append([x[1] for x in group])
    return suites


suites = extract_consecutive_suites(missing)
# print(suites)

dt = 1e-3

for suite in suites:
    if len(suite) > 1:  # only hatch if it's more than 1 point
        x_start = suite[0] * dt
        x_end = suite[-1] * dt
        plt.axvspan(x_start, x_end, facecolor='blue', edgecolor=None, hatch=None, alpha=0.3)

file = 'TAN_LG_NoTension_0.15g_0.01_Sing_NWK_g=0.5_b=0.25.h5'
plot_response(file, '$\mathbf{K}^{tan,lg} - 1\%$', color='red')

with h5py.File(file, 'r') as hf:
    steps_singular = hf['Singular_steps'][()]
# print(steps_singular)
existing = set(steps_singular)
missing = [i for i in range(5001) if i not in existing]
suites = extract_consecutive_suites(missing)
# print(suites)

dt = 5e-4

for suite in suites:
    if len(suite) > 1:  # only hatch if it's more than 1 point
        x_start = suite[0] * dt
        x_end = suite[-1] * dt
        plt.axvspan(x_start, x_end, facecolor='red', edgecolor=None, hatch=None, alpha=0.3)

plt.legend()
plt.grid(True)

plt.xlim((0, 2))
plt.ylim((-15, 20))

plt.savefig('Response_tan.eps')

# %%

# plt.figure(None, (6, 6),dpi=600)
# plt.xlabel('Time [s]',fontsize=17)
# plt.ylabel('Horizontal displacement [mm]',fontsize=17)
# # plt.title('Initial stiffness proportional damping',fontsize=17)


# file = 'TAN_NoTension_0.15g_0.05_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]
#     steps_singular = hf['Singular_steps'][()]

# plt.plot(Time, U_conv, color='blue',label=r'$A=0.15g$ ')

# file = 'TAN_NoTension_0.16g_0.05_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]
#     steps_singular = hf['Singular_steps'][()]

# plt.plot(Time, U_conv, color='red',label=r'$A=0.16g$ ')

# file = 'TAN_NoTension_0.17g_0.05_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]
#     steps_singular = hf['Singular_steps'][()]

# plt.plot(Time, U_conv, color='green',label=r'$A=0.17g$ ')
# plt.legend()
# plt.grid(True)

# plt.xlim((0,2.5))
# plt.ylim((-20,60))

# plt.savefig('Response_increased_amp.eps')

# #%%

# plt.figure(None, (6, 6),dpi=600)
# plt.xlabel('Time [s]',fontsize=17)
# plt.ylabel('Horizontal displacement [mm]',fontsize=17)
# # plt.title('Initial stiffness proportional damping',fontsize=17)


# file = 'MASS_NoTension_0.15g_0.05_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]
#     steps_singular = hf['Singular_steps'][()]

# plt.plot(Time, U_conv, color='blue',label=r'$A=0.15g$ ')

# file = 'MASS_NoTension_0.17g_0.05_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]
#     steps_singular = hf['Singular_steps'][()]

# plt.plot(Time, U_conv, color='red',label=r'$A=0.17g$')

# file = 'MASS_NoTension_0.16g_0.05_NWK_g=0.5_b=0.25.h5'


# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]
#     steps_singular = hf['Singular_steps'][()]

# plt.plot(Time, U_conv, color='purple',label=r'$A=0.16g$')

# file = 'MASS_NoTension_0.13g_0.05_NWK_g=0.5_b=0.25.h5'


# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]
#     steps_singular = hf['Singular_steps'][()]

# plt.plot(Time, U_conv, color='orange',label=r'$A=0.13g$')


# file = 'MASS_NoTension_0.14g_0.05_NWK_g=0.5_b=0.25.h5'

# with h5py.File(file, 'r') as hf:

#     #Import what you need
#     last_conv = hf['Last_conv'][()]
#     U_conv = hf['U_conv'][3*Node, :last_conv]*1000
#     Time = hf['Time'][:last_conv]
#     steps_singular = hf['Singular_steps'][()]

# plt.plot(Time, U_conv, color='green',label=r'$A=0.14g$')

# plt.legend()
# plt.grid(True)

# plt.xlim((0,2.5))
# plt.ylim((-5,10))

# # plt.savefig('Response_increased_amp.eps')
