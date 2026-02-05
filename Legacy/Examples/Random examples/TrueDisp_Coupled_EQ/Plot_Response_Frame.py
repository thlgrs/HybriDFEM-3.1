# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import h5py
import os
import re
import sys
import pathlib


# ============================================================================
# FIXED: Removed hard-coded paths - use relative imports from Legacy package
# Original code (kept for reference):
# folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
# sys.path.append(str(folder))
# ============================================================================


from Legacy.Objects import Structure

plt.rcParams.update({
    "text.usetex": True,  # Use LaTeX
    "font.family": "serif",  # Use a serif font
    "font.size": 12  # Adjust font size
})

# %% Plot accelerogram
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


fig = plt.figure(None, (6, 4), dpi=800)

dt, lmbda = read_accelerogram('Earthquakes/NF13')
lmbda = np.append(lmbda, np.zeros(2000)) / 100
time = np.arange(len(lmbda)) * dt
pga = .8
print(pga * np.max(abs(lmbda)))
lmbda = pga * lmbda / np.max(abs(lmbda))

plt.plot(time, lmbda, color='black')
plt.xlabel('Time [s]')
plt.ylabel('Ground acceleration [g]')
# ax1.set_xticklabels([])
plt.grid(True)
# plt.legend()
plt.xlim(0, 20)
plt.ylim(-1.6, 1.6)

# plt.savefig('earthquake.eps')


# plt.figure(None, figsize=(6,3),dpi=800)


plt.savefig('earthquake_load.eps')

# %% Plot Response

plt.figure(None, (6, 4), dpi=800)

filename_f = 'Response_NF13_0.8_NWK_g=0.5_b=0.25.h5'

with h5py.File(filename_f, 'r') as hf:
    last_conv = hf['Last_conv'][()]
    U_top = hf['U_conv'][0, :last_conv]
    U_oop = hf['U_conv'][7 * 3, :last_conv]
    U_bot = hf['U_conv'][14 * 3, :last_conv]

    Time = hf['Time'][:last_conv]

d_oop_f = U_oop - (U_top + U_bot) / 2

filename_1 = 'Response_NF13_0.8_Isol_CDM_.h5'

with h5py.File(filename_1, 'r') as hf:
    last_conv = hf['Last_conv'][()]
    U_top = hf['U_conv'][0, :last_conv]
    U_oop = hf['U_conv'][7 * 3, :last_conv]
    U_bot = hf['U_conv'][14 * 3, :last_conv]

    Time1 = hf['Time'][:last_conv]

d_oop_1 = U_oop - (U_top + U_bot) / 2
limit = int(len(Time1) / 4)

plt.plot(time, lmbda * 110 / np.max(abs(lmbda)), color='grey', linewidth=0.3, label='Northridge 1994')
plt.plot(Time, d_oop_f * 1000, color='red', label='OOP Disp')
plt.plot(Time1[:limit], d_oop_1[:limit] * 1000, color='blue', label='OOP Disp - Isolated')

plt.xlabel('Time [s]')
plt.ylabel('Out-of-plane displacement [mm]')
# ax1.set_xticklabels([])
plt.grid(True)
plt.legend()
plt.xlim(0, 2.5)
plt.ylim(-30, 30)

plt.savefig('oop_response.eps')

# %% Save deformed shapes

with h5py.File(filename_f, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][:last_conv]
    Time = hf['Time'][:last_conv]

max_oop = np.argmax(d_oop_f)
min_oop = np.argmin(d_oop_f)

import pickle

st_name = 'Composite_Frame.pkl'
with open(st_name, 'rb') as file:
    St = pickle.load(file)

lims = [[-1., 4.], [-1., 7.]]
St.U = U_conv[:, max_oop]
saveto = 'max_oop.eps'
St.plot_structure(scale=2, plot_cf=False, plot_forces=False, plot_supp=False, show=False, save=saveto, lims=lims)

St.U = U_conv[:, min_oop]
saveto = 'min_oop.eps'
St.plot_structure(scale=2, plot_cf=False, plot_forces=False, plot_supp=False, show=False, save=saveto, lims=lims)

# %% Save deformed shape of isolated wall
with h5py.File(filename_1, 'r') as hf:
    # Import what you need
    last_conv = hf['Last_conv'][()]
    U_conv = hf['U_conv'][:last_conv]
    Time = hf['Time'][:last_conv]

index = 13200

import pickle

st_name = 'Isolated_Frame.pkl'
with open(st_name, 'rb') as file:
    St = pickle.load(file)

lims = [[2.5, 4.], [2.5, 6.5]]
St.U = U_conv[:, index]
saveto = 'isolated_oop.eps'
St.plot_structure(scale=1, plot_cf=False, plot_forces=False, plot_supp=False, show=True, save=saveto, lims=lims)
