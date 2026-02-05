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


# ============================================================================
# FIXED: Removed hard-coded paths - use relative imports from Legacy package
# Original code (kept for reference):
# folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
# sys.path.append(str(folder))
# ============================================================================


from Legacy.Objects import Structure as st
from Legacy.Objects import Contact as ct
from Legacy.Objects import Surface as surf
from Legacy.Objects import ContactPair as cp
from Legacy.Objects import Material as mat

N1 = np.array([0, 0])

H = 1
B = 1

E = 30e9
NU = 0.
RHO = 3000

CPS = 0

MAT1 = mat.Material(E / 100, NU)
MAT2 = mat.Material(E, NU)

r = .02
A = np.pi * r **2
h = r

PATTERN = [[1], [1]]
St = st.Structure_2D()
St.add_wall(N1, B, H, PATTERN, RHO, b=1., material=MAT1)

St.make_nodes()
St.make_cfs(True, nb_cps=CPS)

for cf in St.list_cfs: 
    cf.add_reinforcement([0], A, material=MAT2, height=h)

St.plot_structure()

W = 100e3
St.fixNode(0, [0, 1, 2])
St.fixNode(1, [2])
St.loadNode(1, 1, W)

St.solve_dispcontrol(1, .01, 1, 1, tol=1e-5, filename=f'Coulomb_Lin_Cont', max_iter=100)
St.plot_structure(scale=1)

St.plot_stresses()