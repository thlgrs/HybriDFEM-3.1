# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 09:19:04 2025

@author: ibouckaert
"""

import matplotlib.pyplot as plt
import h5py
import pickle
import numpy as np
import pathlib
import sys
import matplotlib as mpl
from datetime import timedelta

BL = 20
path = 'Reinforced Concrete/Localization/KSP_Flexion/'
file = path + f'{BL}Bl_R_Coupled.h5'

with h5py.File(file, 'r') as hf:
    # Import what you need

    t = hf.attrs['Simulation_Time']

    # t = hf['Time'][()]
    # print(t)

td = timedelta(seconds=t)
print(str(td))
