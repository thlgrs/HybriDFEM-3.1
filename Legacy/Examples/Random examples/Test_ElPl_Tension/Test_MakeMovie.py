# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:52:48 2024

@author: ibouckaert
"""

# %% Libraries imports
import matplotlib as mpl

# To have a nice LaTeX rendering (import LaTeX)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman']  # Example of a LaTeX font
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb, amsfonts}'

import matplotlib.pyplot as plt
from scipy import linalg
import pickle

import h5py
import os
import sys
import pathlib
import numpy as np


# ============================================================================
# FIXED: Removed hard-coded paths - use relative imports from Legacy package
# Original code (kept for reference):
# folder = pathlib.Path('C:/Users/ibouckaert/OneDrive - UCL/Bureau/UNIF/PhD/Coding/HybriDFEM 3.0/Objects')
# sys.path.append(str(folder))
# ============================================================================


files = []

with open(f'Lemos_Arch_Coulomb.pkl', 'rb') as file:
    St = pickle.load(file)

files = ['TAN_LG_NoTension_0.5g_0.0_CDM_.h5']

for i, file in enumerate(files):
    with h5py.File(file, 'r') as hf:
        # Import what you need
        last_conv = hf['Last_conv'][()]
        U_conv = hf['U_conv'][:last_conv]
        Time = hf['Time'][:last_conv]

# Set up output directory
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
if not os.path.exists(save_path):
    os.makedirs(save_path)

if not pathlib.Path(save_path).exists():
    pathlib.Path(save_path).mkdir(exist_ok=True)
else:
    print('Deleting Frames folder content...')
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        os.unlink(file_path)

dt = Time[1] - Time[0]
fps = 100

frames_needed = int(Time[-1] * fps)
ratio = int(len(Time) / frames_needed)

speed = .2

power = int(np.ceil(np.log10(frames_needed)))

print(f'Making {frames_needed} frames for movie...')
for i in range(frames_needed):
    St.U = U_conv[:, i * ratio]
    saveto = save_path + f'/Frame{i:0{power}d}.png'
    St.plot_structure(scale=1, plot_cf=False, plot_forces=False, show=False, save=saveto, lims=[[-1., 1.], [-.1, 1.5]])

# %% Make movie
import cv2

frame_pattern = f'Frames/Frame%0{power}d.png'
frame_filenames = [frame_pattern % i for i in range(frames_needed)]
# Create a video clip from the images
video_filename = f'Video_Test.mp4'

first_frame = cv2.imread(frame_filenames[0])
height, width, layers = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_filename, fourcc, fps * speed, (width, height))

for fname in frame_filenames:
    img = cv2.imread(fname)
    video.write(img)

video.release()

# Remove Frame files from folder
print('Video done... Deleting frames')
for filename in os.listdir(save_path):
    file_path = os.path.join(save_path, filename)
    os.unlink(file_path)
