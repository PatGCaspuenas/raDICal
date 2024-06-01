
# PACKAGES
import h5py
import numpy as np
import warnings
import matplotlib.pyplot as plt
import random
import os
import pandas as pd

# LOCAL FILES
from utils.data.read_data import read_FP
from utils.plt.plt_control import plot_input
from utils.plt.plt_snps import *
from utils.plt.plt_config import *

path_flow = r'F:\AEs_wControl\DATA\FP_14k_24k.h5'
path_grid = r'F:\AEs_wControl\DATA\FP_grid.h5'

# LOAD DATA
grid, flow = read_FP(path_grid, path_flow)
del flow['dUdt'], flow['dVdt']

with h5py.File(path_flow, 'w') as h5file:
    for key, item in flow.items():
        h5file.create_dataset(key, data=item)