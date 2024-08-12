# PACKAGES
import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import numpy as np
import re
import random
from matplotlib.ticker import FuncFormatter

from utils.data.read_data import read_FP
from utils.POD.fits import elbow_fit

# PARAMETERS

path_out = r'F:\AEs_wControl\misc\optimal_search'
path_list = r'F:\AEs_wControl\OUTPUT\optimal_search.csv'

files = os.listdir(path_out)
files = [f for f in files if f.endswith('.mat')]

df = pd.read_csv(path_list)
df['nepoch_s'] = 0
df['l_train_f'] = 0
df['l_val_f'] = 0
df['RMSE_AE'] = 0
df['CEA'] = 0
df['Sc'] = 0
df['zdetR'] = 0
df['zmeanR'] = 0
df['Dtime'] = 0

for f in files:
    # Get hyperparameters of file
    vars = re.split('_|lr|nepoch|batch|beta|nr|nt|val|train|reg|struct|drop|history.mat', f)
    vars = list(filter(None, vars))
    AE, lr, n_epochs, batch_size, beta, nr, nt,val, train, reg, struct, drop = vars
    lr, n_epochs, batch_size, beta, nr, nt,reg,drop,val,train = float(lr), int(float(n_epochs)), int(float(batch_size)), float(
        beta), int(float(nr)), int(float(nt)), float(reg), float(drop), int(val), int(train)

    # Match with csv list
    ilist = df.loc[(df['lr'] == lr) & (df['n_epochs'] == n_epochs) & (df['batch_size'] == batch_size)
                   & (df['beta'] == beta) & (df['nr'] == nr) & (df['nt'] == nt) & (df['nr'] == nr) & (
                           df['AE'] == AE) & (df['Val']==val) & (df['train']==train)& (df['struct']==struct)&
                     (df['reg']==reg) & (df['drop']==drop)].index[0]

    M = sio.loadmat(os.path.join(path_out, f))

    i_s = len(M['val_energy_loss'][0, :]) - 1

    df['nepoch_s'][ilist] = i_s + 1

    df['l_train_f'][ilist] = M['energy_loss'][0, -1]
    df['l_val_f'][ilist] = M['val_energy_loss'][0, -1]
    df['CEA'][ilist] = M['CEA'][0][0]
    df['Sc'][ilist] = M['Sc'][0][0]
    df['RMSE_AE'][ilist] = M['RMSE_AE'][0][0]
    df['zmeanR'][ilist] = M['zmeanR'][0][0]
    df['Dtime'][ilist] = M['Dtime'][0][0] / 60
    df['zdetR'][ilist] = M['zdetR'][0][0]

a=0