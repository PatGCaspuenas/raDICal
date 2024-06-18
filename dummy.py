import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = r'F:\Re150\InputValues'
inputs = 3

for i in range(1, inputs+1):
    df = pd.read_csv(path+str(i)+'.txt',header=None,sep=' ')
    if i == 1:
        utrain = df.to_numpy()
    else:
        utrain = np.concatenate((utrain, df.to_numpy()), axis=0)

df = pd.read_csv(path+str(4)+'.txt',header=None,sep=' ')
uval = df.to_numpy()

nt_train = np.shape(utrain)[0]
nt_val = np.shape(uval)[0]

# 0: F, 1: T, 2: B
BB = []
BT = []
M =[]
M_val =[]
FSP =[]

for t in range(nt_val):
    if (uval[t,1] == uval[t,2]) & (uval[t,1] == uval[t,0]): # magnus
        M_val.append(t)

for t in range(nt_train):
    if (utrain[t,1] == -utrain[t,2]) & (utrain[t,1]>0) & (utrain[t,0]==0): # base bleed, T-CCW, B-CW
        BB.append(t)
    if (utrain[t,1] == -utrain[t,2]) & (utrain[t,1]<0) & (utrain[t,0]==0): # boat tailing, T-CW, B-CCW
        BT.append(t)
    if (utrain[t,1] == utrain[t,2]) & (utrain[t,1] == utrain[t,0]): # magnus
        M.append(t)
    if (utrain[t,1] == 0) & (utrain[t,2] == 0) & (utrain[t,0] != 0): # forward stagnation point
        FSP.append(t)

BB = np.array(BB)
BT = np.array(BT)
M =np.array(M)
M_val =np.array(M_val)
FSP =np.array(FSP)

jBB = np.where((BB[1:]-BB[:-1])>1)
jBT = np.where((BT[1:]-BT[:-1])>1)
jM = np.where((M[1:]-M[:-1])>1)
jFSP = np.where((FSP[1:]-FSP[:-1])>1)

a=0