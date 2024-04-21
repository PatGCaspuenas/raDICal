# MD-CNN-AE.py
# 2019 T. Murata
#
# Mode Decomposing CNN Autoencoder
# Author: Takaaki Murata (Keio University, http://kflab.jp/en/index.php?top)
#
# This code consists of five parts: 1. Set parameter, 2. download files, 3. load flow field data, 4. make machine learning model, and 5. train the network.
#
# For citations, please use the reference below:
# > T. Murata, K. Fukami & K. Fukagata,
# > "Nonlinear mode decomposition with convolutional neural networks for fluid dynamics,"
# > J. Fluid Mech. Vol. 882, A13 (2020).
# > https://doi.org/10.1017/jfm.2019.822
#
# Takaaki Murata provides no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission.
# The code is written for educational clarity and not for speed.

import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda, Add, Reshape
from keras.models import Model

from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm as tqdm
import os, sys
import urllib.request

#########################################
# 1. Set parameter
#########################################
n_epoch=2000 # Number of epoch
pat=50 # Patience
filenm='cnn1' # File name of this model 

#########################################
# 2. Download files used in this code
#########################################
# [NOTICE] In this code, we use 2000 snapshots (1 pickle file) with float16 precision to make the file size small altough we use 10000 snapshots (5} pickle files) with float64 precision in the original paper.
# You need about 600MB free space in total for the downloading files.
# The pickle file is loaded in "3. Load flow field."

def dl_progress(count, block_size, total_size):
    sys.stdout.write('\r %d%% of %d MB' %(100*count*block_size/total_size, total_size/1024/1024))

savename = "flow_field_data0.pickle"
if(not os.path.exists(savename)):
    url = "https://dl.dropboxusercontent.com/s/3pnuoxrx9xvqxi2/flow_field_data0.pickle"
    print('Downloading:',savename)
    urllib.request.urlretrieve(url, savename, dl_progress)
    print('')

savename = "mode0.csv"
if(not os.path.exists(savename)):
    url = "https://dl.dropboxusercontent.com/s/x3bw3h1zfwty84x/mode0.csv"
    print('Downloading:',savename)
    urllib.request.urlretrieve(url, savename, dl_progress)
    print('')

#########################################
# 3. Load flow field
#########################################
# Flow fields are stored in "pickle" files
X=[]
for i in tqdm(range(1)):
    fnstr="./flow_field_data" + '{0:01d}'.format(i)+".pickle"
    # Pickle load
    with open(fnstr, 'rb') as f:
        obj = pickle.load(f)
    if i==0:
        X=obj
    else:
        X=np.concatenate([X,obj],axis=0)
print(X.shape)
# The size of X is (# of snapshots, 384(Nx), 192(Ny), 2(u&v))

# Load average field
mode0=pd.read_csv("./mode0.csv", header=None, delim_whitespace=None)
mode0=mode0.values

x_num0=384; y_num0=192;
Uf0=mode0[0:x_num0*y_num0]
Vf0=mode0[x_num0*y_num0:x_num0*y_num0*2]
Uf0=np.reshape(Uf0,[x_num0,y_num0])
Vf0=np.reshape(Vf0,[x_num0,y_num0])

# We consider fluctuation component of flow field
for i in range(len(X)):
    X[i,:,:,0]=X[i,:,:,0]-Uf0
    X[i,:,:,1]=X[i,:,:,1]-Vf0

#########################################
# 4. Make machine learning model
#########################################
input_img = Input(shape=(384, 192, 2))
## Encoder
x = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(4, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Reshape([6*3*4])(x)
encoded = Dense(2,activation='tanh')(x)
## Two variables
val1= Lambda(lambda x: x[:,0:1])(encoded)
val2= Lambda(lambda x: x[:,1:2])(encoded)
## Decoder 1
x1 = Dense(6*3*4,activation='tanh')(val1)
x1 = Reshape([6,3,4])(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(4,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(16,(3,3),activation='tanh',padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1d = Conv2D(2,(3,3),activation='linear',padding='same')(x1)
## Decoder 2
x2 = Dense(6*3*4,activation='tanh')(val2)
x2 = Reshape([6,3,4])(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(4,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(8,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2 = Conv2D(16,(3,3),activation='tanh',padding='same')(x2)
x2 = UpSampling2D((2,2))(x2)
x2d = Conv2D(2,(3,3),activation='linear',padding='same')(x2)

decoded = Add()([x1d,x2d])

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Check the network structure
autoencoder.summary()

#########################################
# 5. Train the network
#########################################
tempfn='./'+filenm+'.hdf5'
model_cb=ModelCheckpoint(tempfn, monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=pat,verbose=1)
cb = [model_cb, early_cb]

X_train,X_test,y_train,y_test=train_test_split(X,X,test_size=0.3,random_state=1)

history=autoencoder.fit(X_train, y_train,
                epochs=n_epoch,
                batch_size=100,
                shuffle=True,
                validation_data=(X_test, y_test),
                callbacks=cb )

df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
tempfn='./'+filenm+'.csv'
df_results.to_csv(path_or_buf=tempfn,index=False)

