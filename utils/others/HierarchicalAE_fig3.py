# HierarchicalAE_fig3.py
# 9/9/2020 K. Fukami

# Hierarchical autoencoder with three subnetworks which is same configuoation as figure 3 in our paper (Fukami, Nakamura, Fukagata, Physics of Fluids, 2020).
# Code author: Kai Fukami (Keio University)

# Authors provide no guarantees for this code.  Use as-is and for academic research use only; no commercial use allowed without permission. For citations, please use the reference below:
## Ref: K. Fukami, T. Nakamura, and K. Fukagata, ``Convolutional neural network based hierarchical autoencoder for nonlinear mode decomposition of fluid field data," Physics of Fluids, 32, 095110, (2020)
# The code is written for educational clarity and not for speed.
# -- version 1: Sep 09, 2020

# IMPORTANT NOTE: This sample code is NOT a stand-alone code.
# The purpose of this sample code is to show how the 3rd subnetwork is constructed after the 1st and 2nd subnetworks are built. 
# The users should prepare own field data, build the 1st subnetwork as a standard CNN-AE, and the 2nd subnetwork similar to this sample code, all in advance.
# For this part, please also refer to the following reference and its sample code available at http://kflab.jp:
# T. Murata, K. Fukami, and K. Fukagata, "Nonlinear mode decomposition with convolutional neural networks for fluid dynamics," J. Fluid Mech. 882, A13 (2020).

from keras.layers import Input, Add, Dense, Conv2D, merge, Conv2DTranspose,Concatenate, MaxPooling2D, UpSampling2D, Flatten, Reshape, LSTM
from keras.models import Model
from keras import backend as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm as tqdm
import pickle

import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True,
        visible_device_list="0"
    )
)
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


    print(i)

mean = y.mean(axis=0).mean(axis=2)
for i in range(num_of_ts):
    y[i,:,:,0] = y[i,:,:,0] - mean
rdataX = []
for i in tqdm(range(0,5)):
    fnstr="/data/omega100/data" + '{0:01d}'.format(i)+".pickle"
    # Pickle load
    with open(fnstr, 'rb') as f:
        obj = pickle.load(f)
    if i==0:
        rdataX=obj
    else:
        rdataX=np.concatenate([rdataX,obj],axis=0)
#     print(Y.shape)
print("Finished", rdataX.shape)

#--- Set params ---#
num_of_ts=10000 # number of time slice
x_num=384; y_num=192; # no need to change
fea_num = 1 # Vorticity:1 Velocity:2
# init

y = np.zeros((num_of_ts,y_num,x_num,1))
for i in range(num_of_ts):
    y[i,:,:,0] = rdataX[i,:,:].T
    print(i)  
    

from keras.models import load_model
model_1 = load_model('./Model_1stsubnet.hdf5')

from keras import backend as K
get_9th_layer_output = K.function([model_1.layers[0].input],
                                  [model_1.layers[17].output])
mode_1 = np.zeros((num_of_ts,1))
for i in range(num_of_ts):
    mode_1[i,0] = get_9th_layer_output([y[i,:,:,:].reshape((1,y_num,x_num,1),order='F')])[0]
    print(i)


from keras.models import load_model
model_2 = load_model('./Model_2ndsubnet.hdf5')

from keras import backend as K
get_9th_layer_output = K.function([model_2.layers[0].input],
                                  [model_2.layers[17].output])
mode_2 = np.zeros((num_of_ts,1))
for i in range(num_of_ts):
    mode_2[i,0] = get_9th_layer_output([y[i,:,:,:].reshape((1,y_num,x_num,1),order='F')])[0]
    print(i)



act = 'linear'
input_img = Input(shape=(y_num,x_num,1))
input_enc_1 = Input(shape=(1,))
input_enc_2 = Input(shape=(1,))

x1 = Conv2D(16, (3,3),activation=act, padding='same')(input_img)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = MaxPooling2D((2,2),padding='same')(x1)
x1 = Reshape([3*6*4])(x1)
x1 = Dense(64,activation=act)(x1)
x1 = Dense(32,activation=act)(x1)
x1 = Dense(16,activation=act)(x1)

x1 = Dense(1,activation=act)(x1)
x1 = Concatenate()([x1, input_enc_1,input_enc_2])

x1 = Dense(16,activation=act)(x1)
x1 = Dense(32,activation=act)(x1)
x1 = Dense(64,activation=act)(x1)
x1 = Dense(72,activation=act)(x1)
x1 = Reshape([3,6,4])(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(4, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(8, (3,3),activation=act, padding='same')(x1)
x1 = UpSampling2D((2,2))(x1)
x1 = Conv2D(16, (3,3),activation=act, padding='same')(x1)

x_final = Conv2D(1, (3,3),padding='same')(x1)
autoencoder = Model([input_img,input_enc_1,input_enc_2], x_final)
autoencoder.compile(optimizer='adam', loss='mse')


from keras.callbacks import ModelCheckpoint,EarlyStopping
y_train, y_test, mode1_train, mode1_test, mode2_train, mode2_test = train_test_split(y, mode_1, mode_2, test_size=0.3, random_state=None)


model_cb=ModelCheckpoint('./Model_3rdsubnet.hdf5', monitor='val_loss',save_best_only=True,verbose=1)
early_cb=EarlyStopping(monitor='val_loss', patience=50,verbose=1)
cb = [model_cb, early_cb]
history = autoencoder.fit([y_train,mode1_train,mode2_train],y_train,nb_epoch=5000,batch_size=100,verbose=1,callbacks=cb,shuffle=True,validation_data=[[y_test,mode1_test,mode2_test], y_test])
import pandas as pd
df_results = pd.DataFrame(history.history)
df_results['epoch'] = history.epoch
df_results.to_csv(path_or_buf='./History_3rdsubnet.csv',index=False)
