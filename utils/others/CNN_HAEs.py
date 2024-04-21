# -*- coding: utf-8 -*-
"""
08/10/2023

Tryout code for Hierarchical Autoencoders.
 
Parameters based on "Nonlinear mode decomposition with convolutional neural 
networks for fluid dynamics" by Murata et al. Methodology based on "Convolutional 
neural network based hierarchical autoencoder for nonlinear mode 
decomposition of fluid field data" by Fukami et al.

Started from template of Tensorflow in "Intro to Autoencoders" section.

"""
#%% Import libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io as sio
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

#%% Load data

path = 'F:\\nextflow\\CODES\\CNN-AEs\\+data\\';
filename = 'SC_00k_00k';

w = sio.loadmat(path+filename,variable_names=['w']);
t = sio.loadmat(path+filename,variable_names=['t']);
u = sio.loadmat(path+filename,variable_names=['u']);
v = sio.loadmat(path+filename,variable_names=['v']);
X = sio.loadmat(path+'SC_grid',variable_names=['X']);

w = w['w']; u = u['u']; v = v['v']; t = t['t']; X=X['X'];
nx = np.size(X,axis=1);
ny = np.size(X,axis=0);
nt = np.size(t,axis=1);
nd = 2;

w = np.reshape(w,(ny,nx,nt),order='F'); w = np.moveaxis(w,-1,0);
u = np.reshape(u,(ny,nx,nt),order='F'); u = np.moveaxis(u,-1,0);
v = np.reshape(v,(ny,nx,nt),order='F'); v = np.moveaxis(v,-1,0);
uv = np.stack([u,v]); uv = np.moveaxis(uv,0,3);
Dt = t[0,1] - t[0,0];

#%% Autoencoder settings -> same as paper of Murata

Adam = dict({'learn_rate': 0.001, 'rate_dec': 0,'beta1':0.9, 'beta2':0.999})
CNN_filter = (3,3);
CNN_pool = (2,2);
nepoch = 2000;
batch = 100;
latent_dim = 1;

uv_train, uv_test = train_test_split(uv, test_size=0.3)

#%% Autoencoder structure
# I have included padding and strides as in the tf template although the paper does not 
# meantion anything, but I think it is useful

class CNNHAE(Model):
  def __init__(self,*args):
      
    super(CNNHAE, self).__init__()
    
    self.encoder = tf.keras.Sequential()   
    self.encoder.add(tf.keras.layers.Input(shape=(ny, nx, nd)))
    # 1st: conv & max pooling
    self.encoder.add(tf.keras.layers.Conv2D(16, CNN_filter, activation='tanh', padding='same', strides=2))
    self.encoder.add(tf.keras.layers.MaxPooling2D( pool_size= CNN_pool, padding='same', strides=2))
    # 2nd: conv & max pooling
    self.encoder.add(layers.Conv2D(8, CNN_filter, activation='tanh', padding='same', strides=2))
    self.encoder.add(layers.MaxPooling2D( pool_size= CNN_pool, padding='same', strides=2))
    # 3rd: conv & max pooling
    self.encoder.add(layers.Conv2D(8, CNN_filter, activation='tanh', padding='same', strides=2))
    self.encoder.add(layers.MaxPooling2D( pool_size= CNN_pool, padding='same', strides=2))
    # 4th: conv & max pooling
    self.encoder.add(layers.Conv2D(8, CNN_filter, activation='tanh', padding='same', strides=2))
    self.encoder.add(layers.MaxPooling2D( pool_size= CNN_pool, padding='same', strides=2))
    # 5th: conv & max pooling
    self.encoder.add(layers.Conv2D(4, CNN_filter, activation='tanh', padding='same', strides=2))
    self.encoder.add(layers.MaxPooling2D( pool_size= CNN_pool, padding='same', strides=2))
    # 6th: conv & max pooling
    self.encoder.add(layers.Conv2D(4, CNN_filter, activation='tanh', padding='same', strides=2))
    self.encoder.add(layers.MaxPooling2D( pool_size= CNN_pool, padding='same', strides=2))
    # fully connected
    # self.encoder.add(layers.Flatten()) # If you flatten, you loose your (None,1,1,1) shape to (None,1) and we don't want that!
    a = self.encoder.add(layers.Dense(latent_dim, activation='tanh'))


    self.decoder = tf.keras.Sequential()
    #b = self.decoder.add(tf.keras.layers.concatenate([a,args],axis=0))
    #new_latent_dim= np.size(b,axis=0);
    # fully connected
    self.decoder.add(tf.keras.layers.Dense(4,activation='tanh'))
    # 7th: conv & 1st upsampling
    self.decoder.add(tf.keras.layers.Conv2DTranspose(4, CNN_filter, strides=2, activation='tanh', padding='same'))
    self.decoder.add(tf.keras.layers.Conv2D(4, CNN_filter, activation='tanh', padding='same', strides=2))
    # 8th: conv & 2nd upsampling
    self.decoder.add(tf.keras.layers.Conv2DTranspose(4, CNN_filter, strides=2, activation='tanh', padding='same'))
    self.decoder.add(tf.keras.layers.Conv2D(8, CNN_filter, activation='tanh', padding='same', strides=2))
    # 9th: conv & 3rd upsampling
    self.decoder.add(tf.keras.layers.Conv2DTranspose(8, CNN_filter, strides=2, activation='tanh', padding='same'))
    self.decoder.add(tf.keras.layers.Conv2D(8, CNN_filter, activation='tanh', padding='same', strides=2))
    # 10th: conv & 4th upsampling
    self.decoder.add(tf.keras.layers.Conv2DTranspose(8, CNN_filter, strides=2, activation='tanh', padding='same'))
    self.decoder.add(tf.keras.layers.Conv2D(8, CNN_filter, activation='tanh', padding='same', strides=2))
    # 11th: conv & 5th upsampling
    self.decoder.add(tf.keras.layers.Conv2DTranspose(8, CNN_filter, strides=2, activation='tanh', padding='same'))
    self.decoder.add(tf.keras.layers.Conv2D(16, CNN_filter, activation='tanh', padding='same', strides=2))        
    # 12th: conv & 6th upsampling
    self.decoder.add(tf.keras.layers.Conv2DTranspose(16, CNN_filter, strides=2, activation='tanh', padding='same'))
    self.decoder.add(tf.keras.layers.Conv2D(2, CNN_filter, activation='tanh', padding='same', strides=2))
    

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

#%% HAE procedure

# 1st mode
HAE1 = CNNHAE()
HAE1.compile(optimizer='adam', loss=losses.MeanSquaredError())
HAE1.fit(uv_train, uv_train,
                epochs=nepoch,
                shuffle=True,
                validation_data=(uv_test, uv_test))
gamma1 = HAE1.encoder(uv_test).numpy();
phi1 = HAE1.decoder(gamma1).numpy();
#%%
# 2nd mode
HAE2 = CNNHAE(gamma1)
HAE2.compile(optimizer='adam', loss=losses.MeanSquaredError())
HAE2.fit(uv_train, uv_train,
                epochs=nepoch,
                shuffle=True,
                validation_data=(uv_test, uv_test))
gamma2 = HAE2.encoder(uv_test).numpy();
phi2 = HAE2.decoder(gamma2).numpy();
#

