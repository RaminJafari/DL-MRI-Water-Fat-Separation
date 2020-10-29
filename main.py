'''
author: ramin jafari (rj259@cornell.edu)
8/10/2020
https://doi.org/10.1002/mrm.28546
'''
#!/usr/bin/env python3
#!/usr/bin/env python2

import os
import math
import string
import sys
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from tensorflow.keras import layers
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten, Lambda, UpSampling2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import keras.backend as k
import keras.callbacks as cbks
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from keras.callbacks import CSVLogger
from model import *
from loss1 import *
from loss2 import *
#========load data==========
path_mat_in = sys.argv[1]
path_mat_out = sys.argv[2]

mat2 = sio.loadmat(path_mat_in)

input_train_t = np.array(mat2['ifield_test'], dtype = np.float64)
output_train_t = np.array(mat2['output_test'], dtype = np.float64)
input_test_t = np.array(mat2['ifield_test'], dtype = np.float64)
output_test_t = np.array(mat2['output_test'], dtype = np.float64)
te_train_t = np.array(mat2['te_test'], dtype = np.float64)
te_test_t = np.array(mat2['te_test'], dtype = np.float64)
dfat_train_t = np.array(mat2['dfat_test'], dtype = np.float64)
dfat_test_t = np.array(mat2['dfat_test'], dtype = np.float64)
mask_test_t = np.array(mat2['mask_test'], dtype = np.float64)

#========data normalization==========
input_m = input_train_t[:,:,:,0:6]
input_m = input_m.flatten()
input_mean_m = input_m[np.nonzero(input_m)].mean()
input_std_m = input_m[np.nonzero(input_m)].std()

input_p = input_train_t[:,:,:,6:12]
input_p = input_p.flatten()
input_mean_p = input_p[np.nonzero(input_p)].mean()
input_std_p = input_p[np.nonzero(input_p)].std()

#======z-score calculated from average of multiple datasets=========
w_mean_r = float(0.2)
w_std_r = float(0.1)
f_mean_r = float(0.1) 
f_std_r = float(0.2)
frq_mean = float(30) 
frq_std = float(30)
r2_mean = float(80)
r2_std = float(50)
w_mean_i = float(-1)
w_std_i = float(1) 
f_mean_i =float(-1)
f_std_i = float(1)

output_test_t[:,:,:,0] = (output_test_t[:,:,:,0]-w_mean_r)/w_std_r
output_test_t[:,:,:,1] = (output_test_t[:,:,:,1]-f_mean_r)/f_std_r
output_test_t[:,:,:,2] = (output_test_t[:,:,:,2]-w_mean_i)/w_std_i
output_test_t[:,:,:,3] = (output_test_t[:,:,:,3]-f_mean_i)/f_std_i
output_test_t[:,:,:,4] = (output_test_t[:,:,:,4]-frq_mean)/frq_std
output_test_t[:,:,:,5] = (output_test_t[:,:,:,5]-r2_mean)/r2_std

output_train_t[:,:,:,0] = (output_train_t[:,:,:,0]-w_mean_r)/w_std_r
output_train_t[:,:,:,1] = (output_train_t[:,:,:,1]-f_mean_r)/f_std_r
output_train_t[:,:,:,2] = (output_train_t[:,:,:,2]-w_mean_i)/w_std_i
output_train_t[:,:,:,3] = (output_train_t[:,:,:,3]-f_mean_i)/f_std_i
output_train_t[:,:,:,4] = (output_train_t[:,:,:,4]-frq_mean)/frq_std
output_train_t[:,:,:,5] = (output_train_t[:,:,:,5]-r2_mean)/r2_std

input_train_tm = input_train_t[:,:,:,0:6]
input_train_tp = input_train_t[:,:,:,6:12]
input_test_tm = input_test_t[:,:,:,0:6]
input_test_tp = input_test_t[:,:,:,6:12]

#input_train_tm = (input_train_tm-input_mean_m)/input_std_m
#input_train_tp = (input_train_tp-input_mean_p)/input_std_p
#input_test_tm = (input_test_tm-input_mean_m)/input_std_m
#input_test_tp = (input_test_tp-input_mean_p)/input_std_p

#==========================network============================
input_shape = (np.squeeze(input_train_tm[:1, :,:, :])).shape
input_train_tm_tens = Input(input_shape, name='input_train_tm')
input_train_tp_tens = Input(input_shape, name='input_train_tp')
input_shape = (np.squeeze(dfat_train_t[:1, :,:, :])).shape
dfat_train_t_tens = Input(input_shape, name='dfat_train_t')
input_shape = (np.squeeze(te_train_t[:1, :,:, :])).shape
te_train_t_tens = Input(input_shape, name='te_train_t')
input_shape = (np.squeeze(output_train_t[:1, :,:, :])).shape
output_train_t_tens = Input(input_shape, name='output_train_t')
model,output_pred = network(input_train_tm_tens,input_train_tp_tens, output_train_t_tens, dfat_train_t_tens, te_train_t_tens)

#==========define loss functions: loss1 for UTD/NTD, loss2 for STD==========
custom_loss3 = model.add_loss(1*loss1(input_train_tm_tens,input_train_tp_tens,output_train_t_tens,dfat_train_t_tens,te_train_t_tens,w_mean_r,w_std_r,f_mean_r,f_std_r,w_mean_i,w_std_i,f_mean_i,f_std_i,frq_mean,frq_std,r2_mean,r2_std,output_pred)\
                             +0*loss2(input_train_tm_tens,input_train_tp_tens,output_train_t_tens,dfat_train_t_tens,te_train_t_tens,output_pred))

#==========compile/train/test==========
model.compile(optimizer=Adam(lr=0.0006), loss=None)

filepath = "saved_weights-{epoch:02d}.hdf5"
csv_logger = CSVLogger('loss_vs_iteration.csv', append=True, separator=';')

callbacks = [ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.0001, verbose=1), ModelCheckpoint(filepath, verbose=1, save_best_only=True, save_weights_only=True, mode='auto',period=1000),csv_logger]

train_output = model.fit([input_train_tm,input_train_tp, output_train_t, dfat_train_t, te_train_t], callbacks=callbacks, validation_split=0, batch_size=2, epochs=10000,shuffle=True)
outresults = model.predict([input_test_tm,input_test_tp, output_test_t, dfat_train_t, te_train_t],batch_size=2)

#==========save results==========
sio.savemat(path_mat_out,{'test_pd':outresults,'test_gt':output_test_t,'w_mean_r':w_mean_r,'w_std_r':w_std_r,'f_mean_r':f_mean_r,'f_std_r':f_std_r,'frq_mean':frq_mean,'frq_std':frq_std,'r2_mean':r2_mean,'r2_std':r2_std,'w_mean_i':w_mean_i,'w_std_i':w_std_i,'f_mean_i':f_mean_i,'f_std_i':f_std_i,'mask':mask_test_t})





