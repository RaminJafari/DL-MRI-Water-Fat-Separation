'''
author: ramin jafari (rj259@cornell.edu)
https://arxiv.org/abs/2004.07923
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


def loss1(input_train_tm_tens,input_train_tp_tens,output_train_t_tens,dfat_train_t_tens,te_train_t_tens,w_mean_r,w_std_r,f_mean_r,f_std_r,w_mean_i,w_std_i,f_mean_i,f_std_i,frq_mean,frq_std,r2_mean,r2_std,output_pred):

    pi = tf.constant(math.pi)


    wat_r = output_pred[:,:,:,0]
    watt_r = k.expand_dims(wat_r,3)
    watt_r = k.repeat_elements(watt_r,6,3)
    watt_r = tf.scalar_mul(w_std_r,watt_r)+w_mean_r

    fat_r = output_pred[:,:,:,1]
    fatt_r = k.expand_dims(fat_r,3)
    fatt_r = k.repeat_elements(fatt_r,6,3)
    fatt_r = tf.scalar_mul(f_std_r,fatt_r)+f_mean_r

    wat_i = output_pred[:,:,:,3]
    watt_i = k.expand_dims(wat_i,3)
    watt_i = k.repeat_elements(watt_i,6,3)
    watt_i = tf.scalar_mul(w_std_i,watt_i)+w_mean_i

    fat_i = output_pred[:,:,:,4]
    fatt_i = k.expand_dims(fat_i,3)
    fatt_i = k.repeat_elements(fatt_i,6,3)
    fatt_i = tf.scalar_mul(f_std_i,fatt_i)+f_mean_i

    frq = output_pred[:,:,:,5]
    frqt = k.expand_dims(frq,3)
    frqt = k.repeat_elements(frqt,6,3)
    frqt = tf.scalar_mul(frq_std,frqt)+frq_mean

    r2 = output_pred[:,:,:,2]
    r2t = k.expand_dims(r2,3)
    r2t = k.repeat_elements(r2t,6,3)
    r2t = tf.scalar_mul(r2_std,r2t)+r2_mean

    watt_c =  tf.cast(tf.complex(watt_r,0*watt_r), tf.complex64)
    fatt_c =  tf.cast(tf.complex(fatt_r,0*fatt_r), tf.complex64)

    watt_ci =  tf.cast(tf.complex(watt_i,0*watt_i), tf.complex64)
    fatt_ci =  tf.cast(tf.complex(fatt_i,0*fatt_i), tf.complex64)

    r2t_c =  tf.cast(tf.complex(r2t,0*r2t), tf.complex64)
    frqt_c = tf.cast(tf.complex(frqt,0*frqt),tf.complex64)
    dfat_train_t_c = tf.cast(tf.complex(dfat_train_t_tens,0*dfat_train_t_tens),tf.complex64)
    te_train_t_c = tf.cast(tf.complex(te_train_t_tens,0*te_train_t_tens),tf.complex64)

    pi_cmp =  tf.cast(tf.complex(pi,0*pi), tf.complex64)


    #signal = tf.cast(((watt_c)*tf.exp(1j*watt_ci) + tf.exp(1j*fatt_ci)*(fatt_c)*tf.exp(-1j*2*pi_cmp*dfat_train_t_c*te_train_t_c))*tf.exp(-1*r2t_c*te_train_t_c)*tf.exp(-1j*2*pi_cmp*frqt_c*te_train_t_c),tf.complex64)
    signal = tf.cast(((watt_c)*tf.exp(1j*watt_ci) + tf.exp(1j*fatt_ci)*(fatt_c)*tf.exp(-1j*2*pi_cmp*dfat_train_t_c*te_train_t_c))*tf.exp(-1*r2t_c*te_train_t_c)*tf.exp(-1j*2*pi_cmp*frqt_c*te_train_t_c),tf.complex64)

    input_train_t_mag = input_train_tm_tens[:,:,:,0:6]
    input_train_t_phs = input_train_tp_tens[:,:,:,0:6]

    #input_train_t_mag = tf.scalar_mul(input_std_m,input_train_t_mag)+input_mean_m
    #input_train_t_phs = tf.scalar_mul(input_std_p,input_train_t_phs)+input_mean_p

    gt_input_train2 = tf.cast(tf.multiply(tf.complex(input_train_t_mag,0*input_train_t_mag),tf.exp(1j*tf.complex(input_train_t_phs,0*input_train_t_phs))),tf.complex64)

    loss = tf.sqrt(tf.linalg.norm(tf.abs(gt_input_train2-signal)))

    return loss



