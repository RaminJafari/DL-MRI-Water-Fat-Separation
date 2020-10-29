'''
author: ramin jafari (rj259@cornell.edu)
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


def network(input_train_tm_tens,input_train_tp_tens, output_train_t_tens, dfat_train_t_tens, te_train_t_tens):


 n_filters = 32
 kernel_size =2


 c1 = Conv2D(12, kernel_size=(kernel_size, kernel_size), activation = 'sigmoid', padding = 'same', kernel_initializer = 'glorot_uniform')(input_train_tm_tens)
 c1 = BatchNormalization()(c1)
 p1 = MaxPooling2D((2, 2)) (c1)

 c2 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (p1)
 c2 = BatchNormalization()(c2)
 p2 = MaxPooling2D((2, 2))(c2)

 c3 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (p2)
 c3 = BatchNormalization()(c3)
 p3 = MaxPooling2D((2, 2)) (c3)

 c4 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (p3)
 c4 = BatchNormalization()(c4)
 p4 = MaxPooling2D((2, 2)) (c4)

 c5 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (p4)
 c5 = BatchNormalization()(c5)
 p5 = MaxPooling2D((2, 2)) (c5)

 c6 = Conv2D(384,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (p5)
 c6 = BatchNormalization()(c6)
 c6 = UpSampling2D(size=(2, 2), data_format=None) (c6)

 u7 = Conv2DTranspose(192,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c6)
 u7 = concatenate([u7, c5])
 c7 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (u7)
 c7 = BatchNormalization()(c7)
 c7 = UpSampling2D(size=(2, 2), data_format=None) (c7)

 u8 = Conv2DTranspose(96,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c7)
 u8 = concatenate([u8, c4],axis=3)
 c8 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (u8)
 c8 = BatchNormalization()(c8)
 c8 = UpSampling2D(size=(2, 2), data_format=None) (c8)

 u9 = Conv2DTranspose(48,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c8)
 u9 = concatenate([u9, c3],axis=3)
 c9 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (u9)
 c9 = BatchNormalization()(c9)
 c9 = UpSampling2D(size=(2, 2), data_format=None) (c9)

 u10 = Conv2DTranspose(24,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c9)
 u10 = concatenate([u10, c2],axis=3)
 c10 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (u10)
 c10 = BatchNormalization()(c10)
 c10 = UpSampling2D(size=(2, 2), data_format=None) (c10)

 u11 = Conv2DTranspose(12,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (c10)
 u11 = concatenate([u11, c1],axis=3)
 c11 = Conv2D(12,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (u11)
 c11 = BatchNormalization()(c11)
 c11 = UpSampling2D(size=(1, 1), data_format=None) (c11)

 output_pred1 = Conv2D(3,(1, 1), activation='linear') (c11)


 cc1 = Conv2D(12, kernel_size=(kernel_size, kernel_size), activation = 'sigmoid', padding = 'same', kernel_initializer = 'glorot_uniform')(input_train_tp_tens)
 cc1 = BatchNormalization()(cc1)
 pp1 = MaxPooling2D((2, 2)) (cc1)

 cc2 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (pp1)
 cc2 = BatchNormalization()(cc2)
 pp2 = MaxPooling2D((2, 2))(cc2)

 cc3 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (pp2)
 cc3 = BatchNormalization()(cc3)
 pp3 = MaxPooling2D((2, 2)) (cc3)

 cc4 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (pp3)
 cc4 = BatchNormalization()(cc4)
 pp4 = MaxPooling2D((2, 2)) (cc4)

 cc5 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (pp4)
 cc5 = BatchNormalization()(cc5)
 pp5 = MaxPooling2D((2, 2)) (cc5)

 cc6 = Conv2D(384,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (pp5)
 cc6 = BatchNormalization()(cc6)
 cc6 = UpSampling2D(size=(2, 2), data_format=None) (cc6)

 uu7 = Conv2DTranspose(192,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc6)
 uu7 = concatenate([uu7, cc5])
 cc7 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (uu7)
 cc7 = BatchNormalization()(cc7)
 cc7 = UpSampling2D(size=(2, 2), data_format=None) (cc7)

 uu8 = Conv2DTranspose(96,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc7)
 uu8 = concatenate([uu8, cc4])
 cc8 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (uu8)
 cc8 = BatchNormalization()(cc8)
 cc8 = UpSampling2D(size=(2, 2), data_format=None) (cc8)


 uu9 = Conv2DTranspose(48,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc8)
 uu9 = concatenate([uu9, cc3])
 cc9 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (uu9)
 cc9 = BatchNormalization()(cc9)
 cc9 = UpSampling2D(size=(2, 2), data_format=None) (cc9)
 
 uu10 = Conv2DTranspose(24,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc9)
 uu10 = concatenate([uu10, cc2])
 cc10 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (uu10)
 cc10 = BatchNormalization()(cc10)
 cc10 = UpSampling2D(size=(2, 2), data_format=None) (cc10)

 uu11 = Conv2DTranspose(12,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same') (cc10)
 uu11 = concatenate([uu11, cc1], axis=3)
 cc11 = Conv2D(12,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same') (uu11)
 cc11 = BatchNormalization()(cc11)
 cc11 = UpSampling2D(size=(1, 1), data_format=None) (cc11)
 output_pred2 = Conv2D(3,(1, 1), activation='linear') (cc11)

 output_pred = concatenate([output_pred1,output_pred2])

 model = Model(inputs=[input_train_tm_tens,input_train_tp_tens, output_train_t_tens, dfat_train_t_tens, te_train_t_tens], outputs=[output_pred])


 return model,output_pred


