#!/usr/bin/env python3 
import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, UpSampling1D, Conv1DTranspose,Lambda,add, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def build_autoencoder(size,channel): #2048

    inputs = Input(shape=[size,channel])

    ## encoding layers
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1) #1024

    conv2 = Conv1D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(64, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2) #512

    conv3 = Conv1D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(128, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3) #256

    conv4 = Conv1D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv1D(256, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4) #128

    conv4_2 = Conv1D(512, 3, activation='relu', padding='same')(pool4)
    conv4_2 = Conv1D(512, 3, activation='relu', padding='same')(conv4_2) #128*512
    pool4_2 = MaxPooling1D(pool_size=2)(conv4_2)  #64*512


    #decoding layers
    conv5_2 = Conv1D(512, 3, activation='relu', padding='same')(pool4_2) #64*512
    conv5_2 = Conv1D(512, 3, activation='relu', padding='same')(conv5_2)
    up1_2 = concatenate([Conv1DTranspose(512, 2, strides=2, activation='relu',padding='same')(conv5_2),conv4_2],axis=-1) # conv5_2 conv1dtranspose: 128*512, after concat: 128*1024(up1_2)
    #up1_2 = Conv1DTranspose(512, 2, strides=2, activation='relu',padding='same')(conv5_2) # if you want to do full, then no concate

    conv5 = Conv1D(256, 3, activation='relu', padding='same')(up1_2) #128*256
    conv5 = Conv1D(256, 3, activation='relu', padding='same')(conv5)
    up1 = concatenate([Conv1DTranspose(512, 2, strides=2, activation='relu',padding='same')(conv5),conv4],axis=-1) #256*XXX

    conv6 = Conv1D(128, 3, activation='relu', padding='same')(up1)
    conv6 = Conv1D(128, 3, activation='relu', padding='same')(conv6)
    up2 = concatenate([Conv1DTranspose(512, 2, strides=2, activation='relu',padding='same')(conv6),conv3],axis=-1) 

    conv7= Conv1D(64, 3, activation='relu', padding='same')(up2)
    conv7= Conv1D(64, 3, activation='relu', padding='same')(conv7)
    up3 = concatenate([Conv1DTranspose(512, 2, strides=2, activation='relu',padding='same')(conv7),conv2],axis=-1)

    conv8= Conv1D(64, 3, activation='relu', padding='same')(up3)
    conv8= Conv1D(64, 3, activation='relu', padding='same')(conv8)
    up4 = concatenate([Conv1DTranspose(512, 2, strides=2, activation='relu',padding='same')(conv8),conv1],axis=-1)
    decoding = Conv1D(1, 3, padding ='same')(up4)

    model = Model(inputs=[inputs], outputs=[decoding])
    model.compile(optimizer=Adam(lr=3e-5, beta_1=0.9, beta_2=0.999, decay=0.0), loss='mean_squared_error', metrics=["accuracy"])
    return model
