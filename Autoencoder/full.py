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
    conv4_2 = Conv1D(512, 3, activation='relu', padding='same')(conv4_2)
    pool4_2 = MaxPooling1D(pool_size=2)(conv4_2)  #64*512

    flattened=Flatten()(pool4_2)
    decoding = Dense(2048)(flattened)

    model = Model(inputs=[inputs], outputs=[decoding])
    model.compile(optimizer=Adam(lr=3e-5, beta_1=0.9, beta_2=0.999, decay=0.0), loss='mean_squared_error', metrics=["accuracy"])
    return model
