#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import time
import scipy.io
import glob
import pickle
from keras.utils.np_utils import to_categorical
from keras import backend as K
import tensorflow as tf
import keras
from keras.models import Model
import autoencoder
size=5120


## record all files for KU data
all_files=glob.glob('/home/l003148/work/vc_batch12/data/processed_files/*/*/*')



all_models=glob.glob('model*')
for the_model in all_models:
    #model = cnn.cnn1d(size,3)
    model = keras.models.load_model(the_model)
    for test_line in all_files:
        t=test_line.split('/')
        os.system('mkdir -p ./feature/'+t[-3]+'/'+t[-2]+'/')
        image = np.genfromtxt(test_line,delimiter=',')[:,1:4]
        image_pad=np.zeros((size,3))
        if (image.shape[0]>5120):
            image_pad[:,:]=image[0:5120,:]
        else:
            image_pad[0:image.shape[0],:]=image

        final_matrix=[]
        final_matrix.append(image_pad)
        final_matrix=np.array(final_matrix)
        for count, layer in enumerate(model.layers):
            if count == 15:
                intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer.name).output)
                intermediate_output = intermediate_layer_model.predict(final_matrix)#[0]
    
        feature=intermediate_output.flatten()
        np.save('./feature/'+t[-3]+'/'+t[-2]+'/'+t[-1],feature)
