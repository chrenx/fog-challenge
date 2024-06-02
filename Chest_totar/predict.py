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
# import cv2
from tensorflow.python.keras.backend import set_session ###from keras.backend.tensorflow_backend import set_session
###sys.path.append("../../../model/")
import cnn_sigmoid_leaky_1024 as cnn

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' ###

###print('example 262144 ../../../processedData/real-pd.training_data/smartphone_accelerometer/')
size=5120 ###512

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

the_model="model."+sys.argv[1] ####
model = cnn.cnn1d(size,3)
model.load_weights(the_model)
###path1=the_path
IDS=open('test.id','r')
all_files=[]
the_label={}

for line in IDS:
    line=line.strip()
    files=glob.glob("../../data/processed_files/Chest/*"+line+"*/*")
    for fff in files:
        all_files.append(fff)
IDS.close()

ACT=open('activity.list','r')
i=0
act_to_id={}
for line in ACT:
    line=line.strip()
    act_to_id[line]=i
    i=i+1
ACT.close()

all_acts=act_to_id.keys()
total_act=len(all_acts)

PRED=open(('pred.'+the_model),'w')
TESTLABEL=open(('gs.txt'),'w')
for test_line in all_files:
    for aaa in all_acts:
        ttt=test_line.split('/')
        if aaa == ttt[-1]:
            the_label=act_to_id[aaa]
    label=np.zeros(21)
    label[the_label]=1
    lll=label[0]
    TESTLABEL.write(str(lll))
    for lll in label[1:]:
        TESTLABEL.write('\t')
        TESTLABEL.write(str(lll))
    TESTLABEL.write('\n')
    image=np.genfromtxt(test_line,delimiter=',')[:,1:4]
    image_pad=np.zeros((size,3))
    image_pad[0:image.shape[0],:]=image

    image_batch=[]
    image_batch.append(image_pad)
    image_batch=np.asarray(image_batch)

    output = model.predict(image_batch)[0]
    ooo=output[0]
    PRED.write(str(ooo))
    for ooo in output[1:]:
        PRED.write('\t')
        PRED.write(str(ooo))
    PRED.write('\n')
    
PRED.close()
