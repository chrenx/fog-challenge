from __future__ import print_function
import glob
import os
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
import scipy.io
import math
import random

import cnn_sigmoid_leaky_1024 as cnn
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
#size=512
size=5120 ###Maximum size of all movement folder files is 10846 changed 

channel=3
batch_size=1 ####chang from 16 to 1 for debugging

model = cnn.cnn1d(size,3)

max_scale=1.3
min_scale=0.8


def norm_axis(a,b,c):
    newa=a/(math.sqrt(float(a*a+b*b+c*c)))
    newb=b/(math.sqrt(float(a*a+b*b+c*c)))
    newc=c/(math.sqrt(float(a*a+b*b+c*c)))
    return ([newa,newb,newc])

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def rotateC(image,theta,a,b,c): ## theta: angle, a, b, c, eular vector
    axis=norm_axis(a,b,c)
    imagenew=np.dot(image, rotation_matrix(axis,theta))
    return imagenew

def scaleImage(image,scale):
    [x,y]= image.shape
    y1=int(y*scale)
    x1=3
    image=cv2.resize(image,(y1,x1))
    new=np.zeros((x,y))
    if (y1>y):
        start=0
        end=start+y
        new=image[:,start:end]
    else:
        new_start=0
        new_end=new_start+y1
        new[:,new_start:new_end]=image
    return new
  
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

IDS=open('train.id','r')
all_files=[]
the_label={}

for line in IDS:
    line=line.strip()
    files=glob.glob("../../data/processed_files/Chest/*"+line+"*/*")
    for fff in files:
        all_files.append(fff)
        for aaa in all_acts:
            ttt=fff.split('/')
            if aaa == ttt[-1]:
                the_label[fff]=act_to_id[aaa]
IDS.close()

print(len(all_files))
random.shuffle(all_files)
partition_ratio=0.8
train_line=all_files[0:int(len(all_files)*partition_ratio)]
test_line=all_files[int(len(all_files)*partition_ratio):len(all_files)]
#print("train_line:\n",train_line)###

def generate_data(train_line, batch_size, if_train):
    """Replaces Keras' native ImageDataGenerator."""
##### augmentation parameters ######
    i = 0
    channel = 3
    while True:
        image_batch = []
        label_batch = []
        for b in range(batch_size):
            if i == len(train_line):
                i = 0
                random.shuffle(train_line)
            sample = train_line[i]
            i += 1
            image=np.genfromtxt(sample,delimiter=',')[:,1:4]
            label=np.zeros(21)
            label[the_label[sample]]=1
            #rrr=random.random() 
            #rrr_scale=rrr*(max_scale-min_scale)+min_scale 
            if (if_train==1):
                #seq=scaleImage(seq,rrr_scale)
                theta=random.random()*math.pi*2
                theta=random.random()*360
                #print(theta)
                a=random.random()
                b=random.random()
                c=random.random()
                image=rotateC(image,theta,a,b,c)
                pass
            image_pad=np.zeros((size,3))
            image_pad[0:image.shape[0],:]=image
            label=np.zeros(total_act)
            label[the_label[sample]]=1
            '''
            if (the_max-the_min)==0:
                label=0.5
            else:
                label=(float(table[0])-the_min)/(the_max-the_min)
            '''
            image_batch.append(image_pad)
            label_batch.append(label)
        image_batch=np.array(image_batch)
        label_batch=np.array(label_batch)
#        print(image_batch.shape,label_batch.shape)
        yield image_batch, label_batch

#model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=False)
#name_model=sys.argv[4] + '.h5'
#name_model= f"model.accel{model}" #'model.accel{}'.format(model) #############
name_model='model.'+sys.argv[1]
callbacks = [
#    keras.callbacks.TensorBoard(log_dir='./',
#    histogram_freq=0, write_graph=True, write_images=False),
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    verbose=0, save_weights_only=False,save_best_only=True,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(train_line, batch_size,True),
    steps_per_epoch=int(len(train_line) // batch_size), nb_epoch=20,  ###changed
    validation_data=generate_data(test_line,batch_size,False),
    validation_steps=int(len(test_line) // batch_size),callbacks=callbacks)

