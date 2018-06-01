#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:20:38 2018

@author: ly
"""

from keras.backend.tensorflow_backend import set_session

import data
import data2

import layers


import time
import os
import shutil
from PIL import Image

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import numpy as np
import pandas as pd
import config
import argparse



tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
set_session(tf.Session(config=tfconfig))


#load config
config=config.config





EPOCHS=100
InitialEpoch=0
data_dir=config['data_prep_dir'] #including  *_clean.npy and *_label.npy
model_dir=config['model_dir']





model=layers.n_net()



#compile
model.compile(optimizer='adam',
              loss='binary_crossentropy')


#controled learning rate for callback  
def lr_decay(epoch):
    lr=0.001
    if epoch>2:
        lr=0.001
    if epoch>5:
        lr=0.0001
    if epoch>10:
        lr=0.00001
    return lr
lr_scheduler = LearningRateScheduler(lr_decay)





data_dir='/data/lungCT/luna/temp/luna_npy'
label_path='/data/lungCT/luna/pull_aiserver/candidates.csv'    






#read data and processing by CPU ,during training.
#Don't load all data into memory at onece!
def generate_arrays(phase,shuffle=True):
    dataset=data2.FPR(data_dir,label_path,config,phase)
    n_samples=dataset.n_pos

    while True:
        for i in range(n_samples):
            box=[]
            x = dataset.get_item(isPos=True)
            x=np.expand_dims(x,axis=-1)
            box.append(x)
#            x1 = dataset.get_item(isPos=False)
#            x1=np.expand_dims(x1,axis=-1)
#            box.append(x1)
#            
#            box=np.concatenate(box,axis=0)
#            print box.shape
            y=np.array([1])
            
            yield (x,y)


#training
model.fit_generator(generate_arrays(phase='train'),
                    steps_per_epoch=2000,
                    epochs=10,
                    verbose=1,
                    workers=1,)