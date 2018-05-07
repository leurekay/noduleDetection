#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:03:17 2018

@author: ly
"""

import os

import numpy as np

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model


import data
import config

SAVED_MODEL='/data/lungCT/luna/temp/model/my_model.h5'
data_dir='/data/lungCT/luna/temp/luna_npy'

#load model
if os.path.exists(SAVED_MODEL):
    print ("*************************\n restore model\n*************************")
    model=load_model(SAVED_MODEL)  
else:
    raise Exception("no model")
    

dataset=data.DataBowl3Detector(data_dir,config.config,phase='val')

x, y ,_ = dataset.__getitem__(10)
x=np.expand_dims(x,axis=0)
y=np.expand_dims(y,axis=0)

pred=model.predict(x)
