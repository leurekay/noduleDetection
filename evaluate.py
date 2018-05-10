#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:03:17 2018

@author: ly
"""

from keras.backend.tensorflow_backend import set_session




import os

import numpy as np

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model


import data
import config
import layers


tfconfig = tf.ConfigProto(device_count={'cpu':0})
#tfconfig.gpu_options.allow_growth = True
set_session(tf.Session(config=tfconfig))


SAVED_MODEL='/data/lungCT/luna/temp/model/my_model.h5'
data_dir='/data/lungCT/luna/temp/luna_npy'

#load model
if os.path.exists(SAVED_MODEL):
    print ("*************************\n restore model\n*************************")
    model=load_model(SAVED_MODEL)  
else:
    raise Exception("no model")
    
def sigmoid(x):
    return 1/(1+np.exp(-x))



dataset=data.DataBowl3Detector(data_dir,config.config,phase='train')
get=layers.GetPBB(data.config)




x, y ,_ = dataset.__getitem__(66)
x=np.expand_dims(x,axis=0)
y=np.expand_dims(y,axis=0)
pos_true=get.__call__(y[0],0.3)


#pred=model.predict(x)
#pred=pred[0]
#pred[:,:,:,:,0]=sigmoid(pred[:,:,:,:,0])
#
#pred_=pred.reshape([-1,5])
#pred_=pred_[np.argsort(-pred_[:, 0])]
#
#
#
#
#

#pos_out=get.__call__(pred,0.3)
##
##pos_out_nms=layers.nms(pos_out,0.5)




