#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:52:58 2018

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


#tfconfig = tf.ConfigProto(device_count={'cpu':0})
#tfconfig.gpu_options.allow_growth = True
#set_session(tf.Session(config=tfconfig))


SAVED_MODEL='/data/lungCT/luna/temp/model3/epoch:024-trainloss:0.21-0.07-valloss:0.58-0.43.h5'
data_dir='/data/lungCT/luna/temp/luna_npy'


    
def sigmoid(x):
    return 1/(1+np.exp(-x))





nms_th=0.6
iou_th=0.3


dataset=data.DataBowl3Detector(data_dir,config.config,phase='test')

get=layers.GetPBB(data.config)


#load model
if os.path.exists(SAVED_MODEL):
    print ("*************************\n restore model\n*************************")
    model=load_model(SAVED_MODEL,compile=False)  
else:
    raise Exception("no model")


labels=dataset.sample_bboxes




image,patches,bboxes = dataset.test_patches(40)

box=[]
for i,patch in enumerate(patches):
    print (i)
    sx,sy,sz=patch
    ex,ey,ez=sx+128,sy+128,sz+128
    x=image[:,sx:ex,sy:ey,sz:ez,:]
    pred=model.predict(x)
    pred=pred[0]
    pred[:,:,:,:,0]=sigmoid(pred[:,:,:,:,0])
     
    pos_pred=get.__call__(pred,0.2)
#    pos_pred=layers.nms(pos_pred,nms_th)
    pos_pred[:,1]+=sx
    pos_pred[:,2]+=sy
    pos_pred[:,3]+=sz
    box.append(pos_pred)
box=np.concatenate(box)


    









