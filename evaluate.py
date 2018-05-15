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


#tfconfig = tf.ConfigProto(device_count={'cpu':0})
#tfconfig.gpu_options.allow_growth = True
#set_session(tf.Session(config=tfconfig))


SAVED_MODEL='/data/lungCT/luna/temp/model3/epoch:024-trainloss:0.21-0.07-valloss:0.58-0.43.h5'
data_dir='/data/lungCT/luna/temp/luna_npy'


    
def sigmoid(x):
    return 1/(1+np.exp(-x))




def tp_fp(confidence_th):
    nms_th=0.6
    iou_th=0.3
    
    
    dataset=data.DataBowl3Detector(data_dir,config.config,phase='train')
    
    get=layers.GetPBB(data.config)
    
    
    labels=dataset.bboxes
    num_label=len(labels)
    
    
    
    
    TP=0
    FP=0
    for index in range(num_label):
        x, y ,_ = dataset.__getitem__(index)
        x=np.expand_dims(x,axis=0)
        y=np.expand_dims(y,axis=0)
        pos_label=get.__call__(y[0],confidence_th)
    
        pred=model.predict(x)
        pred=pred[0]
        pred[:,:,:,:,0]=sigmoid(pred[:,:,:,:,0])
         
        pos_pred=get.__call__(pred,confidence_th)
        pos_pred=layers.nms(pos_pred,nms_th)
    
        pos=0
        for bbox in pos_pred:
            iou=layers.iou(bbox[1:],pos_label[0,1:])
            if iou>=iou_th:
                pos=1
            else:
                FP+=1
        TP+=pos
        
#        print (index,TP,FP)
        
    recall=float(TP)/num_label
    averageFP=float(FP)/num_label
        
    print (recall,averageFP)
    return recall,averageFP




#load model
if os.path.exists(SAVED_MODEL):
    print ("*************************\n restore model\n*************************")
    model=load_model(SAVED_MODEL,compile=False)  
else:
    raise Exception("no model")







if __name__=='__main__':
    sample_thresholds=[0.98,0.7,0.4,0.2]
    box=[]
    for th in sample_thresholds:
        print (th)
        a,b=tp_fp(th)
        box.append([a,b])
    ooxx=np.array(box)
    np.save('froc.npy',ooxx)  

