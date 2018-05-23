#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:52:58 2018

@author: ly
"""

from keras.backend.tensorflow_backend import set_session




import os
import time
import argparse

import numpy as np
import pandas as pd

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model


import data
import config
import layers



#tfconfig = tf.ConfigProto(device_count={'cpu':0})
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
set_session(tf.Session(config=tfconfig))


#load config
config=config.config





#command line parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='val', type=str, 
                    help='train,val,test')
parser.add_argument('--model', default='/data/lungCT/luna/temp/model6/epoch:089-trainloss:(0.124-0.062)-valloss:(0.515-0.452).h5',
                    type=str, help='model path')

args = parser.parse_args()


data_phase=args.phase
SAVED_MODEL=args.model


nms_th=0.6
pos_th=0.2
data_dir=config['data_prep_dir']
split_dir=config['valsplit_dir']
ctinfo_path=config['ctinfo_path']
pred_save_dir=config['pred_save_dir']
if not os.path.exists(pred_save_dir):
    os.makedirs(pred_save_dir)

ctinfo=pd.read_csv(ctinfo_path,index_col='seriesuid')
uid_origin_dict={}
uids=list(ctinfo.index)
for uid in uids:
    origin=list(ctinfo.loc[uid])[:3]
    uid_origin_dict[uid]=origin
    
def sigmoid(x):
    return 1/(1+np.exp(-x))








#load model
if os.path.exists(SAVED_MODEL):
    print ("*************************\n restore model from %s\n*************************"%SAVED_MODEL)
    model=load_model(SAVED_MODEL,compile=False)  
else:
    raise Exception("no model")


dataset=data.DataBowl3Detector(data_dir,split_dir,config,phase=data_phase)
get=layers.GetPBB(data.config)

uids=dataset.uids
len_uids=len(uids)
labels=dataset.sample_bboxes

pred_df=pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','probability'])
count=0

for index in range(len_uids):
    time_s=time.time()
    
    image,patch_box,bboxes = dataset.package_patches(index)
    uid=dataset.uids[index]
    origin=uid_origin_dict[uid]
    
    _,xsize,ysize,zsize,_=image.shape
 
    
    box=[]
    for i,(patch,coord,start) in enumerate(patch_box):
#        print (i)
        sx,sy,sz=start
   
        pred=model.predict([patch,coord])
        pred=pred[0]
        pred[:,:,:,:,0]=sigmoid(pred[:,:,:,:,0])
         
        pos_pred=get.__call__(pred,pos_th)
    #    pos_pred=layers.nms(pos_pred,nms_th)
        pos_pred[:,1]+=sx
        pos_pred[:,2]+=sy
        pos_pred[:,3]+=sz
        
#        pos_pred[:,1]+=origin[0]
#        pos_pred[:,2]+=origin[1]
#        pos_pred[:,3]+=origin[2]
        box.append(pos_pred)
    box=np.concatenate(box)
    box_nms=layers.nms(box,nms_th)
    time_e=time.time()
    print ('%s-%03d, nodules:%2d, pos:%3d, pos_nms:%3d, patches:%3d, shape-[%3d,%3d,%3d], time:%.1fs'%(data_phase,index,bboxes.shape[0],box.shape[0],box_nms.shape[0],len(patch_box),xsize,ysize,zsize,time_e-time_s))

    for entry in box_nms:
        pred_df.loc[count]=[uid,entry[1],entry[2],entry[3],entry[0]]
        count+=1

save_path=os.path.join(pred_save_dir,str(int(time.time()))+'-'+data_phase+'.csv')
pred_df.to_csv(save_path,index=None)  
print ("output %d postive predictions"%pred_df.shape[0])
print ("csv file saved in %s"%save_path)
    


    









