#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 18:14:46 2018

@author: ly
"""

import numpy as np
import pandas as pd

import data
import config


data_dir='/data/lungCT/luna/temp/luna_npy'
save_path='/data/lungCT/luna/temp/my_anno.csv'
dataset=data.DataBowl3Detector(data_dir,config.config,phase='test')


labels=dataset.sample_bboxes[780:880]
uids=dataset.uids[780:880]



df=pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','diameter_mm'])
count=0

for index,uid in enumerate(uids):
    print (index)
    label=labels[index]
    if len(label)==0:
        continue
    for entry in label:
        df.loc[count]=[uid,entry[0],entry[1],entry[2],entry[3]]
        count+=1
df.to_csv(save_path,index=None)
  
#    for i,patch in enumerate(patches):
##        print (i)
#        sx,sy,sz=patch
#        ex,ey,ez=sx+128,sy+128,sz+128
#        x=image[:,sx:ex,sy:ey,sz:ez,:]
#        pred=model.predict(x)
#        pred=pred[0]
#        pred[:,:,:,:,0]=sigmoid(pred[:,:,:,:,0])
#         
#        pos_pred=get.__call__(pred,0.2)
#    #    pos_pred=layers.nms(pos_pred,nms_th)
#        pos_pred[:,1]+=sx
#        pos_pred[:,2]+=sy
#        pos_pred[:,3]+=sz
#        
##        pos_pred[:,1]+=origin[0]
##        pos_pred[:,2]+=origin[1]
##        pos_pred[:,3]+=origin[2]
#        box.append(pos_pred)
#    box=np.concatenate(box)
#    box_nms=layers.nms(box,0.6)
#    time_e=time.time()
#    print ('CT-%03d, nodules:%2d , pos:%3d , pos_nms:%3d , time:%.1fs'%(index,bboxes.shape[0],box.shape[0],box_nms.shape[0],time_e-time_s))
#    for entry in box_nms:
#        pred_df.loc[count]=[uid,entry[1],entry[2],entry[3],entry[0]]
#        count+=1
#        
#pred_df.to_csv(os.path.join(pred_save_dir,str(time.time())+'.csv'),index=None)  