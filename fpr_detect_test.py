#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:07:17 2018

@author: ly
"""

import os

import numpy as np
import pandas as pd


import config
config=config.config

stage1_submit_path='/data/lungCT/luna/temp/submit/model6-epoch46-val.csv'
df=pd.read_csv(stage1_submit_path)

df['p2']=0

data_dir=config['data_prep_dir']
for i in range(df.shape[0]):
    df.iloc[i,5]=i
    point=list(df.iloc[i,1:4].values)
    uid=df.iloc[i,0]
    
    cube_size=np.array([32,32,32],'int')
    
 
    
    path=os.path.join(data_dir,uid+'_clean.npy')
    img=np.load(path)
    img_shape=img.shape
  
    xyz=point[::-1]
    
    xyz=np.array(xyz,'int')
    start=xyz-cube_size/2
    start=start.astype('int')
    comp=np.vstack((start,np.array([0,0,0])))
    start=np.max(comp,axis=0)
    end=start+cube_size
    end=end.astype('int')
    comp1=np.vstack((end,np.array(img_shape[1:])))
    end=np.min(comp1,axis=0)
#        print (end)
    
    
    
    cube=img[:,start[0]:end[0],start[1]:end[1],start[2]:end[2]]
    delta=32-(end-start)
    
    cube=np.pad(cube, ((0,0),(0,delta[0]),(0,delta[1]),(0,delta[2])), 
                         'constant', constant_values=170)
    
    cube = (cube.astype(np.float32)-128)/128
    
