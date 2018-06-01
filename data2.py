#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:31:00 2018

@author: ly
"""

import numpy as np
import pandas as pd
import os
import time
import collections
import random
from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate

import config

config=config.config











class FPR():
    def __init__(self, data_dir,candidates_path,config, phase = 'train'):
        assert(phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase     
        self.stride = config['stride']       


        self.isScale = config['aug_scale']

        self.augtype = config['augtype']
        self.pad_value = config['pad_value']

        self.crop_size = config['crop_size']



        data_dir=os.path.join(data_dir,phase)
        self.data_dir=data_dir
        patient_list=os.listdir(data_dir)
        
        ct_list=filter(lambda x:x.split('_')[-1]=='clean.npy',patient_list)
        label_list=filter(lambda x:x.split('_')[-1]=='label.npy',patient_list)
        id_list_by_ct=map(lambda x:x.split('_')[0],ct_list)
        id_list_by_label=map(lambda x:x.split('_')[0],label_list)
        id_list=set.intersection(set(id_list_by_ct),set(id_list_by_label))
        idcs=list(id_list)
        idcs.sort()
        self.uids=idcs
        
        df_candidates=pd.read_csv(candidates_path)
        df_candidates=df_candidates[df_candidates['seriesuid'].isin(idcs)]
     
        
        for idx in idcs:
            origin,extend,_=np.load(os.path.join(data_dir, '%s_info.npy' %idx))
         
            origin=np.flip(origin,0)
            extend=np.flip(extend,0)
            
            df_candidates.loc[df_candidates['seriesuid']==idx,['coordX','coordY','coordZ']]-=(origin+extend)
        
        df_candidates.loc[((df_candidates['coordX']<0) | (df_candidates['coordY']<0) | (df_candidates['coordZ']<0)),['class']]=np.nan
        df_candidates.dropna(inplace=True)
        self.df_candidates=df_candidates
        
        self.pos=df_candidates[df_candidates['class']==1]
        self.neg=df_candidates[df_candidates['class']==0]
        
        self.n_pos=self.pos.shape[0]
        self.n_neg=self.neg.shape[0]
    def get_item(self,isPos):
        cube_size=np.array([32,32,32])
        
        if isPos:
            index=np.random.randint(0,self.n_pos)
            entry=self.pos.iloc[index]
        else:
            index=np.random.randint(0,self.n_neg)
            entry=self.neg.iloc[index]
        entry=list(entry.values)
        
        path=os.path.join(self.data_dir,entry[0]+'_clean.npy')
        img=np.load(path)
        img_shape=img.shape
        
        xyz=entry[1:4]
        xyz=list(reversed(xyz))
        xyz=np.array(xyz,'int')
        start=xyz-cube_size/2
        comp=np.vstack((start,np.array([0,0,0])))
        start=np.max(comp,axis=0)
        end=start+cube_size
        comp1=np.vstack((end,np.array(img_shape[1:])))
        end=np.min(comp1,axis=0)
        
        cube=img[:,start[0]:end[0],start[1]:end[1],start[2]:end[2]]
        delta=32-(end-start)
        
        cube=np.pad(cube, ((0,0),(0,delta[0]),(0,delta[1]),(0,delta[2])), 
                             'constant', constant_values=170)
        
#        print (xyz)
#        print (start)
#        print (end)
        return cube
            
        
#        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
#
#        
#        labels = []
#        origins=[]
#        extendboxes=[]
#        
#        for idx in idcs:
#            l = np.load(os.path.join(data_dir, '%s_label.npy' %idx))
#            if np.all(l==0):
#                l=np.array([])
#            labels.append(l)
#            origin,extend,_=np.load(os.path.join(data_dir, '%s_info.npy' %idx))
#            origins.append(origin)
#            extendboxes.append(extend)
#
#        self.sample_bboxes = labels
#        self.sample_origins=origins
#        self.sample_extendboxes=extendboxes

if __name__=='__main__':
     data_dir='/data/lungCT/luna/temp/luna_npy'
     label_path='/data/lungCT/luna/pull_aiserver/candidates.csv'    
     data=FPR(data_dir,label_path,config)

     df=data.df_candidates
     a=df[((df['coordX']<0) | (df['coordY']<0) | (df['coordZ']<0))]    
    
    
    
# =============================================================================
#     data_dir='/data/lungCT/luna/temp/luna_npy'
#     label_path='/data/lungCT/luna/pull_aiserver/candidates.csv'
#     annotations_path='/data/lungCT/luna/annotations.csv'
#     df_a=pd.read_csv(annotations_path)
#     df_xyz01=pd.read_csv(label_path)
#     data=FPR(data_dir,label_path,config)
#     df=data.df_candidates
#     a=df[((df['coordX']<0) | (df['coordY']<0) | (df['coordZ']<0)) & (df['class']==1)]
#     aa=a.groupby('seriesuid').count()
#     uid='1.3.6.1.4.1.14519.5.2.1.6279.6001.801945620899034889998809817499'
#     uid='1.3.6.1.4.1.14519.5.2.1.6279.6001.534083630500464995109143618896'
#     info=np.load(os.path.join(data_dir,uid+'_info.npy'))
#     label=np.load(os.path.join(data_dir,uid+'_label.npy'))
#     world_xyz=df_a[df_a['seriesuid']==uid].iloc[:,1:4]
#     xyz01=df_xyz01[(df_xyz01['seriesuid']==uid) & (df_xyz01['class']==1)].iloc[:,1:4]
#     jj=info[0,:]+info[1,:]+label[0,:3]
# =============================================================================
