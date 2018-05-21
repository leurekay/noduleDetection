#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:55:37 2018

@author: ly
"""
import os
import numpy as np

import pandas as pd

import data
import config

import time

#fff=np.load('splitdata/val.npy')

ratio_train=0.8
ratio_val=0.1
ratio_test=0.1
data_dir='/data/lungCT/luna/temp/luna_npy'
shuffle=True
savedir='splitdata'

if not os.path.exists(savedir):
    os.makedirs(savedir)


patient_list=os.listdir(data_dir)
if shuffle:
    pass

ct_list=filter(lambda x:x.split('_')[-1]=='clean.npy',patient_list)
label_list=filter(lambda x:x.split('_')[-1]=='label.npy',patient_list)
id_list_by_ct=map(lambda x:x.split('_')[0],ct_list)
id_list_by_label=map(lambda x:x.split('_')[0],label_list)
id_list=set.intersection(set(id_list_by_ct),set(id_list_by_label))
idcs=list(id_list)
idcs.sort()

length=len(idcs)
cutpoint1=int(ratio_train*length)
cutpoint2=int((ratio_train+ratio_val)*length)
train=idcs[:cutpoint1]
val=idcs[cutpoint1:cutpoint2]
test=idcs[cutpoint2:]

np.save(os.path.join(savedir,'train.npy'),train)
np.save(os.path.join(savedir,'val.npy'),val)
np.save(os.path.join(savedir,'test.npy'),test)
print ('train:%d, val:%d, test:%d'%(len(train),len(val),len(test)))

time.sleep(3)

for phase in ['train','val','test']:
    save_path_anno=os.path.join(savedir,phase+'_anno.csv')
    
    dataset=data.DataBowl3Detector(data_dir,savedir,config.config,phase)
    print ('===========')

    labels=dataset.sample_bboxes
    uids=dataset.uids
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
    df.to_csv(save_path_anno,index=None)
    time.sleep(2)
