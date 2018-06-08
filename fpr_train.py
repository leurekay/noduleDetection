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
from keras.utils import np_utils

import numpy as np
import pandas as pd
import config
import argparse



tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
set_session(tf.Session(config=tfconfig))


#load config
config=config.config



BATCH_SIZE=64

EPOCHS=100
InitialEpoch=0
data_dir=config['data_prep_dir'] #including  *_clean.npy and *_label.npy
model_dir=config['model_dir_fpr']
candidate_path='/data/lungCT/luna/candidates.csv'    


#command line parameter setting
parser = argparse.ArgumentParser()
parser.add_argument('--startepoch', default=InitialEpoch, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--modeldir', default=model_dir, type=str)
args = parser.parse_args()
InitialEpoch=args.startepoch
model_dir=args.modeldir



#deal saved model dir. Mapping epoch id to file
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    epoch_file_dict={}
else:
    saved_models=os.listdir(model_dir)
    saved_models=[x for x in saved_models if x.endswith('.h5')]
    epoch_ids=map(lambda x : int(x.split('-')[0].split(':')[-1]),saved_models)
    epoch_file_dict=zip(epoch_ids,saved_models)
    epoch_file_dict=dict(epoch_file_dict)
    
    
    
#judge how to load model which restore or Start from scratch
if InitialEpoch==0:
    model=layers.fpr_net()
else:
    if InitialEpoch in  epoch_file_dict:
        model_path=os.path.join(model_dir,epoch_file_dict[InitialEpoch])
        model=load_model(model_path,compile=False)
        print ("************\nrestore from %s\n************"%model_path)
    else:
        raise Exception("epoch-%d has not been trained"%InitialEpoch)




#compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])


#checkpoint for callback
checkpoint=ModelCheckpoint(filepath=os.path.join(model_dir,'epoch:{epoch:03d}-trainloss:({loss:.3f}-{loss:.3f})-valloss:({val_loss:.3f}-{val_loss:.3f}).h5'), 
                                monitor='val_loss', 
                                verbose=0, 
                                save_best_only=False, 
                                save_weights_only=False, 
                                period=1)


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


#callback list
callback_list = [checkpoint]








#read data and processing by CPU ,during training.
#Don't load all data into memory at onece!
def generate_arrays(phase,batch_size,shuffle=True):
    dataset=data2.FPR(data_dir,candidate_path,config,phase)
    indexs=dataset.indexs
        
    while True:
        np.random.shuffle(indexs)
        batches=[]
        for i in range(len(indexs)):
            s=i
            e=s+batch_size
            i=e
            if e<len(indexs):
                batches.append(indexs[s:e])
            elif s<(len(indexs)):
                batches.append(indexs[s:])
        for batch in batches:
            x_batch=[]
            y_batch=[]
            for index in batch:
                x,y=dataset.get_item(index)
                x=np.expand_dims(x,axis=-1)
                x_batch.append(x)
                y_batch.append(y)
                
            x_batch=np.concatenate(x_batch) 
            y_batch=np.array(y_batch)
            yield (x_batch,y_batch)





#read data and processing by CPU ,during training.
#Don't load all data into memory at onece!
def generate_arrays2(phase,shuffle=True):
    dataset=data2.FPR(data_dir,candidate_path,config,phase)
    n_samples=dataset.n_pos

    while True:
        for i in range(n_samples):
            box=[]
            y=[]
            for _ in range(4):
                coin=np.random.randint(0,2)
                if coin==1:
                    x = dataset.get_item(isPos=True)
                else:
                    x = dataset.get_item(isPos=False)
                x=np.expand_dims(x,axis=-1)

                box.append(x)
                y.append(coin)
           
            box=np.concatenate(box,axis=0)
            y=np.array(y)
            
#            y = np_utils.to_categorical(y, num_classes=2)
            yield (box,y)


n_train=data2.FPR(data_dir,candidate_path,config,'train').len
n_val=data2.FPR(data_dir,candidate_path,config,'val').len


#training
model.fit_generator(generate_arrays('train',n_train/BATCH_SIZE),
                     steps_per_epoch=2000,
                     epochs=100,
                     verbose=1,
                     callbacks=callback_list,
                     validation_data=generate_arrays('val',n_val/BATCH_SIZE),
                     validation_steps=100,
                     workers=1,)









##read data and processing by CPU ,during training.
##Don't load all data into memory at onece!
#def generate_arrays(phase,shuffle=True):
#    dataset=data2.FPR(data_dir,candidate_path,config,phase)
#    n_samples=dataset.n_pos
#
#    while True:
#        for i in range(n_samples):
#            box=[]
#            y=[]
#            for _ in range(2):
#                coin=np.random.choice([True,False])
#                x = dataset.get_item(isPos=coin)
#                x=np.expand_dims(x,axis=-1)
#                box.append(x)
#                y.append(int(coin))
#           
#            box=np.concatenate(box,axis=0)
#            y=np.array(y)
##            y = np_utils.to_categorical(y, num_classes=2)
#            yield (box,y)
