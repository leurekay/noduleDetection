#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 14:20:17 2018

@author: ly
"""
from keras.backend.tensorflow_backend import set_session

import data

import layers


import time
import os
from PIL import Image

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model,load_model
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

import numpy as np
import pandas as pd
import config

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
set_session(tf.Session(config=tfconfig))



config=config.config




EPOCHS=100
data_dir='/data/lungCT/luna/temp/luna_npy'
SAVED_MODEL='/data/lungCT/luna/temp/model/my_model.h5'

model_dir=SAVED_MODEL.split(SAVED_MODEL.split('/')[-1])[0]    
if not os.path.exists(model_dir):
    os.makedirs(model_dir)    
        
#dataset=data.DataBowl3Detector(data_dir,config)
#patch,label,coord=dataset.__getitem__(295)
#a=label[:,:,:,:,0]



#loss function
myloss=layers.myloss

loss_cls=layers.loss_cls



def lr_decay(epoch):
	lr = 0.0001
	if epoch >= 150:
		lr = 0.0003
	if epoch >= 220:
		lr = 0.00003
	return lr 
lr_scheduler = LearningRateScheduler(lr_decay)


#load model
if os.path.exists(SAVED_MODEL):
    print ("*************************\n restore model\n*************************")
    model=load_model(SAVED_MODEL)  
else:
    model=layers.n_net()
    
adam=keras.optimizers.Adam(lr=1000, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
sgd=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=adam,
              loss=myloss,
              metrics=[loss_cls])



# numbers of sample correspoding train and val
train_dataset=data.DataBowl3Detector(data_dir,config,phase='train')
train_samples=train_dataset.__len__()
val_dataset=data.DataBowl3Detector(data_dir,config,phase='val')
val_samples=val_dataset.__len__()



#call back.   save model named by (time,train_loss,val_loss)
class EpochSave(keras.callbacks.Callback):
    def on_epoch_begin(self,epoch, logs={}):
        self.losses = []
        

    def on_epoch_end(self, epoch, logs={}):
        time_now=int(time.time())
        train_loss=logs.get('loss')
        val_loss=logs.get('val_loss')
        self.losses.append([train_loss,val_loss])
#        print ('epoch:',epoch,'    ',self.losses)
        file_name=str(time_now)+'_'+time.strftime('%Y%m%d-%H:%M:%S')+'_train_%.3f_val_%.3f.h5'%(train_loss,val_loss)
        model.save(os.path.join(model_dir,file_name),include_optimizer=False)
epoch_save = EpochSave()


#read data and processing by CPU ,during training.
#Don't load all data into memory at onece!
def generate_arrays(phase,shuffle=True):
    dataset=data.DataBowl3Detector(data_dir,config,phase=phase)
    n_samples=dataset.__len__()
    ids=np.array(np.arange(n_samples))

    while True:
        if shuffle:
            np.random.shuffle(ids)
        for i in ids:
            x, y ,_ = dataset.__getitem__(i)
            x=np.expand_dims(x,axis=0)
            y=np.expand_dims(y,axis=0)
            yield (x, y)

model.save(SAVED_MODEL,include_optimizer=False)
model.fit_generator(generate_arrays(phase='train'),
                    steps_per_epoch=train_samples,epochs=EPOCHS,
                    verbose=1,callbacks=[epoch_save,lr_scheduler],
                    validation_data=generate_arrays('val'),
                    validation_steps=val_samples,
                    workers=4,)



model.save(SAVED_MODEL,include_optimizer=False)