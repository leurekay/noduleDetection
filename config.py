#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 16:02:40 2018

@author: ly
"""

config = {}
config['anchors'] = [ 6.0, 12.0, 24.]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 3. #mm
config['sizelim2'] = 15
config['sizelim3'] = 22
config['aug_scale'] = False
config['r_rand_crop'] = 0.3
config['pad_value'] = 170
config['augtype'] = {'flip':False,'swap':False,'scale':False,'rotate':False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','990fbe3f0a1b53878669967b9afd1441','adc3bbc63d40f8761c59be10f1e504c3']
config['train_over_total']=0.8


