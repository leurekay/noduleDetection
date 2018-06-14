#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:14:43 2018

@author: ly
"""

import numpy as np
import time

def generate_num(n_samples,shuffle=True):
    
    ids=np.array(np.arange(n_samples))

    while True:
        if shuffle:
            np.random.shuffle(ids)
        for i in ids:
            yield np.array([1,1])*i
        print ('epoch done!')
        time.sleep(10)
        

box=[]
for i in generate_num(20):
    box.append(i[0])
    
    
##path='/data/lungCT/luna/temp/luna_npy/1.3.6.1.4.1.14519.5.2.1.6279.6001.170706757615202213033480003264_label.npy'
##aa=np.load(path)
#
#with open('a.txt','a') as f:
#    f.writelines('asd'+'\n')