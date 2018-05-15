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
            yield (i)
        print ('epoch done!')
        time.sleep(10)
        

box=[]
for i in generate_num(20):
    box.append(i)