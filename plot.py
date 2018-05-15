#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 17:12:14 2018

@author: ly
"""

import numpy as np
import matplotlib.pyplot as plt

data=np.load('froc.npy')
y,x=data[:,0],data[:,1]


fig=plt.figure(figsize=[10,8])
plt.plot(x,y,'ob-',linewidth=2)
plt.xlabel('Average false positive per nodule',fontsize='xx-large')
plt.ylabel('Recall',fontsize='xx-large')
plt.grid()
#ax.set(xlabel='Average false positive per nodule', ylabel='Recall',
#       title='')
#ax.grid()


plt.savefig("images/froc.png")
plt.show()