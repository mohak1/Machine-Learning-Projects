# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 02:02:45 2018

@author: Mohak
"""

import numpy as np
import pandas as pd

ds = pd.read_csv("E:\ML projects\Multiple features - mlr03\mlr03.csv", usecols = ['EXAM1','EXAM2','EXAM3','FINAL'])

x1 = np.array(ds['EXAM1'])
x2 = np.array(ds['EXAM2'])
x3 = np.array(ds['EXAM3'])
Y = np.array(ds['FINAL'])

#ht = b0+ b1*x1 + b2*x2 + b3*x3
b0 = 1
b1 = 1
b2 = 1
b3 = 1
a = 0.01

for k in range(2):
    for i in range(0,21):
        ht = b0 + b1*x1[i] + b2*x2[i] + b3*x3[i]
        y = Y[i]
        error = ((ht-y)/20)**2
        b0 = b0-a*error
        b1 = b1-a*error
        b2 = b2-a*error
        b3 = b3-a*error
        
count=0
acc=0
print("alpha",a)
for i in range(21, 25):
    ht = b0 + b1*x1[i] + b2*x2[i] + b3*x3[i]
    print("predicted = {}, expected = {}".format(ht, Y[i]))
    acc += 1-(abs(Y[i]-ht)/Y[i])
    count+=1    
print("Accuracy = ",acc*100/count)
