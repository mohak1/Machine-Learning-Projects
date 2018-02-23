# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 22:48:55 2018

@author: Mohak

Solving a Linear Regression problem
Dataset: slr01.xls
"""
import numpy as np
import pandas as pd

ds = pd.read_csv("E:\slr01\slr01.csv", usecols = ['X', 'Y'])

#print("Database Details:\nDatabase Size:",ds.shape)
#print("\nHead:",ds.head())

X = np.array(ds['X'])
Y = np.array(ds['Y'])

# ht = b0 + b1 * x 
b0 = 1
b1 = 1
a = 3 #learning Rate

for k in range(5):
    for i in range(0, 20):
        x = X[i]
        y = Y[i]
        ht = b0 + (b1 * x)
        #mean squared error:
        error = ((ht - y)/X.size)**2 
        
        b0 = b0 - a*error
        b1 = b1 - a*(error)
        #print("\nUpdated b0 = {}, b1 = {}".format(b0, b1))
        
print("Value parameters after training:\n")
print("b0 = {}, b1 = {}".format(b0, b1))

print("\nTesting the trained parameters:\n")
acc = 0
for i in range(20, 23   ):
    x = X[i]
    y = Y[i]
    testOut = b0 + b1*x
    acc += (1-abs(testOut-y))
    #print("Predicted = {}, Y = {}, Accuracy = {}".format(testOut, y, acc))
print("Accuracy = {}".format(acc*100/3))
