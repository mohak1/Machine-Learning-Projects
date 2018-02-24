# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:05:38 2018

@author: Mohak

Linear Regression in Multiple Variable (features)
This program predicts 'Avrage Score' of players based on 4 features: Height, Weight, Avrage Goals, Successful Throws  

Feature selection not accurate at all
Accuracy: not acceptable
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#inport file

ds = pd.read_csv("E:\ML projects\Multiple features - mlr09\mlr09.csv",
                 usecols = ['Height in feet', 'Weight in pounds', 'successful goals', 'successful throws', 'avrage points'])
#print(ds)

height = np.array(ds['Height in feet'])     #b1
weight = np.array(ds['Weight in pounds'])   #b2
goals = np.array(ds['successful goals'])    #b3
throws = np.array(ds['successful throws'])  #b4
Y = np.array(ds['avrage points'])           #Y

#ax = plt.subplots()
#scat = ax.scatter(height, weight, goals, throws, Y, marker = 'o')
#plt.show()

#no of records = 54
#training = 50   ~(0-49)
#testing  = 4    ~(50-53)

#ht = b0 + b1*x1 + b2*x2 + b3*x3 + b4*x4

b0 = 1
b1 = 1
b2 = 1
b3 = 1
b4 = 1
a=0.02

m = 50

for i in range(1):
    for j in range(50):
        ht = b0 + b1*height[i] + b2*weight[i] + b3*goals[i] + b4*throws[i]
#        print("b0 = {}, b1 = {}, b2 = {}, b3 = {}, b4 = {}\n".format(b0, b1, b2, b3, b4))
        error = (abs(Y[i]-ht)/50)**2
        #update the variables
        b0-= a*(error)
        b1-= a*(error)
        b2-= a*(error)
        b3-= a*(error)
        b4-= a*(error)
    
print("Trained Parameters:")
#print("b0 = {}, b1 = {}, b2 = {}, b3 = {}, b4 = {}".format(b0, b1, b2, b3, b4))

#Testing against data:
acc=0
for i in range(50, 54):
    predictedPoints = b0 + b1*height[i] + b2*weight[i] + b3*goals[i] + b4*throws[i]
    y = Y[i]
    print("Predicted: {}, Output: {}\n".format(predictedPoints, y))
    acc += abs(predictedPoints-y)
    
#print("Accuracy = {}".format(acc/4*100))
