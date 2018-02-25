# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 03:31:23 2018

@author: Mohak

Data Used: Unknown
Varialbes: Unknown
Output expected: Binary(0 or 1)
Problem Type: Classification
Algorithm Used: Logistic Regression
Function Used: Sigmoid
Cost(or Error) Function: Updated Loss Function
Optimisation Function: Gradient Descent (Using for the first time)
A program to predict the output (0 or 1) based on given set of inputs
"""

import numpy as np
import pandas as pd
import math 

ds = pd.read_csv("E:\ML projects\Logistic Regression\LR01.csv")
#total data available: 768 rows
#training: (0-700)
#testing: (700-769)

X1 = np.array(ds['X1'])
X2 = np.array(ds['X2'])
X3 = np.array(ds['X3'])
X4 = np.array(ds['X4'])
X5 = np.array(ds['X5'])
X6 = np.array(ds['X6'])
X7 = np.array(ds['X7'])
Y = np.array(ds['Y'])

b0=b1=b2=b3=b4=b5=b6=b7=1
db0=db1=db2=db3=db4=db5=db6=db7=0
a = 0.01
#Hypotesis Function
#ht = b0 + b1*X1 + b2*X2 + b3*X3 + b4*X4 + b5*X5 + b6*X6

#Sigmoid
def Sigmoid(t):
    return (1/(1+math.exp(-t)))


for k in range(3):
    for i in range(30):
        ht = b0 + b1*X1[i] + b2*X2[i] + b3*X3[i] + b4*X4[i] + b5*X5[i] + b6*X6[i] + b7*X7[i]
        
        yHat = Sigmoid(ht)  #value of yHat is b/w 0-1
        y = Y[i]
#        print(yHat)
        #Cost Function
        error = y*math.log(abs(yHat)+0.000001) + (1-y)*math.log(abs(1-yHat)+0.000001)
        #Optimization Function - GRADIENT DESCENT
        dz = yHat-y
#        print(dz)
        db0= dz
        db1= dz
        db2= dz
        db3= dz
        db4= dz
        db5= dz
        db6= dz
        db7= dz
        #updating the parameters:
        b0-= a*db0
        b1-= a*db1
        b2-= a*db2
        b3-= a*db3
        b4-= a*db4
        b5-= a*db5
        b6-= a*db6
        b7-= a*db7
        
acc=count=0
for i in range(700, 768):
    ht = b0 + b1*X1[i] + b2*X2[i] + b3*X3[i] + b4*X4[i] + b5*X5[i] + b6*X6[i] + b7*X7[i]
    print("predicted = {}, expected = {}".format(ht, Y[i]))
    acc += 1-(abs(Y[i]-ht)/Y[i])
    count+=1    
#print("Accuracy = ",acc*100/count)
        
