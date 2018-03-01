# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:46:04 2018

@author: Mohak

Data Used: Unknown
Varialbes: Unknown
Output expected: Binary(0 or 1)
Problem Type: Classification
Algorithm Used: ReLu
Function Used: Sigmoid
Cost(or Error) Function: Mean Squared Error
Optimisation Function: Gradient Descent (Using for the first time)
A program to predict the output (0 or 1) based on given set of inputs
Accuracy: 78.645%
"""

import pandas as pd
import numpy as np
import keras as ks

ds = pd.read_csv("E:\ML projects\Logistic Regression\LR01.csv") #read the csv file
ds = np.array(ds)   #convert the csv to numpy array
X = ds[:,0:8]
Y = ds[:,8]

np.random.seed(11)

#1.Defining the model
model1 = ks.models.Sequential()
#Adding layers ot the model
model1.add(ks.layers.Dense(40, input_dim=8, activation="relu")) #input layer
model1.add(ks.layers.Dense(25, activation='relu'))  #hidden layer 1
model1.add(ks.layers.Dense(15, activation='relu'))  #hidden layer 2
#model1.add(ks.layers.Dense(5, activation='relu'))  #hidden Layer 3
model1.add(ks.layers.Dense(1, activation='sigmoid'))  #output layer

#2.Compile Block
model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#3.Fit the model
model1.fit(X, Y, epochs=100, batch_size=15, verbose=2)

#4.Evaluate model
acc = model1.evaluate(X, Y)
print("\nAccuracy =",acc[1]*100)

