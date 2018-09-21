# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:46:04 2018

@author: Mohak

House Price Prediction program
"""

import pandas as pd
import numpy as np
import keras as ks

ds = pd.read_csv("E:\ML projects\Unreliable House Data.csv") #read the csv file
ds = np.array(ds)   #convert the csv to numpy array
X = ds[:,1:4]
print(X.shape)
Y = ds[:,1]
#print(Y)

np.random.seed(11)

#1.Defining the model
model1 = ks.models.Sequential()
#Adding layers to the model
model1.add(ks.layers.Dense(40, input_dim=3, activation="relu")) #input layer
model1.add(ks.layers.Dense(25, activation='relu'))  #hidden layer 1
model1.add(ks.layers.Dense(15, activation='relu'))  #hidden layer 2
#model1.add(ks.layers.Dense(5, activation='relu'))  #hidden Layer 3
model1.add(ks.layers.Dense(1, activation='sgd'))  #output layer

#2.Compile Block
model1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#3.Fit the model
model1.fit(X, Y, epochs=100, batch_size=15, verbose=2)

#4.Evaluate model
acc = model1.evaluate(X, Y)
print("\nAccuracy =",acc[1]*100)

#save the model
model1.save("E:\ML projects\House model.h5")

