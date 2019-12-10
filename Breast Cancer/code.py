#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:32:42 2019

@author: ayaz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('data.csv')
dataset['diagnosis'] = dataset['diagnosis'].map({'M': 1, 'B': 0})
X = dataset.iloc[:,2:32].values
Y = dataset.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0) 

plt.scatter(X,Y)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialize
classifier = Sequential()

#input/first layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu', input_dim = 30))

#2nd layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))

#third layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))

#4th layer
classifier.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))

#output
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting
classifier.fit(x_train, y_train, batch_size = 32, epochs = 100)
#Prediction
y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)
