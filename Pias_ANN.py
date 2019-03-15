# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:50:18 2019

@author: Pias Tanmoy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[: , 3:13].values
Y = dataset.iloc[ : , 13:].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelEncoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2) 


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

classifier = Sequential()
classifier.add(Dense(units = 6, activation='relu', kernel_initializer='glorot_uniform', input_dim=11))
classifier.add(Dense(units = 10, activation='relu', kernel_initializer = 'glorot_uniform'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.save('ANN_model.hdf5')

classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


































