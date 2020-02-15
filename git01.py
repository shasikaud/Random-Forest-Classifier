# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 10:04:13 2020

@author: Shasika
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset_train = pd.read_csv('credit_card_default_train.csv')

#Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_1 = LabelEncoder()
dataset_train.iloc[:,2] = labelencoder_1.fit_transform(dataset_train.iloc[:,2])
labelencoder_2 = LabelEncoder()
dataset_train.iloc[:,1] = labelencoder_2.fit_transform(dataset_train.iloc[:,1])
labelencoder_3 = LabelEncoder()
dataset_train.iloc[:,3] = labelencoder_3.fit_transform(dataset_train.iloc[:,3])
onehotencoder_3 = OneHotEncoder(categorical_features = [3])
labelencoder_4 = LabelEncoder()
dataset_train.iloc[:,4] = labelencoder_4.fit_transform(dataset_train.iloc[:,4])
labelencoder_5 = LabelEncoder()
dataset_train.iloc[:,5] = labelencoder_5.fit_transform(dataset_train.iloc[:,5])

#onehotencoder_1 = OneHotEncoder(categorical_features = [2])
#onehotencoder_1.fit(dataset_train)
#x = onehotencoder_1.transform(dataset_train.iloc[:,2]).toarray()
#dataset_train2 = onehotencoder_1.fit_transform(dataset_train).toarray()


#seperating columns
x = dataset_train.iloc[:, 1:24].values
y = dataset_train.iloc[:, 24].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier.fit(x,y)

#predicting
y_pred = classifier.predict(x)

#comparison of results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)