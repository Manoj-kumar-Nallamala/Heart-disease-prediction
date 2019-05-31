#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
dataset=pd.read_csv('heart.csv')
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,13]


#splitting data to train and test sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


#multiple linear regressor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)


#smothening the predicted result into two classes
for i in range(0,len(y_pred)):
    if y_pred[i]>0.5:
        y_pred[i]=1
    else :
        y_pred[i]=0

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

##accuracy obtained 82.17
