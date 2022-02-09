# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 01:19:36 2022

@author: R PabitraKumar Reddy
"""

#Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('C:/Users/om/Downloads/NSE Out of Time Testing Data - 1st Jan 2022 to 4th Feb 2022.csv')
dataset.head(5)
dataset.tail(5)

#CHECKING DATA TYPE OF A VARIABLE
dataset.info()

#Now drop the Date & Adj Close Variable , as tyey are not necessery for our prediction . 
dataset.drop(["Date","Adj Close"],axis=1,inplace=True)

#Identifying/Finding missing values if any----
dataset.isnull().sum()

#Obtain dependent & independent variale 
x= dataset.drop("Close",axis=1)
y= dataset["Close"]

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

X_train.shape

X_test.shape

# Fitting Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Accuracy of the model

#Calculating the r squared value:
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#Accuracy = 92.41%

#Calculating the coefficients:
print(regressor.coef_)

#Calculating the intercept:
print(regressor.intercept_)

#Create a DataFrame
df1 = {'Actual Price':y_test,
'Predicted Price':y_pred}
df1 = pd.DataFrame(df1,columns=['Actual Price','Predicted Price'])
print(df1)

# Visualising the predicted results
graph=df1.head(20)
graph.plot(kind='bar')
plt.bar('Actual  Price','Predicted  Price')
plt.xlabel('Actual  Price', fontsize=16)
plt.ylabel('Predicted Price', fontsize=16)
plt.title('Barchart - stock price',fontsize=20)


###### ACCURACY = 92.41%











