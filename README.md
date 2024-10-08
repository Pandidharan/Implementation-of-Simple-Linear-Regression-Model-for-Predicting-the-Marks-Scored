# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Pandidharan.G.R   
RegisterNumber:  212222040111
*/
```
```PYTHON
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error

df = pd.read_csv('student_scores.csv')
X = df.iloc[:,:-1].values
X
Y = df.iloc[:,1].values
Y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
Y_pred = regressor.predict(X_test)

Y_pred
y_test

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Hours vs Scores(Training Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```

## Output:

![Screenshot 2024-08-30 103523](https://github.com/user-attachments/assets/05e2d879-3f57-4509-9e3e-0ca4231a65db)
![Screenshot 2024-08-30 103515](https://github.com/user-attachments/assets/d32436bd-1460-4260-8bd6-f4f38705defa)
![Screenshot 2024-08-30 103503](https://github.com/user-attachments/assets/6be4ee7c-caf5-48ac-bab0-46bc42bb4f3c)
![Screenshot 2024-08-30 103549](https://github.com/user-attachments/assets/65dd45b6-d662-44ab-9cc5-0263e0309f21)
![Screenshot 2024-08-30 140517](https://github.com/user-attachments/assets/3fb956f4-7ffc-4dae-8663-ab4d3d8204d5)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
