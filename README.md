# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Import Libraries**: Import required libraries: `numpy`, `matplotlib`, `pandas`, and necessary modules from `sklearn`.

2. **Load Dataset**: Load the `student_scores.csv` dataset into a DataFrame using `pandas`.

3. **Extract Features and Labels**: Split the dataset into input features (`X`, representing hours studied) and the output labels (`Y`, representing scores obtained).

4. **Split Data into Training and Testing Sets**: Use `train_test_split()` to split the data into training and testing sets. The test size is set to 1/3 of the dataset, and a random state ensures reproducibility.

5. **Initialize Linear Regression Model**: Create an instance of the `LinearRegression` class.

6. **Train the Model**: Fit the linear regression model using the training data (`X_train` and `y_train`).

7. **Make Predictions**: Predict the scores (`Y_pred`) using the test data (`X_test`).

8. **Compare Predictions with Actuals**: View the predicted scores (`Y_pred`) and compare them with the actual test scores (`y_test`).

9. **Visualize Training Set Results**: Plot a scatter plot of the training set data points (hours vs scores) and the regression line based on the predicted values.

10. **Label the Plot**: Add title, x-axis (hours), and y-axis (scores) labels to the plot. Then, display the plot.

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
print("X:",X)
Y = df.iloc[:,1].values
print("Y:",Y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
Y_pred = regressor.predict(X_test)

print("Y-pred:",Y_pred)
print("y-test:",y_test)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Hours vs Scores(Training Set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

```

## Output:

![Screenshot 2024-11-24 194054](https://github.com/user-attachments/assets/88e48298-14e6-4527-8a8f-ab32b9d8aed8)

![image](https://github.com/user-attachments/assets/5b03b091-4ba6-49c6-a30d-a223906e3347)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
