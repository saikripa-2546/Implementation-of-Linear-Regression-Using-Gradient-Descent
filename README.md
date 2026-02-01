# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program and import required libraries such as NumPy, Pandas and StandardScaler.

2.Load the dataset using Pandas and separate the independent variables (X) and dependent variable (y).

3.Perform feature scaling on the data using StandardScaler to normalize the values.

4.Initialize model parameters and apply Gradient Descent to update theta values iteratively.

5.Predict the output for new input data and display the predicted result.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SAIKRIPA SK
RegisterNumber:  212224040284
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data = pd.read_csv(r"C:\Users\admin\ML\DATASET-20260129\50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
<img width="752" height="147" alt="image" src="https://github.com/user-attachments/assets/719ae790-f687-4419-a805-1e3c4930faec" />
<img width="361" height="697" alt="image" src="https://github.com/user-attachments/assets/fa65b215-3de6-4c82-a676-960165c52b6a" />
<img width="337" height="607" alt="image" src="https://github.com/user-attachments/assets/f81cb9da-0369-4f7c-b0a8-4742b29f2dff" />
<img width="547" height="757" alt="image" src="https://github.com/user-attachments/assets/9b2d6209-b491-40a3-b925-16cb1b564346" />
<img width="622" height="757" alt="image" src="https://github.com/user-attachments/assets/538db39d-db33-4b88-9454-dc0bdc5133af" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
