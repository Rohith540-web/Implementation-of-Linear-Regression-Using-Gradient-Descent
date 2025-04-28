# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Load data from CSV into a DataFrame.

2.Extract features (x) and target values (y) from the DataFrame.

3.Convert all values to float for computation compatibility.

4.Initialize two StandardScalers: one for x, one for y.

5.Standardize (normalize) both x and y to have mean 0 and std 1.

6.Define linear_regression function that:

   *Adds bias term (intercept) to x

   *Initializes theta (weight vector)

7.Runs num_iters iterations of gradient descent to minimize MSE loss

8.Train the model by calling linear_regression(x_scaled, y_scaled)

9.Prepare a new input sample, scale it using the same x_scaler.

10.Make prediction using theta, then inverse transform the result using y_scaler.

11.Print predicted output in original scale.

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: ROHITH V
RegisterNumber: 212224220083


```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(x).dot(theta).reshape(-1,1)
        errors =(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
x=(data.iloc[1:,:-2].values)
print(x)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
print(x1_scaled)
print(y1_scaled)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"predicted value: {pre}")

```

## Output:
DATA INFORMATION:


![Screenshot 2025-03-10 161300](https://github.com/user-attachments/assets/b32d40a7-39ff-4642-b629-8829a9c9ec5e)

THE VALUE OF X:


![Screenshot 2025-03-10 161310](https://github.com/user-attachments/assets/b066b2b6-2700-46e7-8b7d-9d7731bf6640)

THE VALUE OF Y:


![Screenshot 2025-03-10 161324](https://github.com/user-attachments/assets/c5d8fea7-d51a-4017-b9c8-d6c9521a748e)

THE VALUE OF X_SCALED:


![Screenshot 2025-03-10 161333](https://github.com/user-attachments/assets/8a7a3550-6056-4707-b3af-47d8a75c6e82)

THE VALUE OF Y_SCALED:


![Screenshot 2025-03-10 161343](https://github.com/user-attachments/assets/220eefd4-f19a-460c-97a9-70aaafa0a525)

PREDICTED VALUE:


![Screenshot 2025-03-10 161357](https://github.com/user-attachments/assets/701e616a-761f-4fdf-a0d5-d87c67144798)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
