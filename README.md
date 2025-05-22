# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SONU S
RegisterNumber:  212223220107
*/
```

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
```


```python
data = pd.read_csv('/content/Salary.csv')
print()
# print(data.head())
print(data.head())         
print()
print(data.info())         
print()
print(data.isnull().sum())  

```


```python
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
print(data.head())
```


```python
x = data[["Position", "Level"]]                            # Features
y = data["Salary"]                                         # Target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
```


```python
x = data[["Position", "Level"]]  # Features
y = data["Salary"]               # Target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
```

## Output:

### Dataset 
![image](https://github.com/user-attachments/assets/54932575-43e5-48d3-9431-ca10d4aa8fea)

### Updated Dataset After LableEncoder

![image](https://github.com/user-attachments/assets/4b879a18-4edb-4881-a6f5-12e40cf85797)

### MSE and Predicted 

![image](https://github.com/user-attachments/assets/a65309fa-e82b-437b-8403-0c390478a8d4)

### Tree

![download](https://github.com/user-attachments/assets/9d262407-0119-4e6e-b29d-9a7d99ac1e92)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
