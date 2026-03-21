# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/2-Regression/1-Tools/solution/notebook.ipynb

---

## Linear Regression for Diabetes dataset - Lesson 1

Import needed libraries

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection

```

Load the diabetes dataset, divided into `X` data and `y` features

```python
X, y = datasets.load_diabetes(return_X_y=True)
print(X.shape)
print(X[0])
```

Select just one feature to target for this exercise

```python
# Selecting the 3rd feature
X = X[:, 2]
print(X.shape)

```

```python
#Reshaping to get a 2D array
X = X.reshape(-1, 1)
print(X.shape)
print(X)
```

Split the training and test data for both `X` and `y`

```python
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)

```

Select the model and fit it with the training data

```python
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
```

Use test data to predict a line

```python
y_pred = model.predict(X_test)

```

Display the results in a plot

```python
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.show()
```