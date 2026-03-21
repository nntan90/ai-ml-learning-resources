# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/4-Classification/2-Classifiers-1/solution/notebook.ipynb

---

# Build Classification Models

```python
import pandas as pd
cuisines_df = pd.read_csv("../../data/cleaned_cuisines.csv")
cuisines_df.head()
```

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
from sklearn.svm import SVC
import numpy as np
```

```python
cuisines_label_df = cuisines_df['cuisine']
cuisines_label_df.head()
```

```python
cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
cuisines_feature_df.head()
```

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

```python
lr = LogisticRegression(multi_class='ovr',solver='liblinear')
model = lr.fit(X_train, np.ravel(y_train))

accuracy = model.score(X_test, y_test)
print ("Accuracy is {}".format(accuracy))
```

```python
# test an item
print(f'ingredients: {X_test.iloc[50][X_test.iloc[50]!=0].keys()}')
print(f'cuisine: {y_test.iloc[50]}')
```

```python
#rehsape to 2d array and transpose
test= X_test.iloc[50].values.reshape(-1, 1).T
# predict with score
proba = model.predict_proba(test)
classes = model.classes_
# create df with classes and scores
resultdf = pd.DataFrame(data=proba, columns=classes)

# create df to show results
topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
topPrediction.head()
```

```python
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
```