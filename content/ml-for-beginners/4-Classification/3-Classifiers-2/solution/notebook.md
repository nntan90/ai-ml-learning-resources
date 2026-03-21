# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/4-Classification/3-Classifiers-2/solution/notebook.ipynb

---

# Build More Classification Models

```python
import pandas as pd
cuisines_df = pd.read_csv("../../data/cleaned_cuisines.csv")
cuisines_df.head()
```

```python
cuisines_label_df = cuisines_df['cuisine']
cuisines_label_df.head()
```

```python
cuisines_features_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
cuisines_features_df.head()
```

# Try different classifiers

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np
```

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_features_df, cuisines_label_df, test_size=0.3)
```

```python

C = 10
# Create different classifiers.
classifiers = {
    'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0),
    'KNN classifier': KNeighborsClassifier(C),
    'SVC': SVC(),
    'RFST': RandomForestClassifier(n_estimators=100),
    'ADA': AdaBoostClassifier(n_estimators=100)
    
}

```

```python
n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test,y_pred))
```