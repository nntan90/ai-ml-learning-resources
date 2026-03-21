# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/4-Classification/4-Applied/solution/notebook.ipynb

---

# Build a cuisine recommender

```python
!pip install skl2onnx
```

```python
import pandas as pd 

```

```python
data = pd.read_csv('../../data/cleaned_cuisines.csv')
data.head()
```

```python
X = data.iloc[:,2:]
X.head()
```

```python
y = data[['cuisine']]
y.head()
```

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
```

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
```

```python
model = SVC(kernel='linear', C=10, probability=True,random_state=0)
model.fit(X_train,y_train.values.ravel())

```

```python
y_pred = model.predict(X_test)
```

```python
print(classification_report(y_test,y_pred))
```

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 380]))]
options = {id(model): {'nocl': True, 'zipmap': False}}
onx = convert_sklearn(model, initial_types=initial_type, options=options)
with open("./model.onnx", "wb") as f:
    f.write(onx.SerializeToString())



```