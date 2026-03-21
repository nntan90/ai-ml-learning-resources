# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/4-Classification/1-Introduction/solution/notebook.ipynb

---

# Delicious Asian and Indian Cuisines 


Install Imblearn which will enable SMOTE. This is a Scikit-learn package that helps handle imbalanced data when performing classification. (https://imbalanced-learn.org/stable/)

```python
pip install imblearn
```

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE
```

```python
df  = pd.read_csv('../../data/cuisines.csv')
```

This dataset includes 385 columns indicating all kinds of ingredients in various cuisines from a given set of cuisines.

```python
df.head()
```

```python
df.info()
```

```python
df.cuisine.value_counts()
```

Show the cuisines in a bar graph

```python
df.cuisine.value_counts().plot.barh()
```

```python

thai_df = df[(df.cuisine == "thai")]
japanese_df = df[(df.cuisine == "japanese")]
chinese_df = df[(df.cuisine == "chinese")]
indian_df = df[(df.cuisine == "indian")]
korean_df = df[(df.cuisine == "korean")]

print(f'thai df: {thai_df.shape}')
print(f'japanese df: {japanese_df.shape}')
print(f'chinese df: {chinese_df.shape}')
print(f'indian df: {indian_df.shape}')
print(f'korean df: {korean_df.shape}')
```

## What are the top ingredients by class

```python
def create_ingredient_df(df):
    # transpose df, drop cuisine and unnamed rows, sum the row to get total for ingredient and add value header to new df
    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
    # drop ingredients that have a 0 sum
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    # sort df
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False, inplace=False)
    return ingredient_df

```

```python
thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()
```

```python
japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()
```

```python
chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()
```

```python
indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()
```

```python
korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()
```

Drop very common ingredients (common to all cuisines)

```python
feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
labels_df = df.cuisine #.unique()
feature_df.head()

```

Balance data with SMOTE oversampling to the highest class. Read more here: https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html

```python
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
```

```python
print(f'new label count: {transformed_label_df.value_counts()}')
print(f'old label count: {df.cuisine.value_counts()}')
```

```python
transformed_feature_df.head()
```

```python
# export transformed data to new df for classification
transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
transformed_df
```

```python
transformed_df.info()
```

Save the file for future use

```python
transformed_df.to_csv("../../data/cleaned_cuisines.csv")
```