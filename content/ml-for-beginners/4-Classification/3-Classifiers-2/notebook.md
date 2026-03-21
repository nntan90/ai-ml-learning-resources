# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/4-Classification/3-Classifiers-2/notebook.ipynb

---

# Build Classification Model

```python
import pandas as pd
cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
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