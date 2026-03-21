# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/2-Regression/4-Logistic/notebook.ipynb

---

## Pumpkin Varieties and Color

Load up required libraries and dataset. Convert the data to a dataframe containing a subset of the data: 

Let's look at the relationship between color and variety

```python
import pandas as pd
import numpy as np

full_pumpkins = pd.read_csv('../data/US-pumpkins.csv')

full_pumpkins.head()

```