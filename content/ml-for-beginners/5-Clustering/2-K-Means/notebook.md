# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/5-Clustering/2-K-Means/notebook.ipynb

---

# Nigerian Music scraped from Spotify - an analysis

```python
pip install seaborn
```

Start where we finished in the last lesson, with data imported and filtered.

```python

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv("../data/nigerian-songs.csv")
df.head()
```

We will focus only on 3 genres. Maybe we can get 3 clusters built!

```python
df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
df = df[(df['popularity'] > 0)]
top = df['artist_top_genre'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top.index,y=top.values)
plt.xticks(rotation=45)
plt.title('Top genres',color = 'blue')
```

```python
df.head()
```