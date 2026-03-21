# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/5-Clustering/1-Visualize/solution/notebook.ipynb

---

# Nigerian Music scraped from Spotify - an analysis

```python
!pip install seaborn
```

```python
import matplotlib.pyplot as plt
import pandas as pd
```

```python
df = pd.read_csv("../../data/nigerian-songs.csv")
df.head()
```

Get information about the dataframe

```python
df.info()
```

Double-check for null values.

```python
df.isnull().sum()
```

Look at the general values of the data. Note that popularity can be '0' - and there are many rows with that value

```python
df.describe()
```

Let's examine the genres. Quite a few are listed as 'Missing' which means they aren't categorized in the dataset with a genre 

```python
import seaborn as sns

top = df['artist_top_genre'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top[:5].index,y=top[:5].values)
plt.xticks(rotation=45)
plt.title('Top genres',color = 'blue')
```

Remove 'Missing' genres, as it's not classified in Spotify


```python
df = df[df['artist_top_genre'] != 'Missing']
top = df['artist_top_genre'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top.index,y=top.values)
plt.xticks(rotation=45)
plt.title('Top genres',color = 'blue')
```

The top three genres comprise the greatest part of the dataset, so let's focus on those

```python
df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
df = df[(df['popularity'] > 0)]
top = df['artist_top_genre'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top.index,y=top.values)
plt.xticks(rotation=45)
plt.title('Top genres',color = 'blue')
```

The data is not strongly correlated except between energy and loudness, which makes sense. Popularity has a correspondence to release data, which also makes sense, as more recent songs are probably more popular. Length and energy seem to have a correlation - perhaps shorter songs are more energetic?

```python
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
```

Are the genres significantly different in the perception of their danceability, based on their popularity? Examine our top three genres data distribution for popularity and danceability along a given x and y axis 

```python
sns.set_theme(style="ticks")

# Show the joint distribution using kernel density estimation
g = sns.jointplot(
    data=df,
    x="popularity", y="danceability", hue="artist_top_genre",
    kind="kde",
)
```

In general, the three genres align in terms of their popularity and danceability.  A scatterplot of the same axes shows a similar pattern of convergence. Try a scatterplot to check the distribution of data per genre

```python
sns.FacetGrid(df, hue="artist_top_genre", size=5) \
   .map(plt.scatter, "popularity", "danceability") \
   .add_legend()
```