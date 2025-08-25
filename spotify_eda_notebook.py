# Spotify EDA Notebook

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Step 2: Load and Inspect Data
df = pd.read_csv("tracks.csv")
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
df.head()

# Step 3: Data Cleaning
df = df.drop(columns=["track_id", "album_name", "release_date"], errors='ignore')
print(df.isnull().sum())
df = df.dropna()

# Step 4: Correlation Heatmap (only numeric data)



numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
plt.figure(figsize=(12, 6))
sns.heatmap(corr, annot=True, cmap='inferno')
plt.title("Correlation Heatmap Between Audio Features")
plt.show()




# Step 5: Feature Relationship Plots




plt.figure(figsize=(8, 5))
sns.regplot(x='energy', y='loudness',
             data=df, color='cyan')
plt.title("Loudness vs Energy")
plt.show()


plt.figure(figsize=(8, 5))
sns.regplot(x='acousticness', y='popularity', 
            data=df, color='blue')
plt.title("Popularity vs Acousticness")
plt.show()

# Step 6: Trends Over Time (if year column exists)
if 'year' in df.columns:
    year_data = df.groupby("year")["duration_ms"].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="year", y="duration_ms", data=year_data)
    plt.title("Average Song Duration by Year")
    plt.ylabel("Duration (ms)")
    plt.show()

# Step 7: Top Genres by Popularity (if genre column exists)



if 'genre' in df.columns:
    top_genres = df.groupby("genre")
    ["popularity"].mean().sort_values(ascending=
                                      False).head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_genres.values, y=top_genres.index,
                 palette='viridis')
    plt.title("Top 10 Genres by Average Popularity")
    plt.xlabel("Popularity")
    plt.show()


# Step 8: Feature Distribution
features = ['danceability', 'energy', 'valence', 'tempo']
df[features].hist(bins=30, figsize=(12, 6), color='skyblue')
plt.suptitle("Distribution of Audio Features")
plt.show()

# Step 9: Outlier Detection
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['tempo', 'energy', 'duration_ms']])
plt.title("Boxplot for Tempo, Energy, and Duration")
plt.show()

# Step 10: Save Cleaned Data
df.to_csv("cleaned_spotify_data.csv", index=False)

# Optional: Interactive Plotly Chart
fig = px.scatter(df, x='energy', y='popularity', color='genre' if 'genre' in df.columns else None, hover_data=['name'] if 'name' in df.columns else None)
fig.show()
