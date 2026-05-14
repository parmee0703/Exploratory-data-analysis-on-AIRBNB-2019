import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\KIIT\OneDrive\Desktop\pythonvs\TASK1project2\AB_NYC_2019.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

df['reviews_per_month'].fillna(0, inplace=True)
df.drop(['id', 'name', 'host_name'], axis=1, inplace=True)
df['last_review'] = pd.to_datetime(df['last_review'])
df.drop_duplicates(inplace=True)

print(df.isnull().sum())

plt.figure()
sns.boxplot(x=df['price'])
plt.show()

df = df[df['price'] < 500]

plt.figure()
sns.boxplot(x=df['price'])
plt.show()

X = df[['price', 'number_of_reviews', 'availability_365']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(df['cluster'].value_counts())

plt.figure()
sns.scatterplot(x=df['price'], y=df['number_of_reviews'], hue=df['cluster'])
plt.show()

plt.figure()
sns.barplot(x='cluster', y='price', data=df)
plt.show()

print(df.groupby('cluster')[['price', 'number_of_reviews', 'availability_365']].mean())


numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8)) 
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()