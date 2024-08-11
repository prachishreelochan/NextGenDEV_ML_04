import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
df = pd.read_csv('/content/Mall_Customers.csv')
df
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
df
df.shape
plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'])
plt.show
# To identify clusters we will use Elbow Method
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i)
  kmeans.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])
  wcss.append(kmeans.inertia_)
wcss
plt.plot(range(1,11), wcss)
plt.show()
x = df.iloc[:, [3,4]].values
kmeans = KMeans(n_clusters=5)
y = kmeans.fit_predict(x)
y
plt.scatter(x[y==0,0], x[y==0,1], s=100, c='red')

plt.scatter(x[y==1,0], x[y==1,1], s=100, c='blue')

plt.scatter(x[y==2,0], x[y==2,1], s=100, c='green')

plt.scatter(x[y==3,0], x[y==3,1], s=100, c='cyan')

plt.scatter(x[y==4,0], x[y==4,1], s=100, c='magenta')

plt.show()
