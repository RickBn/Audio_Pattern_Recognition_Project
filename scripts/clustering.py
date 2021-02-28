import numpy as np
import pandas as pd
from scripts.functions import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sn
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA

df = pd.read_csv('data/features.csv', index_col=0)
features = df.keys()[1:]

X = df.loc[:, df.columns != 'genre']
y = df.genre
genres_encoded = LabelEncoder().fit_transform(y)
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=features)

#PCA
pca = PCA(n_components=2)
X = pca.fit_transform(X)

#Plot by genre
X = pd.DataFrame(X, columns=['x', 'y'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
scatter = ax.scatter(X.x, X.y, c=genres_encoded, cmap=cm.tab10, label=y.unique(), s=30)
ax.legend()
legend = ax.legend(*scatter.legend_elements(), title="Genres")
for i, g in enumerate(y.unique()):
	legend.get_texts()[i].set_text(g)

#Elbow method
wcss = []
for i in range(1, 11):
    kmeans = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#K = 3
num_clusters = 3
kmeans = cluster.KMeans(n_clusters=num_clusters)

kmeans.fit(X)

centroids = kmeans.cluster_centers_
clusters = kmeans.labels_

print(centroids)
print(kmeans.score(X))
silhouette_score = metrics.silhouette_score(X, clusters, metric='euclidean')
print(silhouette_score)

X = pd.DataFrame(X, columns=['x', 'y'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
scatter = ax.scatter(X.x, X.y, c=clusters, cmap=cm.tab10, label=y.unique(), s=30)

c_df = pd.DataFrame(clusters, columns=['cluster'])
c_df['genre'] = df.genre

c_groups = dict(tuple(c_df.groupby('cluster')))

summary = pd.DataFrame([], index=np.unique(clusters), columns=df.genre.unique())

for c in c_groups:
	g = c_groups[c]
	v = g.genre.value_counts()
	for i, genre in enumerate(v.index):
		summary[genre][c] = v[i]

summary = summary.fillna(0)
sn.heatmap(summary, annot=True, fmt='d')

#K = 10
num_clusters = 10
kmeans = cluster.KMeans(n_clusters=num_clusters)

kmeans.fit(X)

centroids = kmeans.cluster_centers_
clusters = kmeans.labels_

print(centroids)
print(kmeans.score(X))
silhouette_score = metrics.silhouette_score(X, clusters, metric='euclidean')
print(silhouette_score)

X = pd.DataFrame(X, columns=['x', 'y'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
scatter = ax.scatter(X.x, X.y, c=clusters, cmap=cm.tab10, label=y.unique(), s=30)

c_df = pd.DataFrame(clusters, columns=['cluster'])
c_df['genre'] = df.genre

c_groups = dict(tuple(c_df.groupby('cluster')))

summary = pd.DataFrame([], index=np.unique(clusters), columns=df.genre.unique())

for c in c_groups:
	g = c_groups[c]
	v = g.genre.value_counts()
	for i, genre in enumerate(v.index):
		summary[genre][c] = v[i]

summary = summary.fillna(0)
sn.heatmap(summary, annot=True, fmt='d')


