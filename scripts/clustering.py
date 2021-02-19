import numpy as np
import pandas as pd
from scripts.functions import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA

df = pd.read_csv('data/features.csv', index_col=0)
features = df.keys()[1:]

X = df.loc[:, df.columns != 'genre']
y = df.genre
scaler = MinMaxScaler()
#scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=features)

pca = PCA(n_components=2)
X = pca.fit_transform(X)

num_clusters = 10
kmeans = cluster.KMeans(n_clusters=num_clusters)

kmeans.fit(X)

centroids = kmeans.cluster_centers_
clusters = kmeans.labels_

# wcss = []
# for i in range(1, 11):
#     kmeans = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

print(centroids)
print(kmeans.score(X))

#silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')

#print(silhouette_score)
X = pd.DataFrame(X, columns=['x', 'y'])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
x = X.x
y = X.y
ax.scatter(x, y=y, c=clusters, s=30)

for i, txt in enumerate(df.genre):
    ax.annotate(txt, (x[i], y[i]))

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