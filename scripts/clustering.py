import numpy as np
import pandas as pd
from scripts.functions import *
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA

df = pd.read_csv('data/features.csv', index_col=0)
features = df.keys()[1:]

#Outlier removal
X = df.loc[:, df.columns != 'genre']
y = df.genre
outliers = detectOutliers(X)
X = detectOutliers(X, rm=True)

X = df.loc[:, df.columns != 'genre']
y = df.genre
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=features)

outliers = detectOutliers(X)
X = detectOutliers(X, rm=True)
y = pd.DataFrame(y._drop_axis(outliers.index, axis=0))

num_clusters = 10
kmeans = cluster.KMeans(n_clusters=num_clusters)

pca = PCA(n_components=2)
px = pca.fit_transform(X)

kmeans.fit(px)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

clusters = np.array(labels)

# wcss = []
# for i in range(1, 11):
#     kmeans = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(px)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()

print(labels)

print(centroids)

print(kmeans.score(px))

silhouette_score = metrics.silhouette_score(px, labels, metric='euclidean')

print(silhouette_score)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x=px[:, 0], y=px[:, 1], c=clusters, s=30)