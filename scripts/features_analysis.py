import numpy as np
import pandas as pd
from scripts.functions import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle


df = pd.read_csv('data/features.csv', index_col=0)
features = df.keys()[1:]

X = df.loc[:, df.columns != 'genre']
y = df.genre


k = np.arange(1, 10)


cv_train, cv_test = k_fold_cv(df, 10, k)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k, cv_train)
ax.plot(k, cv_test)
ax.set_xlabel('k')
ax.set_ylabel('Accuracy')
ax.legend(labels=['training error', 'test error'])

features_results = pd.DataFrame([], index=features, columns=['train', 'test'])
for f in features:
	df_1f = df[[f, 'genre']]
	cv_train, cv_test = k_fold_cv(df_1f, 10, k)
	features_results.train[f] = np.mean(cv_train)
	features_results.test[f] = np.mean(cv_test)

features_results = features_results.sort_values('test', ascending=False)

best_features = features_results.index[0:30].values
df_r = df[np.append(best_features, 'genre')]

cv_train, cv_test = k_fold_cv(df_r, 10, k)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k, cv_train)
ax.plot(k, cv_test)
ax.set_xlabel('k')
ax.set_ylabel('Accuracy')
ax.legend(labels=['training error', 'test error'])