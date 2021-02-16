from scripts.functions import *

#df = pd.read_csv('data/features.csv', index_col=0)
df = pd.read_csv('data/new.csv', index_col=0)
features = df.keys()[1:]

X = df.loc[:, df.columns != 'genre']
y = df.genre

k = [1]

features_results = pd.DataFrame([], index=features, columns=['train', 'test'])
for f in features:
	df_1f = df[[f, 'genre']]
	cv_train, cv_test, y_t, y_p = k_fold_cv(df_1f, 10, k)
	features_results.train[f] = np.mean(cv_train)
	features_results.test[f] = np.mean(cv_test)

features_results = features_results.sort_values('test', ascending=False)
