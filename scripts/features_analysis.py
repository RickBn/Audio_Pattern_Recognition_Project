from scripts.functions import *
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

df = pd.read_csv('data/features.csv', index_col=0)
features = df.keys()[1:]

X = df.loc[:, df.columns != 'genre']
y = df.genre

s_df = shuffle(df)
scaler = StandardScaler()

num_folds, repetitions = 10, 10

knn_results = pd.DataFrame([], index=features, columns=['test'])
tree_results = pd.DataFrame([], index=features, columns=['test'])
mlp_results = pd.DataFrame([], index=features, columns=['test'])
svm_results = pd.DataFrame([], index=features, columns=['test'])

for f in features:
	df_1f = s_df[[f, 'genre']]
	knn_cv_train, knn_cv_test, y_t, y_p = k_fold_cv(df_1f, num_folds, KNeighborsClassifier(n_neighbors=3), scaler)
	tree_cv_train, tree_cv_test, y_t, y_p = k_fold_cv(df_1f, num_folds, DecisionTreeClassifier(criterion='entropy'), scaler)
	# mlp_cv_train, mlp_cv_test, y_t, y_p = k_fold_cv(df_1f, num_folds,
	#                                                    MLPClassifier(solver='adam',
	#                                                                  max_iter=1000), scaler)
	svm_cv_train, svm_cv_test, y_t, y_p = k_fold_cv(df_1f, num_folds,
	                                      svm.SVC(C=5, decision_function_shape='ovo', kernel='rbf'), scaler)

	knn_results.test[f] = np.mean(knn_cv_test)
	tree_results.test[f] = np.mean(tree_cv_test)
	# mlp_results.test[f] = np.mean(mlp_cv_test)
	svm_results.test[f] = np.mean(svm_cv_test)

knn_results = knn_results.sort_values('test', ascending=False)
tree_results = tree_results.sort_values('test', ascending=False)
mlp_results = mlp_results.sort_values('test', ascending=False)
svm_results = svm_results.sort_values('test', ascending=False)


print("K-NN: ", knn_results)
print("Decision tree: ", tree_results)
print("MLP: ", mlp_results)
print("DSVM: ", svm_results)


#
# indices = features_results[:30].index
# indices = list(indices.values)
# indices.append('genre')
#
# tree_cv_train, tree_cv_test, tree_y_t, tree_y_p = k_fold_cv(s_df[indices], 10,
#                                                             DecisionTreeClassifier(criterion='entropy'), scaler)
