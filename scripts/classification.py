from scripts.functions import *
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

df = pd.read_csv('data/features.csv', index_col=0)
features = df.keys()[1:]

X = df.loc[:, df.columns != 'genre']
y = df.genre

scaler = StandardScaler()
num_folds, repetitions = 10, 10

#K-NN

#Parameter analysis: K number of neighbors
np = np.arange(1, 10, 1)
knn_train = []
knn_test = []
for n in np:
	knn_m, knn_cm, knn_cv_train, knn_cv_test = repeated_k_fold(df, repetitions, num_folds,
															KNeighborsClassifier(n_neighbors=n), scaler)

	knn_train.append(knn_cv_train)
	knn_test.append(knn_cv_test)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np, knn_train)
ax.plot(np, knn_test)
ax.set_xlabel('n_neighbors')
ax.set_ylabel('accuracy')
ax.legend(labels=['training accuracy', 'test accuracy'])

#Classification
knn_r, knn_cm, knn_cv_train, knn_cv_test = repeated_k_fold(df, repetitions, num_folds,
														KNeighborsClassifier(n_neighbors=4), scaler)

print(np.round(knn_r, 2))
sn.heatmap(np.round(knn_cm, 1), annot=True, fmt='.1f', vmax=100).set_title("K-NN")

#DECISION TREE

#Parameter analysis: maximum tree depth
np = np.arange(1, 14, 1)
tree_train = []
tree_test = []
for n in np:
	tree_m, tree_cv_train, tree_cv_test = repeated_k_fold(df, repetitions, num_folds,
													DecisionTreeClassifier(criterion='entropy', max_depth=n), scaler)

	tree_train.append(tree_cv_train)
	tree_test.append(tree_cv_test)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np, tree_train)
ax.plot(np, tree_test)
ax.set_xlabel('max_depth')
ax.set_ylabel('accuracy')
ax.legend(labels=['training accuracy', 'test accuracy'])

#Classification
tree_r, tree_cm, tree_cv_train, tree_cv_test = repeated_k_fold(df, repetitions, num_folds,
															DecisionTreeClassifier(criterion='entropy'), scaler)

print(np.round(tree_r, 2))
sn.heatmap(np.round(tree_cm, 1), annot=True, fmt='.1f', vmax=100).set_title("DT")

#MULTILAYER PERCEPTRON

#Classification
mlp_r, mlp_cm, mlp_cv_train, mlp_cv_test = repeated_k_fold(df, repetitions, num_folds,
								MLPClassifier(hidden_layer_sizes=(128, 64, 32), solver='adam', max_iter=1000), scaler)

print(np.round(mlp_r, 2))
sn.heatmap(np.round(mlp_cm, 1), annot=True, fmt='.1f', vmax=100).set_title("MLP")

#SUPPORT VECTOR MACHINES (kernel = rbf, one-vs-one)

#Parameter analysis: C regularization parameter
np = [1, 2, 3, 4, 5, 10, 20]
svm_train = []
svm_test = []
for n in np:
	svm_r, svm_m, svm_cv_train, svm_cv_test = repeated_k_fold(df, repetitions, num_folds,
													svm.SVC(C=n, decision_function_shape='ovr', kernel='rbf'), scaler)

	svm_train.append(svm_cv_train)
	svm_test.append(svm_cv_test)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(np, svm_train)
ax.plot(np, svm_test)
ax.set_xlabel('C')
ax.set_ylabel('accuracy')
ax.legend(labels=['training accuracy', 'test accuracy'])

#Classification
svm_r, svm_cm, svm_cv_train, svm_cv_test = repeated_k_fold(df, repetitions, num_folds,
												svm.SVC(C=5, decision_function_shape='ovo', kernel='sigmoid'), scaler)

print(np.round(svm_r, 2))
sn.heatmap(np.round(svm_cm, 1), annot=True, fmt='.1f', vmax=100).set_title("SVM")

#Polynomial degree analysis for the 'polynomial' kernel
# np = [1, 2, 3, 4, 5, 10, 20]
# svm_train = []
# svm_test = []
# for n in np:
# 	svm_r, svm_m, svm_cv_train, svm_cv_test = repeated_k_fold(df, repetitions, num_folds,
# 										svm.SVC(C=5, decision_function_shape='ovo', kernel='poly', degree=n), scaler)
# 	svm_train.append(svm_cv_train)
# 	svm_test.append(svm_cv_test)
#
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(np, svm_train)
# ax.plot(np, svm_test)
# ax.set_xlabel('polynomial_degree')
# ax.set_ylabel('accuracy')
# ax.legend(labels=['training accuracy', 'test accuracy'])
