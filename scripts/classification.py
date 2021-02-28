from scripts.functions import *
import seaborn as sn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

df = pd.read_csv('data/features.csv', index_col=0)
features = df.keys()[1:]

X = df.loc[:, df.columns != 'genre']
y = df.genre

# classifier = KNeighborsClassifier(n_neighbors=a)
# classifier = DecisionTreeClassifier(random_state=0)
# classifier = MLPClassifier(random_state=1, max_iter=800)
# classifier = svm.SVC(C=a, decision_function_shape='ovo')

s_df = shuffle(df)
scaler = StandardScaler()

#K-NN
knn_cv_train, knn_cv_test, knn_y_t, knn_y_p = k_fold_cv(s_df, 10, KNeighborsClassifier(n_neighbors=3), scaler)
knn_conf_matrix = pd.crosstab(knn_y_t, knn_y_p, rownames=['Actual'], colnames=['Predicted'], margins=True)
#sn.heatmap(knn_conf_matrix, annot=True, fmt='d').set_title("K-NN")

report = classification_report(knn_y_t, knn_y_p, output_dict=True)
print(report)

df_classification_report1 = pd.DataFrame(report1).transpose()

#Decision Tree
tree_cv_train, tree_cv_test, tree_y_t, tree_y_p = k_fold_cv(s_df, 10, DecisionTreeClassifier(random_state=0), scaler)
tree_conf_matrix = pd.crosstab(tree_y_t, tree_y_p, rownames=['Actual'], colnames=['Predicted'], margins=True)
sn.heatmap(tree_conf_matrix, annot=True, fmt='d').set_title("Decision Tree")
print(classification_report(tree_y_t, tree_y_p))

#Multylayer Perceptron
mlp_cv_train, mlp_cv_test, mlp_y_t, mlp_y_p = k_fold_cv(s_df, 10, MLPClassifier(hidden_layer_sizes=(128, 64, 32), solver='adam', max_iter=1000), scaler)
mlp_conf_matrix = pd.crosstab(mlp_y_t, mlp_y_p, rownames=['Actual'], colnames=['Predicted'], margins=True)
sn.heatmap(mlp_conf_matrix, annot=True, fmt='d').set_title("Multylayer Perceptron")
print(classification_report(mlp_y_t, mlp_y_p))

#Support Vector Machines (kernel = rbf, one-vs-one)

svm_cv_train, svm_cv_test, svm_y_t, svm_y_p = k_fold_cv(s_df, 10, svm.SVC(C=2, decision_function_shape='ovo', kernel='rbf'), scaler)
svm_conf_matrix = pd.crosstab(svm_y_t, svm_y_p, rownames=['Actual'], colnames=['Predicted'], margins=True)
sn.heatmap(svm_conf_matrix, annot=True, fmt='d').set_title("Support Vector Machines (kernel = rbf)")
print(classification_report(svm_y_t, svm_y_p))


# c = [1, 2, 5, 10, 20, 50, 100, 1000]
#
# for c in c:
# 	svm_cv_train, svm_cv_test, svm_y_t, svm_y_p = k_fold_cv(s_df, 10,
# 	                                                        svm.SVC(C=c, decision_function_shape='ovo', kernel='rbf'),
# 	                                                        scaler)
# 	print(c, accuracy_score(svm_y_t, svm_y_p))



#Accuracy = (TP + TN)/(TP + TN + FP + FN)
#Precision = TP / (TP + FP) number of correct classifications on total positives
#Recall = TP / (TP + FN) number of relevant correct classifications
#F1 score: test accuracy = 2 * (precision*recall) / (precision + recall) <- harmonic mean of precision and recall


# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(k, cv_train)
# ax.plot(k, cv_test)
# ax.set_xlabel('k')
# ax.set_ylabel('Accuracy')
# ax.legend(labels=['training error', 'test error'])