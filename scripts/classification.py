from scripts.functions import *
import matplotlib.pyplot as plt

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
sn.heatmap(knn_conf_matrix, annot=True, fmt='d').set_title("K-NN")
print(classification_report(knn_y_t, knn_y_p))

#Decision Tree
tree_cv_train, tree_cv_test, tree_y_t, tree_y_p = k_fold_cv(s_df, 10, DecisionTreeClassifier(random_state=0), scaler)
tree_conf_matrix = pd.crosstab(tree_y_t, tree_y_p, rownames=['Actual'], colnames=['Predicted'], margins=True)
sn.heatmap(tree_conf_matrix, annot=True, fmt='d').set_title("Decision Tree")
print(classification_report(tree_y_t, tree_y_p))

#Multylayer Perceptron
mlp_cv_train, mlp_cv_test, mlp_y_t, mlp_y_p = k_fold_cv(s_df, 10, MLPClassifier(random_state=1, max_iter=800), scaler)
mlp_conf_matrix = pd.crosstab(mlp_y_t, mlp_y_p, rownames=['Actual'], colnames=['Predicted'], margins=True)
sn.heatmap(mlp_conf_matrix, annot=True, fmt='d').set_title("Multylayer Perceptron")
print(classification_report(mlp_y_t, mlp_y_p))

#Support Vector Machines (kernel = rbf, one-vs-one)
svm_cv_train, svm_cv_test, svm_y_t, svm_y_p = k_fold_cv(s_df, 10, svm.SVC(C=1, decision_function_shape='ovo'), scaler)
svm_conf_matrix = pd.crosstab(svm_y_t, svm_y_p, rownames=['Actual'], colnames=['Predicted'], margins=True)
sn.heatmap(svm_conf_matrix, annot=True, fmt='d').set_title("Support Vector Machines (kernel = rbf)")
print(classification_report(svm_y_t, svm_y_p))


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