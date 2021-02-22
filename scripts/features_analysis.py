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

s_df = shuffle(df)
scaler = StandardScaler()

features_results = pd.DataFrame([], index=features, columns=['train', 'test'])
for f in features:
	df_1f = s_df[[f, 'genre']]
	cv_train, cv_test, y_t, y_p  = k_fold_cv(df_1f, 10, DecisionTreeClassifier(random_state=0), scaler)
	features_results.train[f] = np.mean(cv_train)
	features_results.test[f] = np.mean(cv_test)

features_results = features_results.sort_values('test', ascending=False)
