import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def detectOutliers(df, rm=False):
    for f in df.columns:
        q1 = np.percentile(df[f], 25, interpolation='midpoint')
        q3 = np.percentile(df[f], 75, interpolation='midpoint')
        iqr = q3 - q1

        if not rm:
            return df.loc[(df[f] < (q1 - (1.5 * iqr))) | (df[f] > (q3 + (1.5 * iqr)))]

        else:
            return df.loc[(df[f] > (q1 - (1.5 * iqr))) & (df[f] < (q3 + (1.5 * iqr)))]

def removeOutliers(df):

    X = df.loc[:, df.columns != 'genre']
    y = df.genre
    outliers = detectOutliers(X)
    X = detectOutliers(X, rm=True)
    y = pd.DataFrame(y._drop_axis(outliers.index, axis=0))
    X['genre'] = y

    return X

def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while round(last) < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

#Accuracy = (TP + TN)/(TP + TN + FP + FN)
#Precision = TP / (TP + FP)
#Recall = TP / (TP + FN)

def k_fold_cv(data, num_splits, k):

    cv_train = np.zeros(len(k))
    cv_test = np.zeros(len(k))

    data = shuffle(data)
    xyc = chunk(data, num_splits)

    for i, a in enumerate(k):
        cv_train_error = np.zeros(num_splits)
        cv_test_error = np.zeros(num_splits)

        for j, c in enumerate(xyc):
            y_test = c['genre']
            X_test = c.loc[:, c.columns != 'genre']

            ts = xyc[:j] + xyc[j + 1:]
            ts = pd.concat(ts[:])

            y_train = ts['genre']
            X_train = ts.loc[:, c.columns != 'genre']

            neigh = KNeighborsClassifier(n_neighbors=a)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)

            neigh.fit(X_train, y_train)

            y_p_train = neigh.predict(X_train)
            cv_train_error[j] = accuracy_score(y_train, y_p_train)

            y_p_test = neigh.predict(X_test)
            cv_test_error[j] = accuracy_score(y_test, y_p_test)

        cv_train[i] = np.mean(cv_train_error)
        cv_test[i] = np.mean(cv_test_error)

    return [cv_train, cv_test]