import pandas as pd
import numpy as np
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

def n_shuffle_cv(df, num_splits, params, num_shuffles):
    cv_train, cv_test = [[] for i in range(num_shuffles)], [[] for i in range(num_shuffles)]

    for i in range(num_shuffles):
        train, test = k_fold_cv(df, num_splits, params)
        cv_train[i] = train
        cv_test[i] = test


    train_df = pd.DataFrame(cv_train)
    test_df = pd.DataFrame(cv_test)
    final_train = train_df.mean()
    final_test = test_df.mean()

    return [final_train, final_test]

def k_fold_cv(data, num_splits, classifier, scaler):

    y_t = []
    y_p = []
    cv_train_error = np.zeros(num_splits)
    cv_test_error = np.zeros(num_splits)

    xyc = np.array_split(data, num_splits)

    for j, c in enumerate(xyc):
        y_test = c['genre']
        X_test = c.loc[:, c.columns != 'genre']

        ts = xyc[:j] + xyc[j + 1:]
        ts = pd.concat(ts[:])

        y_train = ts['genre']
        X_train = ts.loc[:, c.columns != 'genre']

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        classifier.fit(X_train, y_train)

        y_p_train = classifier.predict(X_train)
        cv_train_error[j] = accuracy_score(y_train, y_p_train)

        y_p_test = classifier.predict(X_test)
        y_t.append(pd.Series(y_test))
        y_p.append(pd.Series(y_p_test, index=y_test.index))
        cv_test_error[j] = accuracy_score(y_test, y_p_test)

    cv_train = np.mean(cv_train_error)
    cv_test = np.mean(cv_test_error)
    y_t = pd.Series(pd.concat(y_t).values, name='Actual')
    y_p = pd.Series(pd.concat(y_p).values, name='Predicted')

    return [cv_train, cv_test, y_t, y_p]