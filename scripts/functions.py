import pandas as pd
import numpy as np
import librosa
from sklearn.metrics import accuracy_score
from collections import defaultdict

def energy(y, win_len = 2048, hop_len = 512):

    energy = np.array([
        (1 / len(y[i: i + win_len])) * sum(abs(y[i: i + win_len] ** 2))
        for i in range(0, len(y), hop_len)
    ])

    return energy

def energy_entropy(y, win_len = 2048, hop_len = 512, num_short_blocks = 10):

    eps = np.finfo(float).eps
    entropy = []

    for i in range(0, len(y), hop_len):
        window = y[i:i + win_len]
        window_size = len(window)
        e_short = np.sum(window ** 2)
        sub_win_len = int(np.floor(window_size / num_short_blocks))

        if window_size != (sub_win_len * num_short_blocks):
            window = window[0 : (sub_win_len * num_short_blocks)]

        sub_windows = window.reshape(num_short_blocks, sub_win_len)

        e = np.sum(sub_windows ** 2) / (e_short + eps)
        entropy.append(-np.sum(e * np.log2(e + eps)))

    return entropy

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

        if scaler != None:
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

def merge_conf_matrices(cm):
    d = defaultdict(list)

    d['accuracy'] = []
    for k in cm[0].keys():
        if k != 'accuracy':
            d[k] = {}
            for k2 in cm[0][k].keys():
                d[k][k2] = []

    for dict in cm:
        d['accuracy'].append(dict['accuracy'])
        for k in dict.keys():
            if k != 'accuracy':
                for k2 in dict[k].keys():
                    d[k][k2].append(dict[k][k2])

    d['accuracy'] = np.mean(d['accuracy'])
    for k in d.keys():
        if k != 'accuracy':
            for k2 in d[k].keys():
                d[k][k2] = np.mean(d[k][k2])

    return d