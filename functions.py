import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt


def detectOutliers(df, rm=False):
    for f in df.columns:
        q1 = np.percentile(df[f], 25, interpolation='midpoint')
        q3 = np.percentile(df[f], 75, interpolation='midpoint')
        iqr = q3 - q1

        if not rm:
            return df.loc[(df[f] < (q1 - (1.5 * iqr))) | (df[f] > (q3 + (1.5 * iqr)))]

        else:
            return df.loc[(df[f] > (q1 - (1.5 * iqr))) & (df[f] < (q3 + (1.5 * iqr)))]

def chunk(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# def k_fold_cv(data, num_splits, alphas):
#     cv_train = np.zeros(len(alphas))
#     cv_test = np.zeros(len(alphas))
#     cv_r2_train = np.zeros(len(alphas))
#     cv_r2_test = np.zeros(len(alphas))
#
#     xyc = chunk(data, num_splits)
#
#     for i, a in enumerate(alphas):
#         cv_train_error = np.zeros(num_splits)
#         cv_test_error = np.zeros(num_splits)
#         cv_r2_train_score = np.zeros(num_splits)
#         cv_r2_test_score = np.zeros(num_splits)
#
#         for j, c in enumerate(xyc):
#             y_test = c.iloc[:, -1]
#             X_test = c.iloc[:, :-1]
#
#             ts = xyc[:j] + xyc[j + 1:]
#             ts = pd.concat(ts[:])
#
#             y_train = ts.iloc[:, -1]
#             X_train = ts.iloc[:, :-1]
#
#             w = ridge_w(X_train, y_train, a)
#
#             y_p_train = np.dot(X_train, w)
#             cv_train_error[j] = mean_squared_error(y_train, y_p_train)
#             cv_r2_train_score[j] = adjusted_r2(y_train, y_p_train, X_train.shape[0], X_train.shape[1])
#
#             y_p_test = np.dot(X_test, w)
#             cv_test_error[j] = mean_squared_error(y_test, y_p_test)
#             cv_r2_test_score[j] = adjusted_r2(y_test, y_p_test, X_test.shape[0], X_test.shape[1])
#
#         cv_train[i] = np.mean(cv_train_error)
#         cv_test[i] = np.mean(cv_test_error)
#         cv_r2_train[i] = np.mean(cv_r2_train_score)
#         cv_r2_test[i] = np.mean(cv_r2_test_score)
#
#     return [cv_train, cv_test, cv_r2_train, cv_r2_test]