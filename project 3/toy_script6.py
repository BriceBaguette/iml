#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Antonio Sutera & Yann Claes

import os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from tsfresh.transformers import RelevantFeatureAugmenter, FeatureAugmenter
import pandas as pd
import tsfresh
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.ensemble import StackingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, GroupKFold

def load_data(data_path='data'):

    FEATURES = range(2, 33)
    N_TIME_SERIES = 3500

    # Create the training and testing samples
    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')
    X_train, X_test = [np.zeros((N_TIME_SERIES, (len(FEATURES) * 512))) for i in range(2)]

    for f in FEATURES:
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
        X_train[:, (f-2)*512:(f-2+1)*512] = data
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        X_test[:, (f-2)*512:(f-2+1)*512] = data

    y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))

    train_subjects = np.loadtxt(os.path.join(LS_path,'subject_Id.txt'))
    test_subjects = np.loadtxt(os.path.join(TS_path,'subject_Id.txt'))

    print('X_train size: {}.'.format(X_train.shape))
    print('y_train size: {}.'.format(y_train.shape))
    print('X_test size: {}.'.format(X_test.shape))
    print('train_subjects size: {}.'.format(train_subjects.shape))
    print('test_subjects size: {}.'.format(test_subjects.shape))

    return X_train, y_train, X_test, train_subjects, test_subjects

def fourier(X, series_length=512):
    new_length = series_length // 2 + 1
    n_series = X.shape[1] // series_length
    X_out = np.empty((X.shape[0], n_series * new_length))

    for i in range(n_series):
        X_out[:, i * new_length: (i+1) * new_length] = np.real(np.fft.rfft(X[:, i * series_length: (i+1) * series_length]))

    return X_out

def extract_features(X, series_length=512, default_fc_parameters=MinimalFCParameters()):
    n_series = X.shape[1] // series_length
    X_out = np.empty((X.shape[0] * series_length, n_series + 2))

    for i in range(X.shape[0]):
        X_out[i * series_length:(i+1) * series_length, 0] = i
        X_out[i * series_length:(i+1) * series_length, 1] = range(series_length)

    for i in range(n_series):
        X_out[:, i+2] = X[:, i * series_length:(i+1) * series_length].ravel(order='C')

    X_out = pd.DataFrame(X_out, columns=["sample", "time", *map(lambda x: "f"+str(x), range(n_series))])  # had problems when the name of the columns were purely digits, resulting in this mess to add an f (for feature) at the start

    X_out = tsfresh.extract_features(X_out, column_id='sample', column_sort='time', default_fc_parameters=default_fc_parameters, n_jobs=os.cpu_count())

    return X_out.to_numpy()

def normalize(X, subjects, series_length=512):
    """
    Removes the mean and divides by standard deviation accross all measurements from that subject.
    For example, replace each bpm measurement for subject 1 by itself minus the mean bpm accross all bpm measurements for subject 1,
    the whole thing divided by the standard deviation of bpms (again accross all bpm measurements for subject 1).
    Does this for every feature of every subject, in an attempt to make measurements less linked to the particular subject.
    Ignores missing values (-999999.99)
    """
    subjects = subjects.reshape((subjects.size, 1))
    for s in np.unique(subjects):
        for i in range(X.shape[1]//series_length):
            indices = np.where((subjects == s) & (X[:, i*series_length:(i+1)*series_length] != -999999.99))
            indices = (indices[0], indices[1] + i*series_length)
            mean = np.mean(X[indices])
            std = np.std(X[indices])
            X[indices] -= mean
            X[indices] /= std

    return X

def average_missing(X, subjects):
    """
    Replaces each missing value by the average measurement at that time for that subject.
    """
    for s in np.unique(subjects):
        for i in range(X.shape[1]):
            missing = np.where((subjects == s) & (X[:, i] == -999999.99))[0]
            not_missing = np.where((subjects == s) & (X[:, i] != -999999.99))[0]
            X[missing, i] = np.mean(X[not_missing, i])

    return X

def normalize_and_average_missing(X, subjects, series_length=512):
    return average_missing(normalize(X, subjects, series_length), subjects)

def add_magnitudes(X):
    """
    Adds magnitudes for 3D inputs
    """
    magnitudes = np.empty((X.shape[0], 9*512))
    feature_indices = [2,5,8,12,15,18,22,25,28]  # for each vector we have the index of the first component (the other two following right after)
    for i in range(len(feature_indices)):
        for j in range(512):
            magnitudes[:, i*512+j] = np.linalg.norm(X[:, [feature_indices[i]*512+j, (feature_indices[i]+1)*512+j, (feature_indices[i]+2)*512+j]], axis=1)

    return np.hstack((X, magnitudes))

def write_submission(y, where, submission_name='submission.csv'):

    os.makedirs(where, exist_ok=True)

    SUBMISSION_PATH = os.path.join(where, submission_name)
    if os.path.exists(SUBMISSION_PATH):
        os.remove(SUBMISSION_PATH)

    y = y.astype(int)
    outputs = np.unique(y)

    # Verify conditions on the predictions
    if np.max(outputs) > 14:
        raise ValueError('Class {} does not exist.'.format(np.max(outputs)))
    if np.min(outputs) < 1:
        raise ValueError('Class {} does not exist.'.format(np.min(outputs)))
    
    # Write submission file
    with open(SUBMISSION_PATH, 'a') as file:
        n_samples = len(y)
        if n_samples != 3500:
            raise ValueError('Check the number of predicted values.')

        file.write('Id,Prediction\n')

        for n, i in enumerate(y):
            file.write('{},{}\n'.format(n+1, int(i)))

    print('Submission {} saved in {}.'.format(submission_name, SUBMISSION_PATH))

def custom_cross_val(clf, X, y, subjects):

    results = np.empty((np.unique(subjects).size))

    i = 0
    for s in np.unique(subjects):
        indices_val = np.where(subjects == s)[0]
        indices_train = np.where(subjects != s)[0]
        clf.fit(X[indices_train, :], y[indices_train])
        results[i] = clf.score(X[indices_val, :], y[indices_val])
        i += 1
   
    return results

def custom_cross_val_stacking(estimators, X, y, subjects):

    n_subjects = np.unique(subjects).size
    results = np.empty((n_subjects))

    i = 0
    for s in np.unique(subjects):
        indices_val = np.where(subjects == s)[0]
        indices_train = np.where(subjects != s)[0]
        cv = CustomKFold(n_subjects-1, groups=subjects[indices_train])
        clf = StackingClassifier(estimators, final_estimator=RandomForestClassifier(criterion="entropy", n_jobs=-1, random_state=4, n_estimators=100), cv=cv, n_jobs=-1)
        clf.fit(X[indices_train, :], y[indices_train])
        results[i] = clf.score(X[indices_val, :], y[indices_val])
        i += 1

    return results

"""def custom_cross_val_pipeline(clf, y, data_path="data"):

    LS_path = os.path.join(data_path, 'LS')

    results = np.empty((5))

    subjects = np.loadtxt(os.path.join(LS_path,'subject_Id.txt'))
    for i in range(1,6):
        indices_val = np.where(subjects == i)[0]
        indices_train = np.where(subjects != i)[0]
        X_train = pd.DataFrame(index=indices_train)
        X_val = pd.DataFrame(index=indices_val)
        clf = clf.fit(X_train, y[indices_train])
        results[i-1] = clf.score(X_val, y[indices_val])
    
    return results"""

def test_score(clf, X_test):
    y_test = np.loadtxt(os.path.join('hacks', 'activity_Id.txt'))
    indices = np.where(y_test != 0)[0]
    return clf.score(X_test[indices,:], y_test[indices])

class DefaultTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[:, :31*512]

class FourierTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[:, 31*512:31*512+31*257]

class FeatureExtractTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[:, 31*512+31*257:31*512+31*257+279]

class NormFeatureExtractTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[:, 31*512+31*257+279:31*512+31*257+279*2]

class CustomKFold(GroupKFold):
    def __init__(self, n_splits=5, groups=None):
        super().__init__(n_splits)
        self.groups = groups
    def split(self, X, y=None, groups=None):
        return super().split(X, y, groups=self.groups)

if __name__ == '__main__':

    X_train, y_train, X_test, train_subjects, test_subjects = load_data()
    # X_train = np.hstack((extract_features(X_train), extract_features(fourier(X_train), 257)))
    # print(extract_features(X_train[0:2,:]).shape)
    """X_train = np.hstack((X_train, fourier(X_train), extract_features(X_train)))  #, extract_features(normalize_and_average_missing(X_train, train_subjects))))
    X_test = np.hstack((X_test, fourier(X_test), extract_features(X_test)))  #, extract_features(normalize_and_average_missing(X_test, test_subjects))))
    X_train = X_train[:, 31*512+31*257:31*512+31*257+279]
    X_test = X_test[:, 31*512+31*257:31*512+31*257+279]"""
    fc_parameters = {'median':None, 'mean': None, 'length': None, 'standard_deviation': None, 'variance': None, 'root_mean_square': None, 'maximum': None, 'minimum': None, 'agg_linear_trend':[{"attr": 'pvalue', "f_agg": 'mean', "chunk_len": 50}]}
    X_train = extract_features(X_train, 512, fc_parameters)
    X_test = extract_features(X_test, 512, fc_parameters)

    clf = RandomForestClassifier(criterion="entropy", n_jobs=-1, random_state=4, n_estimators=100)

    clf.fit(X_train, y_train)
    print(test_score(clf, X_test))
