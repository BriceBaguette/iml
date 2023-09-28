"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from plot import plot_boundary
from data import make_balanced_dataset, make_unbalanced_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=10, learning_rate=1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate


    def fit(self, X, y):
        """Fit a logistic regression models on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=np.float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")


        # TODO insert your code here
        self.ymin = np.unique(y)[0]
        self.ymax = np.unique(y)[1]
        y = (y - np.unique(y)[0]) / (np.unique(y)[1] - np.unique(y)[0])  # normalize y
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # X -> X'
        theta = np.ones((X.shape[1], 1))

        for i in range(self.n_iter):
            v = 1 / (1 + np.exp(-theta.T @ X.T)) - y.T # P(Y = +1 | x_i, theta) - y_i
            grad = (v @ X).T / X.shape[0]
            theta = theta - self.learning_rate * grad

        self.theta = theta
        return self


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # TODO insert your code here
        return np.argmax(self.predict_proba(X), axis=1) * (self.ymax - self.ymin) + self.ymin

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        # TODO insert your code here
        X = np.hstack((np.ones((X.shape[0], 1)), X))  # X -> X'
        p = 1 / (1 + np.exp(-self.theta.T @ X.T).T)
        return np.hstack((1 - p, p))

if __name__ == "__main__":
    number_of_ls = 1000
    n_iters = [2, 5, 10, 20, 50, 100]
    results = np.empty((5, len(n_iters))) 
    for j in range(len(n_iters)):
        for i in range(5):
            X, y = make_unbalanced_dataset(3000, random_state=13443+i)
            X_train, X_test = X[:number_of_ls,:], X[number_of_ls:,:]
            y_train, y_test = y[:number_of_ls], y[number_of_ls:]
            clf = LogisticRegressionClassifier(n_iter=n_iters[j])
            clf = clf.fit(X_train, y_train)
            results[i, j] = clf.score(X_test, y_test)
            if i == 0:
                plot_boundary("logistic_regression-" + str(n_iters[j]), clf, X_test, y_test)
    print("Results:")
    print(results)
    print("Averages: " + str(np.mean(results, axis=0)))
    print("Standard deviations: " + str(np.std(results, axis=0)))
