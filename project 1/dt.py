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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# (Question 1)

# Put your funtions here
# ...


if __name__ == "__main__":
    # Put your code here
    number_of_ls = 1000
    parameter_values = [2, 8, 32, 64, 128, 500]
    test_results = np.zeros((5, len(parameter_values)))
    for i in range(5):
        X, y = make_unbalanced_dataset(3000, random_state=13443+i)
        X_train, X_test = X[:number_of_ls,:], X[number_of_ls:,:]
        y_train, y_test = y[:number_of_ls], y[number_of_ls:]
        for j in range(len(parameter_values)):
            clf = DecisionTreeClassifier(min_samples_split=parameter_values[j], random_state=0)
            clf = clf.fit(X_train, y_train)
            test_results[i, j] = clf.score(X_test, y_test)
            if i == 0:
                plot_boundary("dt-" + str(parameter_values[j]), clf, X_test, y_test)
    print("Individual results:")
    print(test_results)
    print("Averages:")
    print(np.mean(test_results, axis=0))
    print("Standard deviations:")
    print(np.std(test_results, axis=0))
    pass
