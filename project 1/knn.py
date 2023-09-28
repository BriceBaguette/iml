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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score


# (Question 2)

# Put your funtions here
# ...
def optimal_param(parameter_values, X_train, y_train):
    best_param = parameter_values[0]
    best_score = 0
    for param in parameter_values:
        clf = KNeighborsClassifier(n_neighbors=param)
        score = np.mean(cross_val_score(clf, X_train, y_train, cv=10))
        if score > best_score:
            best_score = score
            best_param = param
    return best_param, best_score


if __name__ == "__main__":
    parameter_values = [1,5,50,100,500]
    number_of_ls = 1000
    X, y = make_unbalanced_dataset(3000, random_state=13443)
    X_train, X_test = X[:number_of_ls,:], X[number_of_ls:,:]
    y_train, y_test = y[:number_of_ls], y[number_of_ls:]
    for param in parameter_values:
        clf = KNeighborsClassifier(n_neighbors=param)
        clf = clf.fit(X_train, y_train)
        plot_boundary("knn-" + str(param), clf, X_test, y_test)

    optimal = optimal_param(parameter_values, X_train, y_train)
    print("Best 10-fold cv parameter: " + str(optimal[0]) + " with a mean accuracy of " + str(optimal[1]))

    ts_sizes = [50, 150, 250, 350, 450, 500]
    X_test, y_test = make_unbalanced_dataset(500, random_state=13442)
    accuracies = np.zeros((500, len(ts_sizes), 10))
    for i in range(10):
        X, y = make_unbalanced_dataset(500, random_state=13443+i)
        for j in range(len(ts_sizes)):
            for k in range(ts_sizes[j]):
                X_train = X[:ts_sizes[j],:]
                y_train = y[:ts_sizes[j]]
                clf = KNeighborsClassifier(n_neighbors=k+1)
                clf = clf.fit(X_train, y_train)
                accuracies[k, j, i] = clf.score(X_test, y_test)
    mean_accuracies = np.mean(accuracies, axis=2)
    print("Mean accuracies for every training set size and number of neighbors (each column corresponds to a training set size):")
    print(mean_accuracies)
    print("(0 means the training set is smaller than n_neighbors)")

    
    optimal_neighbors = np.zeros((len(ts_sizes)))
    for i in range(len(ts_sizes)):
        optimal_neighbors[i] = np.argmax(mean_accuracies[:,i])+1
    print("Optimal values of n_neighbors for each training set size:")
    print(optimal_neighbors)
    plt.plot(ts_sizes,optimal_neighbors)
    plt.ylabel('Optimal values for k')
    plt.xlabel('Training sample size')
    plt.show()
    pass
