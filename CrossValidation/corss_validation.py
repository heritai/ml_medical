# -*- coding: utf-8 -*-
"""
corss_validation.py

This code snippet demonstrates cross-validation techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

"""
# Data Loading

Load the datasets for cross-validation experiments.
1.  Breast Cancer Wisconsin (Diagnostic) Data Set
2.  Mouse Protein Expression Data Set
"""

# Load Breast Cancer dataset
try:
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    breast_data=pd.read_csv(url,header=None)

    breast_cancer_X=breast_data.drop([0,1],axis=1)
    breast_cancer_Y=breast_data[1]
except:
    print("Error loading breast cancer data")

# Load Mouse Protein Expression Data Set
try:
    url='https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls'
    mice_data = pd.read_excel(url,'Hoja1')

    mice_data_X=mice_data.drop(['MouseID','Genotype','Treatment','Behavior','class'],axis=1)
    mice_data_Y=mice_data['class']

    #fill missing values
    mice_data_X=mice_data_X.fillna(value=mice_data_X.mean(axis=0).to_dict())
except:
    print("Error loading mice data")

"""
# Simulated Datasets

Create three simulated datasets for cross-validation experiments.
1.  Two informative features, one cluster per class (classification-like data)
2.  Three blobs (well-separated clusters)
3.  Non-linearly separated data sets (moons shape)
"""

# First simulated data set
plt.figure() # Ensure plots are separated
plt.title("Two informative features, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42) # Added random_state
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
plt.show()

# Second simulated data set
plt.figure()
plt.title("Three blobs", fontsize='small')
X2, Y2 = make_blobs(n_samples=200, n_features=2, centers=3, random_state=42) # Added random_state
plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2, s=25, edgecolor='k')
plt.show()

# Third simulated data set
plt.figure()
plt.title("Non-linearly separated data sets", fontsize='small')
X3, Y3 = make_moons(n_samples=200, shuffle=True, noise=None, random_state=42) # Added random_state
plt.scatter(X3[:, 0], X3[:, 1], marker='o', c=Y3, s=25, edgecolor='k')
plt.show()

"""
# Plotting Logistic Regression Decision Boundary

Define a function to fit a logistic regression model and plot the decision
boundary.
"""

def fitRegPlot(X, Y):
    """
    Fits a logistic regression model and plots the decision boundary.

    Args:
        X (np.ndarray): The input data (features).
        Y (np.ndarray): The target variable (labels).
    """
    logreg = linear_model.LogisticRegression(C=1e5, random_state=42)  # Added random_state

    logreg.fit(X, Y)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()

# Plot decision boundaries for simulated datasets
fitRegPlot(X1, Y1)
fitRegPlot(X2, Y2)
fitRegPlot(X3, Y3)

"""
# Cross-Validation with scikit-learn

Define a function to perform cross-validation using scikit-learn's
`cross_val_score` function.
"""

from sklearn.model_selection import cross_val_score

def crossValRegFit(X, y):
    """
    Performs cross-validation using scikit-learn's `cross_val_score` function.

    Args:
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).
    """
    logreg = linear_model.LogisticRegression(C=1e5, random_state=42) #Added random_state
    scores = cross_val_score(logreg, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Perform cross-validation for simulated datasets
crossValRegFit(X1, Y1)
crossValRegFit(X2, Y2)
crossValRegFit(X3, Y3)

"""
# Logistic Regression Implementation from Scratch

Implement logistic regression from scratch using gradient descent and define
functions for sigmoid and log-likelihood.
"""

def sigmoid(scores):
    """
    Calculates the sigmoid function.

    Args:
        scores (np.ndarray): The input scores.

    Returns:
        np.ndarray: The sigmoid values.
    """
    return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
    """
    Calculates the log-likelihood of the logistic regression model.

    Args:
        features (np.ndarray): The input features.
        target (np.ndarray): The target variable (labels).
        weights (np.ndarray): The weights of the logistic regression model.

    Returns:
        float: The log-likelihood value.
    """
    scores = np.dot(features, weights)
    ll = np.sum(target * scores - np.log(1 + np.exp(scores)))
    return ll

def logistic_regression(features, target, max_steps, treshold, learning_rate, add_intercept=False):
    """
    Implements logistic regression using gradient descent.

    Args:
        features (np.ndarray): The input features.
        target (np.ndarray): The target variable (labels).
        max_steps (int): The maximum number of steps for gradient descent.
        treshold (float): The convergence threshold.
        learning_rate (float): The learning rate for gradient descent.
        add_intercept (bool): Whether to add an intercept term.

    Returns:
        np.ndarray: The learned weights of the logistic regression model.
    """
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(max_steps):
        per_weights = weights.copy()
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        if np.linalg.norm(per_weights - weights) < treshold:
            print("converged in ", step, " steps")
            break

    return weights

# Train logistic regression model
mylogRegWeights = logistic_regression(X1, Y1, max_steps=30000, treshold=1e-5, learning_rate=5e-5, add_intercept=True)

"""
# Cross-Validation from Scratch

Define a function to perform cross-validation from scratch using the
implemented logistic regression.
"""

def crossValScore(X, y):
    """
    Performs cross-validation from scratch using the implemented logistic regression.

    Args:
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).
    """
    kf = KFold(n_splits=10,shuffle = True, random_state = 42) # added shuffle, random_state
    scSklearn = []
    scLogReg = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        logregSklearn = linear_model.LogisticRegression(C=1e5, max_iter=3000, random_state=42) # Added random_state
        logregSklearn.fit(X_train, y_train)
        scSklearn.append(logregSklearn.score(X_test, y_test))

        mylogRegWeights = logistic_regression(X_train, y_train, max_steps=30000, treshold=1e-4, learning_rate=5e-5, add_intercept=True)
        proba = sigmoid(np.hstack([np.ones((X_test.shape[0], 1)), X_test]).dot(mylogRegWeights))
        scLogReg.append(accuracy_score(y_test, proba > 0.5)) # corrected the parameters order in accuracy_score

    print("sklearn logistic regression score: ", np.array(scSklearn).mean())
    print("my logistic regression score: ", np.array(scLogReg).mean())

# Perform cross-validation for simulated and real-world datasets
crossValScore(X1, Y1)
crossValScore(X2, Y2 == 1)
crossValScore(X3, Y3)
try:
    crossValScore(breast_cancer_X.values, breast_cancer_Y == 'M')
except NameError:
    print("Breast cancer data not loaded, skipping cross-validation.")
try:
    crossValScore(mice_data_X.values, mice_data_Y == 'c-CS-m')
except NameError:
    print("Mice data not loaded, skipping cross-validation.")
