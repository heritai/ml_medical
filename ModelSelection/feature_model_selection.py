# -*- coding: utf-8 -*-
"""
feature_model_selection.py

This code snippet demonstrates some techniques for feature selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFdr

"""
# Data Loading

Load the datasets for feature selection experiments.
1.  Golub dataset (gene expression data)
2.  Breast Cancer dataset
"""

# Load Golub dataset
try:
    Golub_X = pd.read_csv('data/Golub_X', sep=' ')  # Observations
    Golub_y = pd.read_csv('data/Golub_y', sep=' ')  # Classes
except FileNotFoundError:
    print("Error: Golub dataset files not found. Please ensure 'data/Golub_X' and 'data/Golub_y' are in the correct directory.")
    raise

# Load Breast Cancer dataset
try:
    X = pd.read_csv('data/Breast.txt', sep=' ')
    Breast_y = X.values[:, 30]  # Classes
    Breast_X = X.values[:, 0:29]  # Observations
except FileNotFoundError:
    print("Error: Breast Cancer dataset file not found. Please ensure 'data/Breast.txt' is in the correct directory.")
    raise

"""
# Feature Selection Techniques

Apply different feature selection techniques to the loaded datasets.
1.  Variance Threshold: Removes features with low variance.
2.  SelectFdr: Selects features based on false discovery rate.
3.  Lasso Regression: L1 regularization for feature selection.
4.  Linear SVC: Linear Support Vector Classifier with L1 penalty.
5.  Elastic Net: Combines L1 and L2 regularization for feature selection.
"""

# Variance Threshold
sel1 = VarianceThreshold(threshold=0.01)
Golub_X_varTresh = sel1.fit_transform(Golub_X)

sel2 = VarianceThreshold(threshold=0.1)
Breast_X_varTresh = sel2.fit_transform(Breast_X)

# SelectFdr (False Discovery Rate)
selFdr1 = SelectFdr(alpha=0.01)
Golub_X_selFdr = selFdr1.fit_transform(Golub_X, Golub_y.values.ravel())

selFdr2 = SelectFdr(alpha=0.05)
Breast_X_selFdr = selFdr2.fit_transform(Breast_X, Breast_y)

# Lasso Regression (L1 regularization)
LassoReg1 = linear_model.Lasso(alpha=0.01, random_state=42)  # added random_state
LassoReg1.fit(Golub_X, Golub_y)
print("Number of selected features on Golub dataset (Lasso): ", LassoReg1.sparse_coef_.getnnz())

LassoReg2 = linear_model.Lasso(alpha=0.1, random_state=42)  # added random_state
LassoReg2.fit(Breast_X, Breast_y)
print("Number of selected features on Breast Cancer dataset (Lasso): ", LassoReg2.sparse_coef_.getnnz())

# Linear SVC (L1 penalty)
LinSVC1 = LinearSVC(C=1, penalty="l1", dual=False, random_state=42)  # added random_state
LinSVC1.fit(Golub_X, Golub_y.values.ravel())  # ravel Golub_y to avoid DataConversionWarning
print("Number of selected features on Golub dataset (LinearSVC): ", np.sum(LinSVC1.coef_ != 0)) # Corrected the condition

LinSVC2 = LinearSVC(C=1, penalty="l1", dual=False, random_state=42)  # added random_state
LinSVC2.fit(Breast_X, Breast_y)
print("Number of selected features on Breast Cancer dataset (LinearSVC): ", np.sum(LinSVC2.coef_ != 0)) # Corrected the condition

# Elastic Net (L1 and L2 regularization)
ElasNet1 = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=42)  # added random_state
ElasNet1.fit(Golub_X, Golub_y)
print("Number of selected features on Golub dataset (ElasticNet): ", ElasNet1.sparse_coef_.getnnz())

ElasNet2 = ElasticNet(alpha=0.1, l1_ratio=0.7, random_state=42)  # added random_state
ElasNet2.fit(Breast_X, Breast_y)
print("Number of selected features on Breast Cancer dataset (ElasticNet): ", ElasNet2.sparse_coef_.getnnz())

"""
# Model Evaluation

Evaluate the performance of the models with selected features using accuracy
score.
"""

# Print accuracy scores
print("\nAccuracy on Golub dataset:")
print("Accuracy score Lasso Regression: ", LassoReg1.score(Golub_X, Golub_y))
print("Accuracy score Linear SVC: ", LinSVC1.score(Golub_X, Golub_y))
print("Accuracy score Elastic Net: ", ElasNet1.score(Golub_X, Golub_y))

print("\nAccuracy on Breast Cancer dataset:")
print("Accuracy score Lasso Regression: ", LassoReg2.score(Breast_X, Breast_y))
print("Accuracy score Linear SVC: ", LinSVC2.score(Breast_X, Breast_y))
print("Accuracy score Elastic Net: ", ElasNet2.score(Breast_X, Breast_y))

#print("\nLinear SVC is the best on both datasets") # This conclusion needs proper validation and comparison using cross-validation techniques.
