# -*- coding: utf-8 -*-
"""
dimension_redcut.py

This code snippet explores dimension reduction techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

"""
# Data Loading

Load the datasets for dimension reduction experiments.
1.  Golub dataset (gene expression data)
2.  Breast Cancer dataset
"""

# Load Golub dataset
try:
    Golub_X = pd.read_csv('data/Golub_X', sep=' ')  # Observations
    Golub_y = pd.read_csv('data/Golub_y', sep=' ')  # Classes
    Golub_y = Golub_y.values.squeeze() # Remove extra dimension
except FileNotFoundError:
    print("Error: Golub dataset files not found. Please ensure 'data/Golub_X' and 'data/Golub_y' are in the correct directory.")
    raise

# Load Breast Cancer dataset
try:
    X = pd.read_csv('data/Breast.txt', sep=' ')
    Breast_y = X.values[:, 30]  # Classes
    Breast_y = Breast_y == 1  # Convert to boolean
    Breast_X = X.values[:, 0:29]  # Observations
except FileNotFoundError:
    print("Error: Breast Cancer dataset file not found. Please ensure 'data/Breast.txt' is in the correct directory.")
    raise

"""
# Dimension Reduction and Plotting Functions

Define functions to apply different dimension reduction techniques and visualize
the results.
1.  PCA (Principal Component Analysis)
2.  Kernel PCA
3.  Incremental PCA
"""

def PCA_fit_plot(X, y):
    """
    Applies PCA and plots the first two principal components.

    Args:
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).
    """
    pca = PCA(n_components=2, random_state=42)  # Added random_state
    X_pca = pca.fit_transform(X)
    plt.figure()  # Create a new figure for each plot
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=y, s=25, edgecolor='k')
    plt.title("PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Apply PCA to the two datasets
PCA_fit_plot(Golub_X, Golub_y)
PCA_fit_plot(Breast_X, Breast_y)

def kerPca_fit_plot(X, y):
    """
    Applies Kernel PCA and plots the first two principal components.

    Args:
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).
    """
    transformer = KernelPCA(n_components=2, kernel='linear', random_state=42)  # Added random_state
    X_pca = transformer.fit_transform(X)
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=y, s=25, edgecolor='k')
    plt.title("Kernel PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Apply Kernel PCA to the two datasets
kerPca_fit_plot(Golub_X, Golub_y)
kerPca_fit_plot(Breast_X, Breast_y)

def IcrementalPca_fit_plot(X, y):
    """
    Applies Incremental PCA and plots the first two principal components.

    Args:
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).
    """
    transformer = IncrementalPCA(n_components=2, batch_size=100)
    X_pca = transformer.fit_transform(X)
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=y, s=25, edgecolor='k')
    plt.title("Incremental PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

# Apply Incremental PCA to the two datasets
IcrementalPca_fit_plot(Golub_X, Golub_y)
IcrementalPca_fit_plot(Breast_X, Breast_y)

"""
# Evaluate Classification Accuracy with PCA

Define a function to evaluate the classification accuracy of SVM and Logistic
Regression with different numbers of PCA components.
"""

from sklearn.linear_model import LogisticRegression

def fitPca_accuracy(X, y):
    """
    Evaluates the classification accuracy of SVM and Logistic Regression with
    different numbers of PCA components.

    Args:
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).

    Returns:
        list: A list of tuples, where each tuple contains the SVM and Logistic
              Regression accuracy scores for a given number of components.
    """
    componnt = [2, 5, 10, 20]
    acc = []
    for c in componnt:
        pca = PCA(n_components=c, random_state=42)  # Added random_state
        pca.fit(X)
        X_pca = pca.transform(X)

        svcModel = svm.SVC(random_state=42)  # Added random_state
        svcModel.fit(X_pca, y)

        logreg = LogisticRegression(random_state=42)  # Added random_state
        logreg.fit(X_pca, y)

        acc.append((svcModel.score(X_pca, y), logreg.score(X_pca, y)))
    plt.figure()
    plt.plot(componnt, [a[0] for a in acc], label="SVM")  # Extract SVM scores
    plt.plot(componnt, [a[1] for a in acc], label="Logistic Regression")  # Extract LogReg scores
    plt.xlabel("Number of Components")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Components")
    plt.legend()
    plt.show()
    return acc

# Evaluate classification accuracy for the two datasets
fitPca_accuracy(Golub_X, Golub_y)
fitPca_accuracy(Breast_X, Breast_y)

"""
# Compare PCA and LDA for Visualization

Compare PCA and LDA for visualizing the data in a reduced-dimensional space.
"""

def Pca_Lda_compare(X, y):
    """
    Compares PCA and LDA for visualizing the data in a reduced-dimensional space.

    Args:
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).
    """
    print("WARNING")
    print('''lda works when thaere are more than 2 features''')
    target_names = ["0", "1"]

    pca = PCA(n_components=2, random_state=42)  # Added random_state
    X_r = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=1) # Changed to 1 component - LDA needs n_components < min(n_classes - 1, n_features)
    X_r2 = lda.fit(X, y).transform(X)

    # PCA plot
    plt.figure()
    colors = ['navy', 'turquoise']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    # LDA plot
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r2[y == i], np.zeros_like(X_r2[y == i]), alpha=0.8, color=color,  # Plot LDA on one dimension with a zero y value
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA')
    plt.xlabel("Discriminant Component 1") # Set appropriate x label
    plt.yticks([]) # Remove Y ticks as we are plotting along one line.
    plt.show()

# Compare PCA and LDA for the two datasets
Pca_Lda_compare(Golub_X, Golub_y)
Pca_Lda_compare(Breast_X, Breast_y)

"""
# Evaluate Error Rate with PCA

Define a function to evaluate the error rate of SVM and Logistic Regression
with different numbers of PCA components.
"""

def fitPca_Error(X, y):
    """
    Evaluates the error rate of SVM and Logistic Regression with different
    numbers of PCA components.

    Args:
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).

    Returns:
        list: A list of tuples, where each tuple contains the SVM and Logistic
              Regression error rates for a given number of components.
    """
    componnt = [2, 5, 10, 20]
    err = []
    for c in componnt:
        pca = PCA(n_components=c, random_state=42)  # Added random_state
        pca.fit(X)
        X_pca = pca.transform(X)

        svcModel = svm.SVC(random_state=42)  # Added random_state
        svcModel.fit(X_pca, y)

        logreg = LogisticRegression(random_state=42)  # Added random_state
        logreg.fit(X_pca, y)

        err.append((1 - svcModel.score(X_pca, y), 1 - logreg.score(X_pca, y)))

    svcModel = svm.SVC(random_state=42)  # Added random_state
    svcModel.fit(X, y)

    logreg = LogisticRegression(random_state=42)  # Added random_state
    logreg.fit(X, y)

    err.append((1 - svcModel.score(X, y), 1 - logreg.score(X, y)))
    componnt.append(X.shape[1]) #100 changed to original # of components

    plt.figure()
    plt.plot(componnt, [e[0] for e in err], label="SVM")  # Extract SVM errors
    plt.plot(componnt, [e[1] for e in err], label="Logistic Regression")  # Extract LogReg errors
    plt.xlabel("Number of Components")
    plt.ylabel("Error Rate")
    plt.title("Error Rate vs Number of Components")
    plt.legend()
    plt.show()
    return err

# Evaluate error rate for the two datasets
fitPca_Error(Golub_X, Golub_y)
fitPca_Error(Breast_X, Breast_y)

print("error rates of reduced data are not smaller but they are competing with the full model")
