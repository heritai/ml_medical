# -*- coding: utf-8 -*-
"""
neural_nets.py

This code snippet explores neural network models using Scikit-learn, TensoFlow and Keras.
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

"""
# Data Loading and Preprocessing

Load and preprocess the datasets for neural network experiments.
1.  Golub dataset (gene expression data)
2.  Breast Cancer dataset
3.  SPLEX dataset (environmental, host, and microbial data)
"""

# Load Golub dataset
try:
    Golub_X = pd.read_csv('data/Golub_X', sep=' ', header=None)  # Observations
    Golub_y = pd.read_csv('data/Golub_y', sep=' ', header=None)  # Classes
    Golub_X = Golub_X.values
    Golub_y = Golub_y.values.squeeze()  # Remove extra dimension
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

# Load SPLEX dataset
try:
    SPLEX_env = pd.read_csv('data/SPLEX_env.txt', sep=' ')
    SPLEX_host = pd.read_csv('data/SPLEX_host.txt', sep=' ')
    SPLEX_micro = pd.read_csv('data/SPLEX_micro.txt', sep=' ')
    SPLEX_class = pd.read_csv('data/classes.csv', sep=',')

    # Remove rows where SPLEX class is NA
    non_NA_indexs = ~SPLEX_class.isna().squeeze().values

    SPLEX_env = SPLEX_env.iloc[non_NA_indexs, :]
    SPLEX_host = SPLEX_host.iloc[non_NA_indexs, :]
    SPLEX_micro = SPLEX_micro.iloc[non_NA_indexs, :]
    SPLEX_class = SPLEX_class.iloc[non_NA_indexs, :]

    SPLEX = pd.concat([SPLEX_env, SPLEX_host, SPLEX_micro], axis=1)
except FileNotFoundError:
    print("Error: SPLEX dataset files not found. Please ensure 'data/SPLEX_env.txt', 'data/SPLEX_host.txt', 'data/SPLEX_micro.txt', and 'data/classes.csv' are in the correct directory.")
    raise

"""
# scikit-learn MLP Classifier

Define and evaluate a scikit-learn MLP classifier using cross-validation.
"""

# Define the scikit-learn MLP model
sk_model = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(10, 5), random_state=42) # Added random_state

# Evaluate the model using cross-validation
print('10 fold accuracy score for Gloub data (sklearn MLP):', np.mean(cross_val_score(sk_model, Golub_X, Golub_y, cv=10)))
print('10 fold accuracy score for Breast cancer data (sklearn MLP):', np.mean(cross_val_score(sk_model, Breast_X, Breast_y, cv=10)))

"""
# Keras MLP Classifier with Cross-Validation

Define a function to create, train, and evaluate a Keras MLP classifier using
cross-validation.
"""

def keras_cross_val_score(X, y):
    """
    Creates, trains, and evaluates a Keras MLP classifier using stratified
    k-fold cross-validation.

    Args:
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).

    Returns:
        list: A list of accuracy scores for each fold.
    """
    in_dim = X.shape[1]  # Input dimension

    # Define the Keras model
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=in_dim))
    model.add(Dense(1, activation='sigmoid'))  #Output layer for binary classification
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',  #Loss for binary classification
                  metrics=['accuracy'])

    batch_size = 10
    epochs = 100

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # Added shuffle and random_state

    scores = []
    for train, test in skf.split(X, y):
        model.fit(X[train], y[train], batch_size=batch_size, verbose=0, epochs=epochs, validation_split=0.1)
        score = model.evaluate(X[test], y[test], batch_size=batch_size, verbose=0)
        scores.append(score[1])  # Append accuracy

    return scores

"""
# Evaluate Keras MLP on Datasets

Evaluate the Keras MLP classifier on the loaded datasets using cross-validation.
"""

# Evaluate on Breast Cancer dataset
keras_cv_score_Breast = np.mean(keras_cross_val_score(Breast_X, Breast_y))
print("Keras model 10 fold cv score (Breast Cancer):", keras_cv_score_Breast)

# Evaluate on Golub dataset
keras_cv_score_Golub = np.mean(keras_cross_val_score(Golub_X, Golub_y))
print("Keras model 10 fold cv score (Golub):", keras_cv_score_Golub)

# Prepare SPLEX class labels
Gene_count_y = (SPLEX_class == 'HGC').squeeze().values

# Evaluate on SPLEX datasets
keras_cv_score_SPLEX_env = np.mean(keras_cross_val_score(SPLEX_env.values, Gene_count_y))
print("Keras model 10 fold cv score (SPLEX env):", keras_cv_score_SPLEX_env)

keras_cv_score_SPLEX_host = np.mean(keras_cross_val_score(SPLEX_host.values, Gene_count_y))
print("Keras model 10 fold cv score (SPLEX host):", keras_cv_score_SPLEX_host)

keras_cv_score_SPLEX_micro = np.mean(keras_cross_val_score(SPLEX_micro.values, Gene_count_y))
print("Keras model 10 fold cv score (SPLEX micro):", keras_cv_score_SPLEX_micro)

keras_cv_score_SPLEX = np.mean(keras_cross_val_score(SPLEX.values, Gene_count_y))
print("Keras model 10 fold cv score (SPLEX combined):", keras_cv_score_SPLEX)
