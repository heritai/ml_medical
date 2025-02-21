# -*- coding: utf-8 -*-
"""
EM Algorithm.py

Objective:
Understand the Expectation-Maximization algorithm and learn how to apply it
in practical scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import mixture
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn import metrics
import pandas as pd

"""
# 1D Gaussian Mixture Model (GMM) Implementation

Implement the EM algorithm for a 1D GMM with two components.
"""

# Generate sample data for 1D GMM
mu1, sigma1 = 0, 0.3  # mean and standard deviation for component 1
s1 = np.random.normal(mu1, sigma1, 100)  # 100 samples from component 1
y1 = np.repeat(0, 100)  # Labels for component 1
mu2, sigma2 = 2, 0.3  # mean and standard deviation for component 2
s2 = np.random.normal(mu2, sigma2, 100)  # 100 samples from component 2
y2 = np.repeat(1, 100)  # Labels for component 2

data = np.concatenate([s1, s2])  # Combine the samples
y = np.concatenate([y1, y2])  # Combine the labels

def pr_single_comp(mu, sigma, x):
    """
    Calculates the probability density for each data point x under a single
    Gaussian component.

    Args:
        mu (float): Mean of the Gaussian component.
        sigma (float): Standard deviation of the Gaussian component.
        x (np.ndarray): Data points (1D array).

    Returns:
        list: List of probability densities for each data point.
    """
    prob = []
    for i in range(0, x.shape[0]):
        prob.append(np.exp(-0.5 * ((x[i] - mu) / sigma)**2) / sigma)  # Corrected indexing
    return prob

def pr_single_normalized(mu, sigma, x):
    """
    Calculates normalized probabilities for each data point belonging to each
    Gaussian component.

    Args:
        mu (list): List of means for each Gaussian component.
        sigma (list): List of standard deviations for each Gaussian component.
        x (np.ndarray): Data points (1D array).

    Returns:
        list: List of normalized probabilities for each data point belonging to
              each Gaussian component.
    """
    unnorm_prob = [pr_single_comp(mu[k], sigma[k], x) for k in range(len(mu))]  # List of lists
    normalization = np.sum(unnorm_prob, axis=0)  # Sum across components for each point
    prob = []
    for i in range(x.shape[0]):
        component_probs = [unnorm_prob[k][i] / normalization[i] for k in range(len(mu))] # Prob for each component
        prob.append(component_probs)  # list of component probs for each point
    return prob

def update_mu(x, mu, sigma):
    """
    Updates the means of the Gaussian components based on the responsibilities.

    Args:
        x (np.ndarray): Data points (1D array).
        mu (list): List of current means for each Gaussian component.
        sigma (list): List of current standard deviations for each Gaussian component.

    Returns:
        np.ndarray: Updated means for each Gaussian component.
    """
    prob = pr_single_normalized(mu, sigma, x)
    hat_mu = np.zeros(len(mu))  # Array of zeros, one for each mu
    for k in range(len(mu)): # Iterate over the components
        for i in range(x.shape[0]):
            hat_mu[k] += prob[i][k] * x[i]
        hat_mu[k] /= np.sum([p[k] for p in prob])  # Sum probabilities for component k
    return hat_mu

def update_sigma(x, mu, sigma):
    """
    Updates the standard deviations of the Gaussian components based on the
    responsibilities.

    Args:
        x (np.ndarray): Data points (1D array).
        mu (list): List of current means for each Gaussian component.
        sigma (list): List of current standard deviations for each Gaussian component.

    Returns:
        np.ndarray: Updated standard deviations for each Gaussian component.
    """
    prob = pr_single_normalized(mu, sigma, x)
    hat_sigma = np.zeros(len(sigma)) # Array of zeros, one for each sigma
    for k in range(len(sigma)):  # Iterate over the components
        for i in range(x.shape[0]):
            hat_sigma[k] += prob[i][k] * (x[i] - mu[k])**2
        hat_sigma[k] /= np.sum([p[k] for p in prob]) # Sum probabilities for component k
    return hat_sigma


# Initialize parameters for the EM algorithm
mu_old = [random.uniform(-2, 2), random.uniform(0, 4)]  # Initial means
sigma_old = [0.3, 0.3]  # Initial standard deviations
NbIter = 10  # Number of iterations

# Learning procedure (optimization)
for iter in range(1, NbIter + 1):  # Corrected iteration range
    hat_mu = update_mu(data, mu_old, sigma_old)
    hat_sigma = update_sigma(data, mu_old, sigma_old)
    print('iter', iter)
    print('updated mu = ', hat_mu)
    print('updated sigma = ', hat_sigma)
    mu_old = hat_mu
    sigma_old = hat_sigma + 1e-13 # Add a small constant to prevent zero sigma

"""
# 2D Gaussian Mixture Model (GMM) Implementation

Implement the EM algorithm for a 2D GMM with two components.
"""

# First simulated data set
plt.figure() # Ensures plots are separate
plt.title("Two informative features, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42) # Added random state for reproducibility
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
plt.show() # Ensures plot displays

# Second simulated data set
plt.figure()
plt.title("Three blobs", fontsize='small')
X2, Y2 = make_blobs(n_samples=200, n_features=2, centers=3, random_state=42)  # Added random_state
plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2, s=25, edgecolor='k')
plt.show()

# Third simulated data set
plt.figure()
plt.title("Non-linearly separated data sets", fontsize='small')
X3, Y3 = make_moons(n_samples=200, shuffle=True, noise=None, random_state=42) # Added random_state
plt.scatter(X3[:, 0], X3[:, 1], marker='o', c=Y3, s=25, edgecolor='k')
plt.show()

def pr_single_comp2d(mu, sigma, x):
  """
  Calculates the probability density for each data point x under a single
  2D Gaussian component.

  Args:
      mu (np.ndarray): Mean vector of the Gaussian component (2D).
      sigma (np.ndarray): Covariance matrix of the Gaussian component (2x2).
      x (np.ndarray): Data points (2D array).

  Returns:
      list: List of probability densities for each data point.
  """
  prob = []
  for i in range(0, x.shape[0]):
    try:
      prob1 = np.exp(-0.5 * ((x[i] - mu).reshape(1, -1) @ np.linalg.inv(sigma) @ (x[i] - mu).reshape(-1, 1))) / (np.sqrt(np.linalg.det(sigma))) # Reshape for matrix multiplication
      prob.append(prob1[0][0]) # Append the scalar value
    except np.linalg.LinAlgError:
        print("Singular matrix encountered.  Check covariance matrix initialization or add regularization.")
        return None # Or raise the exception, depending on your needs
  return prob

def pr_single_normalized2d(mu, sigma, x):
    """
    Calculates normalized probabilities for each data point belonging to each
    2D Gaussian component.

    Args:
        mu (list): List of mean vectors for each Gaussian component.
        sigma (list): List of covariance matrices for each Gaussian component.
        x (np.ndarray): Data points (2D array).

    Returns:
        list: List of normalized probabilities for each data point belonging to
              each Gaussian component.
    """
    unnorm_prob = [pr_single_comp2d(mu[k], sigma[k], x) for k in range(len(mu))]
    # Check for None values (singular matrix)
    if any(p is None for p in unnorm_prob):
        return None

    normalization = np.sum(unnorm_prob, axis=0)
    prob = []
    for i in range(x.shape[0]):
        component_probs = [unnorm_prob[k][i] / normalization[i] for k in range(len(mu))]
        prob.append(component_probs)
    return prob

def update_mu2d(x, mu, sigma):
  """
  Updates the mean vectors of the 2D Gaussian components based on the
  responsibilities.

  Args:
      x (np.ndarray): Data points (2D array).
      mu (list): List of current mean vectors for each Gaussian component.
      sigma (list): List of current covariance matrices for each Gaussian component.

  Returns:
      np.ndarray: Updated mean vectors for each Gaussian component.
  """
  prob = pr_single_normalized2d(mu, sigma, x)
  if prob is None:
      return None

  hat_mu = np.zeros((len(mu), 2)) # Initialize with correct shape

  for k in range(len(mu)): # Iterate over the components
      for i in range(x.shape[0]):  # Iterate over data points
          hat_mu[k] += prob[i][k] * x[i]

      hat_mu[k] /= np.sum([p[k] for p in prob])

  return hat_mu

def update_sigma2d(x, mu, sigma):
  """
  Updates the covariance matrices of the 2D Gaussian components based on the
  responsibilities.

  Args:
      x (np.ndarray): Data points (2D array).
      mu (list): List of current mean vectors for each Gaussian component.
      sigma (list): List of current covariance matrices for each Gaussian component.
  Returns:
      np.ndarray: Updated covariance matrices for each Gaussian component.
  """
  prob = pr_single_normalized2d(mu, sigma, x)
  if prob is None:
      return None
  hat_sigma = np.zeros((len(mu), 2, 2))

  for k in range(len(mu)):  # Iterate over the components
    for i in range(x.shape[0]):  # Iterate over data points
        diff = (x[i] - mu[k]).reshape(-1, 1) # Reshape for correct broadcasting

        hat_sigma[k] += prob[i][k] * (diff @ diff.T)
    hat_sigma[k] /= np.sum([p[k] for p in prob])

  return hat_sigma


# Initialize parameters for the 2D EM algorithm
mu_old = np.random.uniform(-2, 2, size=(2, 2))  # Initial mean vectors
sigma_old = [[[0.5, 0], [0, 0.75]], [[1, 0], [0, 1]]]  # Initial covariance matrices
NbIter = 10  # Number of iterations

# Learning procedure (optimization)
for iter in range(1, NbIter + 1):
  hat_mu = update_mu2d(X2, mu_old, sigma_old)
  hat_sigma = update_sigma2d(X2, mu_old, sigma_old)
  if hat_mu is None or hat_sigma is None:
      print("Terminating EM due to singular covariance matrix.")
      break

  print('iter', iter)
  print('updated mu = ', hat_mu)
  print('updated sigma = ', hat_sigma)
  mu_old = hat_mu
  sigma_old = hat_sigma + np.eye(2)* 1e-13 # Add a small constant to prevent singularity, adding to diagonal.

"""
# Model Evaluation using scikit-learn's GaussianMixture

Use scikit-learn's GaussianMixture to find the best GMM for each dataset and
evaluate the clustering performance using homogeneity score.
"""

def bestGM(X):
  """
  Finds the best Gaussian Mixture Model (GMM) for a given dataset using
  Bayesian Information Criterion (BIC).

  Args:
      X (np.ndarray): The input data.

  Returns:
      np.ndarray: Predicted cluster labels for the input data.
  """
  lowest_bic = np.infty
  bic = []
  n_components_range = range(1, 5)  # Test different numbers of components
  cv_types = ['spherical', 'tied', 'diag', 'full']  # Test different covariance types

  for cv_type in cv_types:
    for n_components in n_components_range:
      # Fit a Gaussian mixture with EM
      gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, random_state=42)
      gmm.fit(X)
      bic.append(gmm.bic(X))
      if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

  y_predicted = best_gmm.predict(X)
  return y_predicted

# Evaluate on simulated datasets
print("homogeneity Score for X1 :", metrics.homogeneity_score(Y1, bestGM(X1))) # Fixed order of arguments
print("homogeneity Score for X2 :", metrics.homogeneity_score(Y2, bestGM(X2)))
print("homogeneity Score for X3 :", metrics.homogeneity_score(Y3, bestGM(X3)))

"""
# Real World Datasets Evaluation

Evaluate the GMM clustering performance on real-world datasets.
"""

# Load real-world datasets
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
breast_data=pd.read_csv(url,header=None)

breast_cancer_X=breast_data.drop([0,1],axis=1)
breast_cancer_Y=breast_data[1]

url='https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls'
mice_data = pd.read_excel(url,'Hoja1')

mice_data_X=mice_data.drop(['MouseID','Genotype','Treatment','Behavior','class'],axis=1)
mice_data_Y=mice_data['class']

# Fill missing values
mice_data_X=mice_data_X.fillna(value=mice_data_X.mean(axis=0).to_dict())

# Evaluate on real-world datasets
print("homogeneity Score for breast_cancer :", metrics.homogeneity_score(breast_cancer_Y, bestGM(breast_cancer_X)))  #Fixed order of arguments.
print("homogeneity Score for mice_data :", metrics.homogeneity_score(mice_data_Y, bestGM(mice_data_X)))
