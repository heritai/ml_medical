# -*- coding: utf-8 -*-
"""
ClusterComparison.py

Objective:
Learn and apply popular clustering methods in unsupervised learning and
interpret the results effectively.
"""

import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
import numpy as np
import pandas as pd

"""
# Simulated Data

Create three simulated datasets for clustering:
1.  Two informative features, one cluster per class (classification-like data)
2.  Three blobs (well-separated clusters)
3.  Non-linearly separated data sets (moons shape)
"""

# First simulated data set
plt.figure()  # Create a new figure for each plot
plt.title("Two informative features, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1, random_state=42)  # Added random_state
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1,s=25, edgecolor='k')
plt.show()

# Second simulated data set
plt.figure()
plt.title("Three blobs", fontsize='small')
X2, Y2 = make_blobs(n_samples=200, n_features=2, centers=3, random_state=42)  # Added random_state
plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2, s=25, edgecolor='k')
plt.show()

# Third simulated data set
plt.figure()
plt.title("Non-linearly separated data sets", fontsize='small')
X3, Y3 = make_moons(n_samples=200, shuffle=True, noise=None, random_state=42)  # Added random_state
plt.scatter(X3[:, 0], X3[:, 1], marker='o', c=Y3, s=25, edgecolor='k')
plt.show()

"""
# Clustering Algorithms and Plotting Functions

Define functions to apply KMeans and Agglomerative clustering to the simulated
datasets and visualize the results.
"""

def kmFitPlot(X, nbClust):
  """
  Applies KMeans clustering and plots the results.

  Args:
      X (np.ndarray): The input data.
      nbClust (int): The number of clusters.
  """
  km = KMeans(n_clusters=nbClust, init='k-means++', max_iter=100, n_init=1, random_state=42) # Added random_state
  km.fit(X)
  plt.figure() # Create a new figure
  plt.title(f"KMeans with {nbClust} clusters")
  plt.scatter(X[:, 0], X[:, 1], s=10, c=km.labels_)
  plt.show()

# Apply KMeans to the three datasets
kmFitPlot(X1, 2)
kmFitPlot(X2, 3)
kmFitPlot(X3, 2)

def AgglomeratFitPlot(X, nbClust):
  """
  Applies Agglomerative clustering with different linkages and plots the results.

  Args:
      X (np.ndarray): The input data.
      nbClust (int): The number of clusters.
  """
  plt.figure(figsize=(15, 5))  # Adjust figure size for subplots
  for i, linkage in enumerate(('ward', 'average', 'complete')):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=nbClust)
    clustering.fit(X)
    plt.subplot(1, 3, i + 1)
    plt.title(linkage)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=clustering.labels_)
  plt.show()


# Apply Agglomerative clustering to the three datasets
AgglomeratFitPlot(X1, 2)
AgglomeratFitPlot(X2, 3)
AgglomeratFitPlot(X3, 2)

def spectralFitPlot(X, nbClust):
  """
  Applies Spectral clustering and plots the results.

  Args:
      X (np.ndarray): The input data.
      nbClust (int): The number of clusters.
  """
  spectral = cluster.SpectralClustering(n_clusters=nbClust, eigen_solver='arpack', affinity="nearest_neighbors", random_state=42) # Added random_state
  spectral.fit(X)
  plt.figure()
  plt.title(f"Spectral Clustering with {nbClust} clusters")
  plt.scatter(X[:, 0], X[:, 1], s=10, c=spectral.labels_)
  plt.show()

# Apply Spectral clustering to the three datasets
spectralFitPlot(X1, 2)
spectralFitPlot(X2, 3)
spectralFitPlot(X3, 2)

"""
# Clustering Evaluation Function

Define a function to evaluate the performance of different clustering algorithms
using various metrics.
"""

def clusterFunc(datalist, datasetName, metric, labels):
  """
  Evaluates the performance of KMeans, Spectral Clustering, and Agglomerative
  Clustering (with ward, average, and complete linkages) using a specified metric.

  Args:
      datalist (list): A list of datasets (e.g., [X1, X2, X3]).
      datasetName (list): A list of dataset names (e.g., ['X1', 'X2', 'X3']).
      metric (str): The name of the metric to use for evaluation
                    (e.g., 'homogeneity', 'silhouette').
      labels (list): A list of ground truth labels for each dataset (e.g., [Y1, Y2, Y3]).

  Returns:
      pd.DataFrame: A DataFrame containing the scores for each clustering algorithm
                    on each dataset.
  """

  metricF = {
      'homogeneity': metrics.homogeneity_score,
      'completeness': metrics.completeness_score,
      'v_measure': metrics.v_measure_score,
      'adjusted_rand': metrics.adjusted_rand_score,
      'silhouette': metrics.silhouette_score
  }

  F = metricF.get(metric)
  if F is None:
      raise ValueError(f"Invalid metric: {metric}.  Choose from: {list(metricF.keys())}")

  is_silhouette = metric == 'silhouette'
  results = []

  for x, y in zip(datalist, labels):
    nClust = len(np.unique(y))

    # KMeans
    km = KMeans(n_clusters=nClust, init='k-means++', max_iter=100, n_init=1, random_state=42)
    km = km.fit(x)

    # Spectral Clustering
    spectral = cluster.SpectralClustering(n_clusters=nClust, eigen_solver='arpack', affinity="nearest_neighbors", random_state=42)
    spectral.fit(x)

    # Evaluate KMeans and Spectral Clustering
    if is_silhouette:
      km_score = F(x, km.labels_)
      spectral_score = F(x, spectral.labels_)
    else:
      km_score = F(km.labels_, y)
      spectral_score = F(spectral.labels_, y)

    tup = (km_score, spectral_score)

    # Agglomerative Clustering
    for linkage in ('ward', 'average', 'complete'):
      clustering = AgglomerativeClustering(linkage=linkage, n_clusters=nClust)
      clustering.fit(x)

      if is_silhouette:
        tup = tup + (F(x, clustering.labels_),)
      else:
        tup = tup + (F(clustering.labels_, y),)
    results.append(tup)

  colnames = ['k-means', 'spectral', 'hirarchy-ward', 'hirarchy-average', 'hirarchy-complete']
  print(f"Results of {metric} score")
  return pd.DataFrame(results, columns=colnames, index=datasetName)

"""
# Evaluate Clustering on Simulated Data

Evaluate the clustering algorithms on the simulated datasets using different metrics
and print the results.
"""

# Evaluate using different metrics
print(clusterFunc([X1, X2, X3], ['X1', 'X2', 'X3'], 'homogeneity', [Y1, Y2, Y3]))
print(clusterFunc([X1, X2, X3], ['X1', 'X2', 'X3'], 'completeness', [Y1, Y2, Y3]))
print(clusterFunc([X1, X2, X3], ['X1', 'X2', 'X3'], 'v_measure', [Y1, Y2, Y3]))
print(clusterFunc([X1, X2, X3], ['X1', 'X2', 'X3'], 'adjusted_rand', [Y1, Y2, Y3]))
print(clusterFunc([X1, X2, X3], ['X1', 'X2', 'X3'], 'silhouette', [Y1, Y2, Y3]))

"""
# Real World Datasets

Load and preprocess two real-world datasets:
1.  Breast Cancer Wisconsin (Diagnostic) Data Set
2.  Mouse Protein Expression Data Set
"""

# Breast Cancer Wisconsin (Diagnostic) Data Set
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
breast_data=pd.read_csv(url,header=None)

breast_data.head()

breast_cancer_X=breast_data.drop([0,1],axis=1)
breast_cancer_Y=breast_data[1]

print("Breast Cancer Data - Missing values:", breast_data.isnull().sum().sum())
print("No missing values")

# Mouse Protein Expression Data Set
url='https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls'
mice_data = pd.read_excel(url,'Hoja1')

mice_data.head()

mice_data_X=mice_data.drop(['MouseID','Genotype','Treatment','Behavior','class'],axis=1)
mice_data_Y=mice_data['class']

print("Mouse Data - Missing values before imputation:", mice_data_X.isnull().sum().sum())

# Fill missing values
mice_data_X=mice_data_X.fillna(value=mice_data_X.mean(axis=0).to_dict())

print("Mouse Data - Missing values after imputation:", mice_data_X.isnull().sum().sum()) # Confirm no missing values

# Evaluate Clustering on Real World Data
print(clusterFunc([breast_cancer_X,mice_data_X],['breast Cancer','mice'],'homogeneity',[breast_cancer_Y,mice_data_Y]))
