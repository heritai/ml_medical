# -*- coding: utf-8 -*-
"""
ml_heart_disease.py

This code snippet explores various machine learning techniques for heart disease
prediction and clustering.
"""

import pandas as pd
import numpy as np
import scipy.stats as sp  # For mode calculation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, homogeneity_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.svm import LinearSVC
from sklearn import linear_model  # Redundant import, already imported Lasso
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D PCA plots
from sklearn.decomposition import PCA
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.manifold.t_sne import TSNE

"""
# Data Loading and Preprocessing

Load the heart disease dataset, preprocess it by handling missing values,
creating dummy variables, and standardizing the data.
"""

# Load the dataset
try:
    heart_data = pd.read_csv('data/heart.csv', na_values='?')
except FileNotFoundError:
    print("Error: Heart disease dataset file not found. Please ensure 'data/heart.csv' is in the correct directory.")
    raise

# Display the first few rows of the dataset
heart_data.head()

# Separate features and target variable
heart_X = heart_data.drop('target', axis=1)
heart_Y = heart_data['target']

"""
# Preprocessing

1.  Handle missing values by imputing with the mean for continuous variables
    ('ca') and mode for categorical variables ('thal').
2.  Create dummy variables for categorical features.
3.  Standardize the data using StandardScaler.
"""

# Impute missing values
heart_X1 = heart_X.fillna(value={'ca': np.mean(heart_X['ca']), 'thal': sp.mode(heart_X['thal'])[0][0]}, axis=0)
print("Missing values after imputation:", np.sum(heart_X1.isna().sum()))  # Verify no missing values

# Create dummy variables
category_variables = ['sex', 'cp', 'fbs', 'restecg', 'slope', 'exang', 'thal']
dummyHeartX = pd.get_dummies(heart_X1, columns=category_variables, prefix=category_variables, drop_first=True)

# Standardize the data
scaler = StandardScaler()
heart_X2 = scaler.fit_transform(dummyHeartX)

"""
# Visualization

Apply PCA for dimensionality reduction and visualize the data in 2D and 3D.
"""

# 2D PCA visualization
plt.figure(figsize=(8, 6))
X_reduced = PCA(n_components=2, random_state=42).fit_transform(heart_X2)  # Added random_state
r = 2  # Margin for plot limits
x_min, x_max = X_reduced[:, 0].min() - r, X_reduced[:, 0].max() + r
y_min, y_max = X_reduced[:, 1].min() - r, X_reduced[:, 1].max() + r

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=heart_Y, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('First Component')
plt.ylabel('Second Component')
plt.title('2D PCA Visualization')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# 3D PCA visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')  # Use add_subplot to create 3D axes
X_reduced = PCA(n_components=3, random_state=42).fit_transform(heart_X2)  # Added random_state
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=heart_Y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()

"""
# Classification

Apply different classification models and evaluate their performance using
cross-validation. Also, perform hyperparameter tuning for Gradient Boosting
Classifier using GridSearchCV.
"""

# Feature Importance from Gradient Boosting Classifier
XGBclf = GradientBoostingClassifier(random_state=42)  # Added random_state
XGBclf.fit(heart_X2, heart_Y)

plt.figure(figsize=(16, 6))
plt.bar(dummyHeartX.columns, XGBclf.feature_importances_)
plt.xlabel("Variables")
plt.ylabel("Importance")
plt.title("Feature Importance from Gradient Boosting Classifier")
plt.xticks(rotation=90) # Rotate x-axis labels for better readability
plt.show()

# Hyperparameter Tuning for Gradient Boosting Classifier
XGBparameters = {'n_estimators': [100, 150, 200], 'max_depth': [2, 3, 4]}
XGBclf = GradientBoostingClassifier(random_state=42)  # Added random_state
gridClf = GridSearchCV(XGBclf, XGBparameters, n_jobs=-1, cv=6)
gridClf.fit(heart_X2, heart_Y)

# Print best parameters and score from GridSearchCV
print("Best parameters from GridSearchCV:", gridClf.best_params_)
print("Best score from GridSearchCV:", gridClf.best_score_)

"""
## Classification Models and Cross-Validation

Evaluate the performance of different classification models using cross-validation.
1.  Gradient Boosting Classifier (XGBoost)
2.  Random Forest
3.  MLP (Multi-Layer Perceptron)
4.  Lasso Regression
5.  Linear SVC (Support Vector Classifier)
6.  Elastic Net Regression
"""

# Gradient Boosting Classifier (XGBoost)
XGBclf = GradientBoostingClassifier(max_depth=1, n_estimators=60, random_state=42)  # Added random_state
print("Gradient Boosting Classifier 10-fold CV accuracy:", np.mean(cross_val_score(XGBclf, heart_X2, heart_Y, cv=10)))

# Random Forest
RFclf = RandomForestClassifier(random_state=42)  # Added random_state
print("Random Forest 10-fold CV accuracy:", np.mean(cross_val_score(RFclf, heart_X2, heart_Y, cv=10)))

# MLP (Multi-Layer Perceptron)
NNclf = MLPClassifier(activation='tanh', batch_size=50, max_iter=2000, hidden_layer_sizes=(20, 10), random_state=42)  # Added random_state
print("MLP 10-fold CV accuracy:", np.mean(cross_val_score(NNclf, heart_X2, heart_Y, cv=10)))

# Lasso Regression
LassoReg = linear_model.Lasso(alpha=0.001, random_state=42)  # Added random_state
LassoReg.fit(heart_X2, heart_Y)
print("Lasso Regression: Number of selected features:", LassoReg.sparse_coef_.getnnz())
print("Lasso Regression 10-fold CV accuracy:", np.mean(cross_val_score(LassoReg, heart_X2, heart_Y, cv=10)))

# Linear SVC
LinSVC = LinearSVC(C=1, penalty="l2", dual=False, random_state=42)  # Added random_state
LinSVC.fit(heart_X2, heart_Y)
print("Linear SVC 10-fold CV accuracy:", np.mean(cross_val_score(LinSVC, heart_X2, heart_Y, cv=10)))
print("Linear SVC: Number of selected features:", np.sum(LinSVC.coef_ != 0))

# Elastic Net Regression
ElasNet = ElasticNet(alpha=0.01, l1_ratio=0.8, random_state=42)  # Added random_state
ElasNet.fit(heart_X2, heart_Y)
print("Elastic Net: Number of selected features:", ElasNet.sparse_coef_.getnnz())
print("Elastic Net 10-fold CV accuracy:", np.mean(cross_val_score(ElasNet, heart_X2, heart_Y, cv=10)))

"""
# Clustering and Visualization

Apply different clustering algorithms and visualize the clustering regions in 2D
using t-SNE for dimensionality reduction.
"""

# Plotting the clustering region in 2D
def ClassificationRegion(preds, X, y):
    """
    Visualizes the classification regions using t-SNE for dimensionality reduction
    and KNeighborsClassifier for approximating Voronoi tesselation.

    Args:
        preds (np.ndarray): Predicted labels from the clustering algorithm.
        X (np.ndarray): The input data (features).
        y (np.ndarray): The target variable (labels).
    """
    X_Train_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)  # Added random_state

    y_predicted = preds
    y_predicted = np.round(y_predicted)

    # Create meshgrid
    resolution = 100  # 100x100 background pixels
    X2d_xmin, X2d_xmax = np.min(X_Train_embedded[:, 0]), np.max(X_Train_embedded[:, 0])
    X2d_ymin, X2d_ymax = np.min(X_Train_embedded[:, 1]), np.max(X_Train_embedded[:, 1])
    xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

    # Approximate Voronoi tesselation on resolution x resolution grid using 1-NN
    background_model = KNeighborsClassifier(n_neighbors=1).fit(X_Train_embedded, y_predicted)
    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
    voronoiBackground = voronoiBackground.reshape((resolution, resolution))

    # plot
    plt.contourf(xx, yy, voronoiBackground)
    plt.scatter(X_Train_embedded[:, 0], X_Train_embedded[:, 1], c=y, cmap='viridis') # Use colormap directly in scatter
    plt.xlabel("t-SNE Component 1") # Changed to be more descriptive
    plt.ylabel("t-SNE Component 2") # Changed to be more descriptive
    plt.title("Classification Region Visualization") # More descriptive title.
    plt.show()

nClust = len(np.unique(heart_Y))

"""
## Clustering Algorithms

Apply different clustering algorithms to the preprocessed heart disease dataset.
1.  K-means
2.  Spectral Clustering
3.  Gaussian Mixture Model (GMM)
4.  Agglomerative Clustering
"""

# K-means
km = KMeans(n_clusters=nClust, init='k-means++', max_iter=100, n_init=1, random_state=42)  # Added random_state
km.fit(heart_X2)
print("K-means: Homogeneity score", homogeneity_score(heart_Y, km.labels_))
ClassificationRegion(km.predict(heart_X2), heart_X2, heart_Y)

# Spectral Clustering
spectral = SpectralClustering(n_clusters=nClust, eigen_solver='arpack', affinity="nearest_neighbors", random_state=42)  # Added random_state
spectral.fit(heart_X2)
print("Spectral Clustering: Homogeneity score", homogeneity_score(heart_Y, spectral.labels_))
ClassificationRegion(spectral.labels_, heart_X2, heart_Y)

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=nClust, covariance_type='spherical', random_state=42)  # Added random_state
gmm.fit(heart_X2)
print("Gaussian Mixture: Homogeneity score", homogeneity_score(heart_Y, gmm.predict(heart_X2)))
ClassificationRegion(gmm.predict(heart_X2), heart_X2, heart_Y)

# Agglomerative Clustering
agglo = AgglomerativeClustering(linkage='ward', n_clusters=nClust)
agglo.fit(heart_X2)
print("Agglomerative Clustering: Homogeneity Score:", homogeneity_score(heart_Y, agglo.labels_))
ClassificationRegion(agglo.labels_, heart_X2, heart_Y)
