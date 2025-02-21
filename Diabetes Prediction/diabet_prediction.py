# -*- coding: utf-8 -*-
"""
Diabetes Remission Prediction

Objective:
Develop practical skills in applying decision trees and random forests to
real-world biological data using the scikit-learn Python library.
"""

import pandas as pd
import graphviz
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from IPython.display import Image  # For displaying images in Jupyter Notebook

"""
Data Description:

This is a subset of the Diabetes Remission dataset
(https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).
The problem is to predict whether a diabetic patient will experience remission
or not after undergoing gastric bypass surgery.

Data Files:
- patients_data.txt: Clinical data for 200 patients. Contains the following features:
    - Age (continuous)
    - HbA1c levels (continuous)
    - Insulin usage (categorical: yes/no)
    - Usage of other anti-diabetic drugs (categorical: yes/no)
- patients_classes.txt: Class labels for the 200 patients:
    - 0 -> Diabetes Remission (DR)
    - 1 -> Non-Remission (NDR)
"""

# Data Loading (Essential - must handle file not found)
try:
    data_diabetes = pd.read_table('data/patients_data.txt', sep='\t', header=None)
    classes_diabetes = pd.read_table('data/patients_classes.txt', sep='\t', header=None)

    # Assign meaningful column names
    data_diabetes.columns = ['age', 'hba1c', 'insulin_taken', 'other_drugs_taken']

except FileNotFoundError:
    print("Error: One or more data files not found.  Please ensure 'data/patients_data.txt' and 'data/patients_classes.txt' are in the correct directory.")
    raise  # Re-raise the exception to stop execution if data is essential

"""
Machine Learning Models:

1. Decision Tree Classifier
2. Random Forest Classifier
3. DiaRem (Diabetes Remission score) - A state-of-the-art clinical score
   introduced by Still et al., 2013 [1].

[1] Still, Christopher D., et al. "Preoperative prediction of type 2 diabetes
    remission after Roux-en-Y gastric bypass surgery: a retrospective cohort study."
    The lancet Diabetes & endocrinology 2.1 (2014): 38-45.
"""

# Decision Tree Classifier
clf_dtree = tree.DecisionTreeClassifier()
clf_dtree.fit(data_diabetes, classes_diabetes)

# Visualize the Decision Tree
feature_names = ['age', 'hba1c', 'insulin_taken', 'other_drugs_taken']
classes = ['DR', 'NDR']  # DR: Diabetes Remission, NDR: Non-Remission

dot_data = tree.export_graphviz(
    clf_dtree,
    out_file=None,
    feature_names=feature_names,
    class_names=classes,
    filled=True,
    rounded=True,
    special_characters=True,
)

graph = graphviz.Source(dot_data)
graph.render("diabetes_remission")  # Saves the tree as diabetes_remission.pdf

# Random Forest Classifier
clf_rf = RandomForestClassifier(max_depth=2, random_state=0) # Setting random_state for reproducibility
clf_rf.fit(data_diabetes, classes_diabetes)

# Variable Importance Plot
plt.bar(feature_names, clf_rf.feature_importances_)
plt.xlabel("Variables")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest")  # Added a title to the plot
plt.show()

# DiaRem Clinical Score Model
class DiaRem:
    """
    Implements the DiaRem (Diabetes Remission) scoring system.
    """

    def __init__(self):
        pass

    def scoreFunc(self, v):
        """Calculates the DiaRem score for a single patient."""
        age_score = sum(v[0] >= pd.Series([40, 50, 60]))
        HbA_score = sum(v[1] >= pd.Series([6.5, 7, 9]))
        other_drug_score = v[2] * 3
        treat_insul_score = v[3] * 10
        return age_score + HbA_score + other_drug_score + treat_insul_score

    def scoreCalculater(self, X):
        """Calculates DiaRem scores for all patients."""
        scores = X.apply(self.scoreFunc, axis=1)
        return scores

    def predict(self, X):
        """Predicts diabetes remission based on the DiaRem score."""
        scores = self.scoreCalculater(X)
        labels = scores > 7  # Patient will be classified as having non-remission
        return labels.astype(int) # Convert to integers (0 or 1)


# Model Evaluation and Comparison
DiaRem_model = DiaRem()

dtree_preds = clf_dtree.predict(data_diabetes)
random_forest_preds = clf_rf.predict(data_diabetes)
DiaRem_prediction = DiaRem_model.predict(data_diabetes)

print("Decision tree accuracy score:", accuracy_score(dtree_preds, classes_diabetes))
print("Random forest accuracy score:", accuracy_score(random_forest_preds, classes_diabetes))
print("DiaRem model accuracy score:", accuracy_score(DiaRem_prediction, classes_diabetes))
