# Machine Learning for Clinical Data

This repository contains a collection of Python scripts that demonstrate various machine learning techniques applied to clinical datasets. The primary goal is to explore and implement both classification and clustering methods for analyzing and predicting outcomes in different medical contexts.

## Overview

This repository covers the following topics:

*   **Feature Selection:** Techniques for identifying the most relevant features in a dataset to improve model performance and interpretability.
*   **Dimension Reduction:** Methods like Principal Component Analysis (PCA) to reduce the dimensionality of data while preserving essential information.
*   **Classification:** Implementation and evaluation of various classification algorithms for predicting clinical outcomes.
*   **Clustering:** Application of clustering techniques to identify patterns and group similar patients based on their characteristics.
*   **Bayesian Networks:** Construction and analysis of Bayesian Networks for modeling probabilistic relationships between variables.
*   **Neural Networks:** Implementation and evaluation of both scikit-learn and Keras based Neural Networks for various classification tasks.
*   **Cross-Validation:** Techniques to evaluate the performance of machine learning models on the training data using several CV scores.

## Code and Project Summary

| Code/Project                   | Model                                      | Data                                                                                                                                                                                                             | Application | Packages                                      |
| :----------------------------- | :----------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :-------------------------------------------- |
| Diabetes Remission Prediction    | Random Forest, Decision Tree               | [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)                                                                                                    | Medical     | scikit-learn, pandas                          |
| Clustering Comparison Metrics | K-means, Hierarchical clustering, Spectral clustering | [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), [Mice Protein Data Set](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) | Medical     | scikit-learn, pandas                          |
| EM Algorithm                 | Gaussian Mixtures                          | [Breast Cancer Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), [Mice Protein Dataset](https://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression)         | Medical     | scikit-learn, pandas                          |
| Model/Feature Selection        | Lasso Regression, Linear SVC, Elastic Net     | [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), [Gene Expression Dataset](https://www.kaggle.com/datasets/crawford/gene-expression)            | Medical     | scikit-learn, pandas                          |
| Cross Validation/Class Regions | Logistic Regression                          | [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), [Gene Expression Dataset](https://www.kaggle.com/datasets/crawford/gene-expression)            | Medical     | scikit-learn, pandas                          |
| Dimension Reduction            | PCA, LDA                                   | [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), [Gene Expression Dataset](https://www.kaggle.com/datasets/crawford/gene-expression)            | Medical     | scikit-learn, pandas                          |
| Bayesian Networks              | Bayesian Networks                          | Dietary data of obese patients ([MICRO-Obes](https://pubmed.ncbi.nlm.nih.gov/25330000/))                                                                                                               | Medical     | pyAgrum, pandas                             |
| Neural Networks                | MLP                                          | Dietary data of obese patients ([MICRO-Obes](https://pubmed.ncbi.nlm.nih.gov/25330000/)), [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), [Gene Expression Dataset](https://www.kaggle.com/datasets/crawford/gene-expression) | Medical     | Keras, Tensorflow, scikit-learn                 |
| Heart Disease Prediction       | Gaussian Mixtures, Gradient Boost, PCA       | [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)                                                                                                                               | Medical     | scikit-learn, scipy                           |


## Datasets

The following datasets are used in this repository:

*   **Heart Disease Dataset (`ml_heart_disease.py`):**
    *   Source: Cleveland dataset from the UCI Machine Learning Repository ([https://archive.ics.uci.edu/ml/datasets/Heart+Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)).
    *   Description: Contains clinical data for predicting the presence of heart disease.
    *   File: `data/heart.csv` (This file needs to be downloaded separately and placed in the `data/` folder.)

*   **Golub Gene Expression Dataset (`feature_model_selection.py`, `dimension_redcut.py`, `neural_nets.py`):**
    *   Description: Gene expression data used for leukemia classification.
    *   Files: `data/Golub_X` (observations), `data/Golub_y` (classes) (These files need to be downloaded separately and placed in the `data/` folder.)

*   **Breast Cancer Wisconsin (Diagnostic) Dataset (`feature_model_selection.py`, `dimension_redcut.py`, `neural_nets.py`, `cross_validation.py`):**
    *   Description: Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
    *   File: `data/Breast.txt` (This file needs to be downloaded separately and placed in the `data/` folder.)

*   **SPLEX Dataset (`neural_nets.py`, `cross_validation.py`, `bayes_nets.py`):**
    *   Description: Host and environmental data of obese patients.  Includes environmental (`SPLEX_env.txt`), host (`SPLEX_host.txt`), and microbial (`SPLEX_micro.txt`) data, along with class labels (`classes.csv`).
    *   Files: `data/SPLEX_env.txt`, `data/SPLEX_host.txt`, `data/SPLEX_micro.txt`, `data/classes.csv` (These files need to be downloaded separately and placed in the `data/` folder.)

*   **Mouse Protein Expression Dataset (`cross_validation.py`):**
    *   Description: Data related to protein expression levels in the cerebral cortex of mice.
    *   File: The dataset should be downloaded from : [https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls](https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls)
    *   save the file as Data_Cortex_Nuclear.xls in the `/data` directory.

*   **Dynamic Data for Diabetes Remission (`bayes_nets.py`):**
    *   Description: Dynamic data with HbA1C (glycated hemoglobin), Gly (glycemia), Poids (weight of patients), and Status (remission, non-remission, or partial remission) for time 0, 1 and 5 years after the surgery.
    *   File: `data/dynamic.txt` (This file needs to be downloaded separately and placed in the `data/` folder.)

**Important Note:**  Due to licensing restrictions or file size limitations, some dataset files are not included directly in this repository. You will need to download them separately and place them in the `data/` directory for the scripts to run correctly.

## Scripts and Notebooks

The repository contains the following key files:

*   `ml_heart_disease.py`: Explores various machine learning techniques for heart disease prediction and clustering.

*   `feature_model_selection.py`: Demonstrates feature selection techniques using different models.

*   `dimension_redcut.py`: Explores dimension reduction techniques like PCA, Kernel PCA, and LDA.

*   `cross_validation.py`: Implements cross-validation for model evaluation using both scikit-learn and custom logistic regression.

*   `neural_nets.py`: Explores neural network models using scikit-learn and Keras.

*   `bayes_nets.py`: Focuses on Bayesian Networks for clinical data analysis, including dynamic Bayesian networks.

*   `em_algorithm.py`: Implements and applies the Expectation-Maximization algorithm for Gaussian Mixture Models.

*   `clustering_comparison.py`: compares several clustering algorithms using the sklearn library.

## Dependencies

To run the code in this repository, you will need to install the following Python libraries:

```
pip install pandas numpy scikit-learn matplotlib pyAgrum tensorflow
```

For more complex installation guidance, see the instructions:

* **PyAgrum**: Consult the PyAgrum documentation for detailed installation instructions: http://agrum.gitlab.io/pages/pyagrum.html

* **Graphviz**: You will also need to install Graphviz system-wide to render Bayesian networks and decision trees. On Ubuntu/Debian: `sudo apt-get install graphviz`. On macOS using Homebrew: `brew install graphviz`.

## Usage

* Clone the Repository:

```
git clone [repository URL]
cd [repository directory]
```


* Install Dependencies:

```
pip install -r requirements.txt
```

* Download Datasets:

Download the necessary dataset files and place them in the `data/` directory.

make sure to rename Data_Cortex_Nuclear.xls to Data_Cortex_Nuclear.xls

* Run the Scripts:

You can run the Python scripts directly from the command line:

```
python ml_heart_disease.py
```


## Notes

The code has been designed to be modular and easy to understand.

Feel free to experiment with different parameters, models, and datasets.

Contributions to this repository are welcome!

License

This project is licensed under the [Specify License - e.g., MIT License] - see the LICENSE file for details.
