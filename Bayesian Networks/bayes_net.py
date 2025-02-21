# -*- coding: utf-8 -*-
"""
bayes_net.py

The goal of this project is to gain skills in using the PyAgrum library and
to learn how to construct Bayesian networks and dynamic Bayesian networks.
"""

import pandas as pd
import numpy as np
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.bn2graph as bnPlot  # Unused import
import pyAgrum.lib.dynamicBN as gdyn  # Unused import

"""
# Data

*   Prediction of Type 2 Diabetes Remission: dynamic data
*   Host and Environmental Data of Obese Patients
"""

"""
# Analysis

## Creating a network using PyAgrum.

We would like to model the problem of type 2 diabetes remission after a
gastric by-pass surgery which can be represented by the following graph
(note, that the problem is simplified extremely!).

Create an empty BN network and add nodes
"""

bn = gum.BayesNet('Diabetes')
print(bn)

nodes = [('g', 'Glycemia>6.5?', 2),
         ('i', 'Insulin taken?', 2),
         ('o', 'Other drugs taken?', 2),
         ('r', 'Remission', 2)]
g, i, o, r = [bn.add(gum.LabelizedVariable(k[0], k[1], k[2])) for k in nodes]

print(g, i, o, r)
print(bn)

"""
Create the arcs and the probability tables
"""

for link in [(g, i), (g, o), (i, r), (o, r)]:
    bn.addArc(*link)
print(bn)

bn.cpt(g).fillWith([0.5, 0.5])

bn.cpt(o)[{'g': 1}] = [0.25, 0.75]
bn.cpt(o)[{'g': 0}] = [0.7, 0.3]

bn.cpt(i)[{'g': 1}] = [0.1, 0.9]
bn.cpt(i)[{'g': 0}] = [0.9, 0.1]

bn.cpt(r)[{'i': 1, 'o': 1}] = [0.9, 0.1]
bn.cpt(r)[{'i': 0, 'o': 1}] = [0.6, 0.4]
bn.cpt(r)[{'i': 1, 'o': 0}] = [0.3, 0.7]
bn.cpt(r)[{'i': 0, 'o': 0}] = [0.1, 0.9]

"""
Visualize the graph
"""

bn

"""
Perform inference (withLazyPropagation())
"""

ie = gum.LazyPropagation(bn)

ie.makeInference()
ie.posterior(r)

"""
Perform inference with evidence. What is the probability to get the remission
if the glycemia level is less than 6.5 and no drugs are taken?
"""

ie.setEvidence({'g': 0, 'o': 0})
ie.makeInference()
ie.posterior(r)

"""
What is the probability to get the remission if the glycemia level is bigger
than 6.5 and insulin is prescribed?
"""

ie.setEvidence({'g': 1, 'i': 1})
ie.makeInference()
ie.posterior(r)

"""
### Construct Bayesian networks from real data.

The data are in SPLEXenv.txt and SPLEXhost.txt files. Construct one network
for the environmental variables, one for the host variables, and one with both
environmental and host data.

Load and discretize the data (the Bayesian networks are learned from discrete
data only) To discretize the data, each column into 5 bins
"""

try:
    SPLEXhost = pd.read_table('SPLEX_host.txt', sep=' ')
    SPLEXenv = pd.read_table('SPLEX_env.txt', sep=' ')

    concatenated_data = pd.concat([SPLEXhost, SPLEXenv], axis=1)
except FileNotFoundError:
    print("Error: SPLEX dataset files not found. Please ensure 'SPLEX_host.txt' and 'SPLEX_env.txt' are in the correct directory.")
    raise

def discr_save(data, save_file_name):
    """
    Discretizes the data by binning each column into 5 bins and saves the
    discretized data to a CSV file.

    Args:
        data (pd.DataFrame): The input data.
        save_file_name (str): The name of the file to save the discretized data to.
    """
    l = []
    for col in data.columns.values:
        bins = np.linspace(min(data[col]), max(data[col]), 5)
        l.append(pd.DataFrame(np.digitize(data[col], bins), columns=[col]))
    discr_data = pd.concat(l, join='outer', axis=1)
    discr_data.to_csv(save_file_name + ".csv", index=False)

discr_save(SPLEXhost, 'discr_SPLEXhost')
discr_save(SPLEXenv, 'discr_SPLEXenv')
discr_save(concatenated_data, 'discr_concatenated_data')

"""
Run a learner to learn a networks (test useLocalSearchWithTabuList() and
useGreedyHillClimbing() functions)
"""

def BNlearnWrapp(filePath, method):
    """
    Learns a Bayesian network from a CSV file using either local search with
    tabu list or greedy hill climbing.

    Args:
        filePath (str): The path to the CSV file containing the data.
        method (str): The learning method to use ('LocalSearch' or 'HillClimb').

    Returns:
        gum.BayesNet: The learned Bayesian network.
    """
    learner = gum.BNLearner(filePath)
    if method == 'LocalSearch':
        learner.useLocalSearchWithTabuList()
    else:
        learner.useGreedyHillClimbing()
    return learner.learnBN()


bn_SPLEXhost_localSearch = BNlearnWrapp("discr_SPLEXhost.csv", 'LocalSearch')
gnb.showBN(bn_SPLEXhost_localSearch)

bn_SPLEXhost_HillClimb = BNlearnWrapp("discr_SPLEXhost.csv", 'HillClimb')
gnb.showBN(bn_SPLEXhost_HillClimb)

bn_SPLEXenv_LocalSearch = BNlearnWrapp("discr_SPLEXenv.csv", 'LocalSearch')
gnb.showBN(bn_SPLEXenv_LocalSearch)

bn_SPLEXenv_HillClimb = BNlearnWrapp("discr_SPLEXenv.csv", 'HillClimb')
gnb.showBN(bn_SPLEXenv_HillClimb)

bn_concatenated_data_LocalSearch = BNlearnWrapp("discr_concatenated_data.csv", 'LocalSearch')
gnb.showBN(bn_concatenated_data_LocalSearch)

bn_concatenated_data_HillClimb = BNlearnWrapp("discr_concatenated_data.csv", 'HillClimb')
gnb.showBN(bn_concatenated_data_HillClimb)

"""
Save the obtained networks
"""

gum.saveBN(bn_SPLEXhost_localSearch, 'bn_SPLEXhost_localSearch.bif')
gum.saveBN(bn_SPLEXhost_HillClimb, 'bn_SPLEXhost_HillClimb.bif')
gum.saveBN(bn_SPLEXenv_LocalSearch, 'bn_SPLEXenv_LocalSearch.bif')
gum.saveBN(bn_SPLEXenv_HillClimb, 'bn_SPLEXenv_HillClimb.bif')
gum.saveBN(bn_concatenated_data_LocalSearch, 'bn_concatenated_data_LocalSearch.bif')
gum.saveBN(bn_concatenated_data_HillClimb, 'bn_concatenated_data_HillClimb.bif')

"""
Are the networks learned with useLocalSearchWithTabuList() and
useGreedyHillClimbing() similar?

No!

### Dynamic Bayesian networks

Load data from dynamic.txt. In this file, you have HbA1C (glycated hemoglobin),
Gly (glycemia), Poids (weight of patients), and Status (remission,
non-remission, or partial remission) for time 0, 1 and 5 years after the
surgery. Construct a dynamic network to explore temporal dependencies in the
data.
"""

try:
    dynamic_data = pd.read_table('dynamic.txt', sep=' ')
except FileNotFoundError:
    print("Error: dynamic dataset file not found. Please ensure 'dynamic.txt' is in the correct directory.")
    raise


discr_save(dynamic_data, 'dynamic_discr')

"""
The first step is to learn a Bayesian network bn_dynamic as in the previous task
"""

learner = gum.BNLearner("dynamic_discr.csv")
learner.useLocalSearchWithTabuList()
bn_dynamic = learner.learnBN()
gnb.showBN(bn_dynamic)

"""
Visualize the network with time slices
"""

gdyn.showTimeSlices(bn_dynamic)
