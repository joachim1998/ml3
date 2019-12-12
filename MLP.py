# -*- coding: utf-8 -*-
"""
Test the SVC estimator for different values of C, degree and different kind of
kernel
"""

# -*- coding: utf-8 -*-

from real_submission import load_from_csv
from real_submission import measure_time
from real_submission import create_fingerprints
from real_submission import make_submission

import os
import time
import datetime
import argparse
from contextlib import contextmanager

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.preprocessing import scale
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier


from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

from rdkit import Chem
from rdkit.Chem import AllChem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make a toy submission")

    path_ordi = "/home/joachim/Documents/cours_2019_2020/ml/projet3/"

    parser.add_argument("--ls", default=path_ordi + "training_set.csv",
                        help="Path to the learning set as CSV file")
    parser.add_argument("--ts", default=path_ordi + "test_set.csv",
                        help="Path to the test set as CSV file")
    parser.add_argument("--dt", action="store_true", default=True,
                        help="Use a decision tree classifier (by default, "
                             "make a random prediction)")

    args = parser.parse_args()

    # Load training data
    LS = load_from_csv(args.ls)
    # Load test data
    TS = load_from_csv(args.ts)

    # -------------------------- Model --------------------------- #

    # LEARNING
    # Create fingerprint features and output
    with measure_time("Creating fingerprint"):
        X_LS = create_fingerprints(LS["SMILES"].values)
    y_LS = LS["ACTIVE"].values

    # Set the parameters by cross-validation
    tuned_parameters = [{'hidden_layer_sizes': [(10,),(50,),(100,),(200,),(300,)], 'activation': ['logistic','tanh','relu'],
    					 'learning_rate': ['constant', 'adaptive'], 'learning_rate_init': [0.0005,0.001,0.003]
                        }]
    scores = ['roc_auc']


    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        # Chercher GridSearchCV dans documentation
        clf = GridSearchCV(MLPClassifier(random_state = 42, alpha=1e-5, early_stopping=True), tuned_parameters, cv=2, scoring='%s' % score, n_jobs=-1, verbose=10)
        clf.fit(X_LS, y_LS)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()