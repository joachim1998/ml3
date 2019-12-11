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


#KNN
#0.762 (+/-0.059) for {'algorithm': 'auto', 'n_neighbors': 53, 'weights': 'distance'}
#KNeighborsClassifier(n_neighbors=53, weights='distance', algorithm='auto', n_jobs=-1)

#Random forest

#RandomForestClassifier(max_depth=6, criterion='entropy', min_samples_leaf=6, min_samples_split=2, n_estimators=222)

#0.795 (+/-0.083) for {'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 12, 'n_estimators': 134}
#0.780 (+/-0.067) for {'max_depth': 21, 'n_estimators': 400}

#MLP
#0.704 (+/-0.089) for {'activation': 'tanh', 'hidden_layer_sizes': (150,), 'learning_rate': 'adaptive', 'solver': 'sgd'}

#SVC
#0.757 (+/-0.090) for {'gamma': 'auto', 'kernel': 'rbf'}
#0.769 (+/-0.087) for {'gamma': 'scale', 'kernel': 'rbf'}
#0.775 (+/-0.083) for {'C': 3.0, 'gamma': 'scale'}

#MultinomialNB
#0.695 (+/-0.115) for {'alpha': 10.413793103448276, 'fit_prior': 'False'}
#Brnouilli
#0.687 (+/-0.130) for {'alpha': 0.01, 'fit_prior': 'True'}
#  important = [
#        4,
#        8,
#        15,
#        29,
#        30,
#        37,
#        39,
#        51,
#        56,
#        64,
#        71,
#        86,
#        113,
#        116,
#        121
#    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make a toy submission")

    path_ordi = "D:/Elodie/Documents/ULg/Master 2/Machine learning/Projets/3/"

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
tuned_parameters = [{ 'C' : [1, 10]
                    }]
scores = ['roc_auc']


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    # Chercher GridSearchCV dans documentation
    clf = GridSearchCV(LinearSVC(penalty='l1', dual=False, class_weight='balanced'), tuned_parameters, cv=2, scoring='%s' % score, n_jobs=-1, verbose=10)
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
#------------------------------------------------------------------------------
#Permet de selectionner

knn = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=12, n_estimators=134, n_jobs=-1)

# classifications
rfecv = RFECV(estimator=knn, step=3, cv=StratifiedKFold(2),scoring='roc_auc')
rfecv.fit(X_LS, y_LS)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
