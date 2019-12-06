import os
import time
import datetime
import argparse
from contextlib import contextmanager

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectFromModel

from rdkit import Chem
from rdkit.Chem import AllChem

def randomForest(x_ls, y_ls, x_ts):
	model = RandomForestClassifier(n_estimators = 3000, max_depth = None, criterion='entropy', n_jobs=-1, verbose=9)

	model.fit(x_ls, y_ls)

	y_pred = model.predict_proba(x_ts)[:,1]

	auc_predicted = 0.50

	return y_pred, auc_predicted