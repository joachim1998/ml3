"""
Test the RandomForestClassifier estimator for different values of trees, depths

"""

import argparse

from real_submission import load_from_csv
from real_submission import measure_time
from real_submission import create_fingerprints


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# public score = 0.76789, score calulated = 0.785 (+/- 0.002)

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
tuned_parameters = [{'class_weight' : [None, 'balanced', 'balanced_subsample'] #n_estimators' : np.linspace(3000, 4000, num=11, dtype='int'), 'max_depth' : [1, 5, 10, 100, None], 'bootstrap' : [True, False], 'class_weight' : [None, 'balanced', 'balanced_subsample']
                    }]
scores = ['roc_auc']


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    # Chercher GridSearchCV dans documentation
    clf = GridSearchCV(RandomForestClassifier(n_estimators=3100, criterion='entropy', max_depth=None, bootstrap=True, n_jobs=-1), tuned_parameters, cv=2, scoring='%s' % score, n_jobs=-1, verbose=10)
    clf.fit(X_LS, y_LS)

    print("Best parameters set found on development set:")
    print()
    print("%0.3f (+/-%0.03f) for %r" %(clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score'], clf.best_params_))
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
