"""
KNearestNeighboursClassifier with the best combination of hyper-parameters

"""
import argparse

from real_submission import load_from_csv
from real_submission import measure_time
from real_submission import create_fingerprints

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


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
    tuned_parameters = [{ # every hyper-parameter can be tested
                        }]
    scores = ['roc_auc']
    
    for score in scores:

        # Chercher GridSearchCV dans documentation
        clf = GridSearchCV(KNeighborsClassifier(n_neighbors=53, algorithm='auto',
                                                weights='distance'),
    tuned_parameters, cv=2, scoring='%s' % score, n_jobs=-1, verbose=10)
        clf.fit(X_LS, y_LS)
    
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

