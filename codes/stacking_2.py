"""
Stack of RandomForestClassifier
This code is not ours, it has been found on this link:
https://github.com/dawidkopczyk/blog/blob/master/stacking.py

"""
import argparse
import numpy as np

from real_submission import load_from_csv
from real_submission import measure_time
from real_submission import create_fingerprints
from real_submission import make_submission

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from copy import copy as make_copy

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

    #==============================================================================
    # Generate classification data    
    #==============================================================================
    SEED = 2018

    # Load training data
    LS = load_from_csv(args.ls)
    # Load test data
    TS = load_from_csv(args.ts)

    with measure_time("Creating fingerprint"):
        X_train = create_fingerprints(LS["SMILES"].values)
    y_train = LS["ACTIVE"].values

    TS = load_from_csv(args.ts)
    X_test = create_fingerprints(TS["SMILES"].values)

    #==============================================================================
    # Define Base (level 0) and Stacking (level 1) estimators
    #==============================================================================
    base_clf = [RandomForestClassifier(n_estimators=3100, bootstrap=True,
                                       max_depth=None, class_weight='balanced_subsample')
                ]
    stck_clf = MLPClassifier(random_state = 42, alpha=1e-5, early_stopping=True,
                             activation='logistic', hidden_layer_sizes=(200,),
                             learning_rate='constant', learning_rate_init=0.003)

    #==============================================================================
    # Create Hold Out predictions (meta-features)
    #==============================================================================
    def hold_out_predict(clf, X, y, cv):
            
        """Performing cross validation hold out predictions for stacking"""
        
        # Initilize
        n_classes = len(np.unique(y)) # Assuming that training data contains all classes
        meta_features = np.zeros((X.shape[0], n_classes)) 
        n_splits = cv.get_n_splits(X, y)
        
        # Loop over folds
        print("Starting hold out prediction with {} splits for {}.".format(n_splits, clf.__class__.__name__))
        cnt = 0
        for train_idx, hold_out_idx in cv.split(X, y): 
            
            # Split data
            X_train = X[train_idx]    
            y_train = y[train_idx]
            X_hold_out = X[hold_out_idx]
            
            # Fit estimator to K-1 parts and predict on hold out part
            est = make_copy(clf)
            est.fit(X_train, y_train)
            y_hold_out_pred = est.predict_proba(X_hold_out)

            print("Loop nb " + str(cnt))
            cnt += 1
            
            # Fill in meta features
            meta_features[hold_out_idx] = y_hold_out_pred

        return meta_features

    #==============================================================================
    # Create meta-features for training data
    #==============================================================================
    # Define 4-fold CV
    cv = KFold(n_splits=2, random_state=SEED)
    print("Create meta-features for training data")
    # Loop over classifier to produce meta features
    meta_train = []
    for clf in base_clf:
        
        # Create hold out predictions for a classifier
        meta_train_clf = hold_out_predict(clf, X_train, y_train, cv)
        
        # Remove redundant column
        meta_train_clf = np.delete(meta_train_clf, 0, axis=1).ravel()
        
        # Gather meta training data
        meta_train.append(meta_train_clf)
        
    meta_train = np.array(meta_train).T 

    #==============================================================================
    # Create meta-features for testing data
    #==============================================================================
    print("Create meta-features for testing data")
    meta_test = []
    for clf in base_clf:
        print("Starting create meta-features for {}.".format(clf.__class__.__name__))
        # Create hold out predictions for a classifier
        clf.fit(X_train, y_train)
        meta_test_clf = clf.predict_proba(X_test)
        
        # Remove redundant column
        meta_test_clf = np.delete(meta_test_clf, 0, axis=1).ravel()
        
        # Gather meta training data
        meta_test.append(meta_test_clf)
        
    meta_test = np.array(meta_test).T 

    #==============================================================================
    # Predict on Stacking Classifier
    #==============================================================================
    print("Predict on Stacking Classifier")
    # Set seed
    if 'random_state' in stck_clf.get_params().keys():
        stck_clf.set_params(random_state=SEED)

    # Optional (Add original features to meta)
    original_flag = False
    if original_flag:
        meta_train = np.concatenate((meta_train, X_train), axis=1)
        meta_test = np.concatenate((meta_test, X_test), axis=1)

    #scores=cross_val_score(stck_clf, meta_train, y_train, cv=KFold(n_splits=3, shuffle =True, random_state=13), scoring='roc_auc', n_jobs=-1, verbose=1)
    #print(scores)

    # Fit model
    stck_clf.fit(meta_train, y_train)

    # Predict
    y_pred = stck_clf.predict_proba(meta_test)[:,1]
    
    # Estimated AUC of the model
    auc_predicted = 0.7

    # Making the submission file
    fname = make_submission(y_pred, auc_predicted, 'stacking_prediction')
    print('Submission file "{}" successfully written'.format(fname))
