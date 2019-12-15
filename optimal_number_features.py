import numpy as np

from extract_data import generate_learning
from extract_data import create_number_fingerprints
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


features_numbers = [50, 123, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                    1500, 2000 ,2500, 3000, 3500, 4000, 4500, 5000, 6000]


LS, __ = generate_learning()
y_LS = LS["ACTIVE"].values

with open('optimal_features.txt', 'w') as handle:
    handle.write('Number,Mean,Std\n')




    for ftr in features_numbers:

        X_LS = create_number_fingerprints(LS["SMILES"].values, nBits=ftr)

        model=RandomForestClassifier(n_estimators=3100, bootstrap=True, max_depth= None, class_weight='balanced_subsample', n_jobs=-1, verbose=10)
        scores = cross_val_score(model, X_LS, y_LS, scoring='roc_auc', cv=3, n_jobs=None)

        mean = np.mean(scores)
        std = np.std(scores)

        line = '{:d}, {}, (+/-%0.03f)\n'.format(ftr, np.mean(scores), np.std(scores)*2)
        handle.write(line)