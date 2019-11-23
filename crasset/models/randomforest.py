import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import base_methods as base


# ------------------------------- Hyperparameters tuning ------------------------------- #
def parameter_tuning():
    
    # Number of features to consider at every split
    n_estimators = list(range(36,44,2))
    max_depth= list(range(9,12,1))
    # Minimum number of samples required to split a node
    min_samples_split = [5,6,7]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4,8,10]
    # Create the random grid
    random_grid = {'n_estimators' : n_estimators,
                        'max_depth' : max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                        }

    model =  RandomForestRegressor( criterion='mse', random_state = 42)
    optimal_parameters = base.hyper_tuning(model, random_grid)

    print('Optimal parameters: ', optimal_parameters)
    

def randomforest():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    R = pd.read_csv('predicted_matrix.txt', sep=" ", header=None)
    
    user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    X_train = base.create_learning_matrices_features(R.values, user_movie_pairs)
    y_train = training_labels

    # Best estimator after hyperparameter tuning
    model = RandomForestRegressor(bootstrap=True, 
                                    criterion='mse',
                                    random_state=42,
                                    n_estimators=38,
                                    n_jobs=-1,
                                    verbose = 2,
                                    max_depth= 9)

    print(model)
    print("Training...")
    with base.measure_time('Training'):
        model.fit(X_train, y_train)

    #Check for overfitting
    y_pred_train = model.predict(X_train)
    MSE_train = mean_squared_error(y_train, y_pred_train)
    print("Train set MSE: {}".format(MSE_train))

    base.submit_from_model(model,"MF_withRandomForest")

if __name__ == '__main__':
    parameter_tuning()
    # randomforest()