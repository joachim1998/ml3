import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import base_methods as base


# ------------------------------- Hyperparameters tuning ------------------------------- #
def tuning():
    # Setting the range of parameters
    max_depth = list(range(1,10))
                        
    min_samples_split = [1,2,5]
    learning_rate = [0.01,0.05, 0.1, 0.2]
    n_estimators = [30, 100, 200]
    
    grid = {'max_depth' : max_depth,
                        'min_samples_split' : min_samples_split,
                        'learning_rate':learning_rate,
                        'n_estimators':n_estimators,
                        }

    model = GradientBoostingRegressor(random_state = 42)
    optimal_parameters = base.hyper_tuning(model, grid)
    print('Optimal parameters: ', optimal_parameters)

def gradient_boosting():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    R = pd.read_csv('predicted_matrix.txt', sep=" ", header=None)
    user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Build the training learning matrix
    X_train = base.create_learning_matrices(R.values, user_movie_pairs)

    # Build the model
    y_train = training_labels

    # Best estimator after hyperparameter tuning
    model = GradientBoostingRegressor(max_depth = 4, min_samples_split=2, min_samples_leaf=0.0001)
    with base.measure_time('Training'):
        print("Training with gradient boosting...")
        model.fit(X_train, y_train)

    # -----------------------Submission: Running model on provided test_set---------------------------- #
    base.submit_from_model(model, "MF_withGradientBoosting")
if __name__ == '__main__':
    # gradient_boosting()
    tuning()
    