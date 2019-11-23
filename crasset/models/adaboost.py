import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
import base_methods as base

# ------------------------------- Hyperparameters tuning ------------------------------- #
def tuning():
    # Setting the range of parameters
    n_estimators = list(range(1,100,10))
    loss = ["linear", "square", "exponential"]
    learning_rate = [2,1,0.5]
    grid =      {'n_estimators' : n_estimators,
                'loss':loss,
                'learning_rate':learning_rate
                }
    base_model = DecisionTreeRegressor()
    model = AdaBoostRegressor(base_model)
    optimal_parameters = base.hyper_tuning(model, grid)
    print('Optimal parameters: ', optimal_parameters)

def adaboost():
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
    base_model = DecisionTreeRegressor()
    model =  AdaBoostRegressor(base_model)
    with base.measure_time('Training'):
        print("Training with adaboost...")
        model.fit(X_train, y_train)

    # -----------------------Submission: Running model on provided test_set---------------------------- #

    base.submit_from_model(model,"MF_withAdaboost")
if __name__ == '__main__':
    # adaboost()
    tuning()
    