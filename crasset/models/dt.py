import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import base_methods as base


def decision_tree():
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

    # Build model
    depth = 11
    with base.measure_time('Training'):
        print('Training DT with a depth of {}...'.format(depth))
        model = DecisionTreeRegressor(max_depth = depth)
        model.fit(X_train, y_train)
       
    # -----------------------Submission: Running model on provided test_set---------------------------- #
    base.submit_from_model(model,"MF_withDecisionTree")

if __name__ == "__main__":
    decision_tree()