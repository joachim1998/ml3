import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import base_methods as base


def linearregression():
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
    model = LinearRegression()
    with base.measure_time('Training'):
        print("Training with linear regression...")
        model.fit(X_train, y_train)

    base.submit_from_model(model,"MF_withLinearRegression")

if __name__ == '__main__':
    linearregression()