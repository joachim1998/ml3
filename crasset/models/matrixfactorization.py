import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import base_methods as base
from mf import MF

def matrix_factorization():
    prefix='Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    training_user_movie_pairs = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    training_labels = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    # Concatenating data
    user_movie_rating_triplets = np.hstack((training_user_movie_pairs, training_labels.reshape((-1, 1))))

    # Build the learning matrix
    rating_matrix = base.build_rating_matrix(user_movie_rating_triplets)

    # Build the model
    model = MF(rating_matrix, K=30, alpha=1e-5, beta=0.02, iterations=2000)
    with base.measure_time('Training'):
        print('Training matrix factorization...')
        model.train()
    
    # Save the predicted matrix
    predicted_matrix = np.matrix(model.full_matrix())
    with open('predicted_matrix.txt','wb') as f:
        for line in predicted_matrix:
            np.savetxt(f, line, fmt='%.5f')
    
    # -----------------------Submission: Running model on provided test_set---------------------------- #
    df = pd.read_csv("Data/data_test.csv")
    R = pd.read_csv('predicted_matrix.txt', sep=" ", header=None)
    R = R.values
    users = df['user_id'].values
    movies = df['movie_id'].values
    ratings = []
    for u,m in zip(users,movies):
        if (R[u-1][m-1] > 5.00) :
            ratings.append(5.00)
        else:
            ratings.append(R[u-1][m-1])

    fname = base.make_submission(ratings, df.values.squeeze(), 'MatrixFactorization')
    print('Submission file "{}" successfully written'.format(fname))

if __name__ == '__main__':
    matrix_factorization()