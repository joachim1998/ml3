import os
import time
import datetime
from contextlib import contextmanager
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib

@contextmanager
def measure_time(label):
    """
    Context manager to measure time of computation.
    >>> with measure_time('Heavy computation'):
    >>>     do_heavy_computation()
    'Duration of [Heavy computation]: 0:04:07.765971'

    Parameters
    ----------
    label: str
        The label by which the computation will be referred
    """
    start = time.time()
    yield
    end = time.time()
    print('Duration of [{}]: {}'.format(label,
                                        datetime.timedelta(seconds=end-start)))


def load_from_csv(path, delimiter=','):
    """
    Load csv file and return a NumPy array of its data

    Parameters
    ----------
    path: str
        The path to the csv file to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    D: array
        The NumPy array of the data contained in the file
    """
    return pd.read_csv(path, delimiter=delimiter).values.squeeze()


def build_sparsed_rating_matrix(user_movie_rating_triplets):
    """
    Create the rating matrix from triplets of user and movie and ratings.

    A rating matrix `R` is such that `R[u, m]` is the rating given by user `u`
    for movie `m`. If no such rating exists, `R[u, m] = 0`.

    Parameters
    ----------
    user_movie_rating_triplets: array [n_triplets, 3]
        an array of trpilets: the user id, the movie id, and the corresponding
        rating.
        if `u, m, r = user_movie_rating_triplets[i]` then `R[u, m] = r`

    Return
    ------
    R: sparse csr matrix [n_users, n_movies]
        The rating matrix
    """
    rows = user_movie_rating_triplets[:, 0]
    cols = user_movie_rating_triplets[:, 1]
    training_ratings = user_movie_rating_triplets[:, 2]

    return sparse.coo_matrix((training_ratings, (rows, cols))).tocsr()

def build_rating_matrix(user_movie_rating_triplets):
    """
    Create the rating matrix from triplets of user and movie and ratings.

    A rating matrix `R` is such that `R[u, m]` is the rating given by user `u`
    for movie `m`. If no such rating exists, `R[u, m] = 0`.

    Parameters
    ----------
    user_movie_rating_triplets: array [n_triplets, 3]
        an array of trpilets: the user id, the movie id, and the corresponding
        rating.
        if `u, m, r = user_movie_rating_triplets[i]` then `R[u, m] = r`

    Return
    ------
    R: sparse csr matrix [n_users, n_movies]
        The rating matrix
    """
    rows = user_movie_rating_triplets[:, 0]
    cols = user_movie_rating_triplets[:, 1]
    training_ratings = user_movie_rating_triplets[:, 2]

    return sparse.coo_matrix((training_ratings, (rows, cols))).toarray()


def create_learning_matrices_features(rating_matrix, user_movie_pairs_features):
    """
    Create the learning matrix `X` from the `rating_matrix`.

    If `u, m = user_movie_pairs[i]`, then X[i] is the feature vector
    corresponding to user `u` and movie `m`. The feature vector is composed
    of `n_users + n_movies` features. The `n_users` first features is the
    `u-th` row of the `rating_matrix`. The `n_movies` last features is the
    `m-th` columns of the `rating_matrix`

    In other words, the feature vector for a pair (user, movie) is the
    concatenation of the rating the given user made for all the movies and
    the rating the given movie receive from all the user.

    Parameters
    ----------
    rating_matrix: sparse matrix [n_users, n_movies]
        The rating matrix. i.e. `rating_matrix[u, m]` is the rating given
        by the user `u` for the movie `m`. If the user did not give a rating for
        that movie, `rating_matrix[u, m] = 0`
    user_movie_pairs: array [n_predictions, 2]
        If `u, m = user_movie_pairs[i]`, the i-th raw of the learning matrix
        must relate to user `u` and movie `m`

    Return
    ------
    X: sparse array [n_predictions, n_users + n_movies]
        The learning matrix in csr sparse format
    """

    prefix = 'Data/'

    # Feature for users
    rating_matrix = rating_matrix.tocsr()
    user_features = rating_matrix[user_movie_pairs_features['user_id'].values]
    # Features for movies
    rating_matrix = rating_matrix.tocsc()
    movie_features = rating_matrix[:, user_movie_pairs_features['movie_id'].values].transpose()

    #Add additional features
    user_cols = ['age', 'gender_M']
    movie_cols = ['Action','Adventure','Animation',
                    'Children','Comedy','Crime','Documentary',
                    'Drama','Fantasy','Film-Noir','Horror',
                    'Musical','Mystery','Romance','Sci-Fi',
                    'Thriller','War','Western']
    
    additional_user_features = user_movie_pairs_features[user_cols].values
    additional_movie_features = user_movie_pairs_features[movie_cols].values

    
    X = sparse.hstack((user_features, 
                        movie_features,
                        additional_user_features,
                        additional_user_features,
                        additional_movie_features))
    return X.tocsr()


def create_learning_matrices(rating_matrix, user_movie_pairs):
    """
    Create the learning matrix `X` from the `rating_matrix`.

    If `u, m = user_movie_pairs[i]`, then X[i] is the feature vector
    corresponding to user `u` and movie `m`. The feature vector is composed
    of `n_users + n_movies` features. The `n_users` first features is the
    `u-th` row of the `rating_matrix`. The `n_movies` last features is the
    `m-th` columns of the `rating_matrix`

    In other words, the feature vector for a pair (user, movie) is the
    concatenation of the rating the given user made for all the movies and
    the rating the given movie receive from all the user.

    Parameters
    ----------
    rating_matrix: sparse matrix [n_users, n_movies]
        The rating matrix. i.e. `rating_matrix[u, m]` is the rating given
        by the user `u` for the movie `m`. If the user did not give a rating for
        that movie, `rating_matrix[u, m] = 0`
    user_movie_pairs: array [n_predictions, 2]
        If `u, m = user_movie_pairs[i]`, the i-th raw of the learning matrix
        must relate to user `u` and movie `m`

    Return
    ------
    X: sparse array [n_predictions, n_users + n_movies]
        The learning matrix in csr sparse format
    """
    # Feature for users

    user_features = rating_matrix[user_movie_pairs[:, 0]]
    # Features for movies
    movie_features = rating_matrix[:, user_movie_pairs[:, 1]].transpose()

    X = np.hstack((user_features, movie_features))
    return X

def cross_validation(model, X_cv, y_cv, K):
	"""
	Perform cross validation for a given model

	Parameters
    ----------
    model: the model to cross validate
    X_cv: array [nb_ratings, nb_features]
		X_cv[r, f] is the feature f of a rating r given by a user to a movie
    y_cv: array [nb_ratings]
		y_cv[r] is the rating r given by a user to a movie
    K: int
    	the number of folds

    Return
    ------
	mean_MSE: float
		the mean of the mean squared error for a split
	"""

	scores = cross_validate(model, X_cv, y_cv, cv = K, scoring = 'neg_mean_squared_error', return_train_score = False, n_jobs = -1)
	mean_MSE = abs(np.mean(scores['test_score']))

	return mean_MSE

def hyper_tuning(model, grid):
    """
    Tune hyperparameters of a model

    Parameters:
    -----------
    model :
        Estimator on which to tune the hyperparameters
    grid :
        Dictionnary with keys being the name of the parameter
        and having as values the list of values of the parameter
        to try
    Returns:
    -------

    Best combination of hyperparameters for the given model
    """
    prefix = 'Data/'

    # Load training data
    X_train = load_from_csv(os.path.join(prefix, 'train_user_movie_merge.csv'))
    y_train = load_from_csv(os.path.join(prefix, 'output_train.csv'))

    rf_random = RandomizedSearchCV(estimator = model, 
                                    param_distributions = grid, 
                                    cv = 3,
                                    verbose=2, 
                                    n_jobs = -1  
                                    )
    print('Starting search...')
    rf_random.fit(X_train, y_train)

    return rf_random.best_params_


def make_submission(y_predict, user_movie_ids, file_name='submission',
                    date=True):
    """
    Write a submission file for the Kaggle platform

    Parameters
    ----------
    y_predict: array [n_predictions]
        The predictions to write in the file. `y_predict[i]` refer to the
        user `user_ids[i]` and movie `movie_ids[i]`
    user_movie_ids: array [n_predictions, 2]
        if `u, m = user_movie_ids[i]` then `y_predict[i]` is the prediction
        for user `u` and movie `m`
    file_name: str or None (default: 'submission')
        The path to the submission file to create (or override). If none is
        provided, a default one will be used. Also note that the file extension
        (.txt) will be appended to the file.
    date: boolean (default: True)
        Whether to append the date in the file name

    Return
    ------
    file_name: path
        The final path to the submission file
    """

    # Naming the file
    if date:
        file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))

    file_name = '{}.txt'.format(file_name)

    # Writing into the file
    with open(file_name, 'w') as handle:
        handle.write('"USER_ID_MOVIE_ID","PREDICTED_RATING"\n')
        for (user_id, movie_id), prediction in zip(user_movie_ids,
                                                 y_predict):

            if np.isnan(prediction):
                raise ValueError('The prediction cannot be NaN')
            line = '{:d}_{:d},{}\n'.format(user_id, movie_id, prediction)
            handle.write(line)
    return file_name



def submit_from_file(estimator_file, submission_file):
    """
    Predict the model on the provided test set and
    create a submission file

    Parameters
    ----------
    estimator_file : 
        Name of the estimator to load from file
    submission_file: 
        Name to give to the submission file

    """
    prefix = 'Data/'

    # Load test data
    X_test = load_from_csv(os.path.join(prefix, 'test_user_movie_merge.csv'))
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    # Loading the chosen estimator
    model = joblib.load(estimator_file)
    y_pred = model.predict(X_test)

    for i in range(len(y_pred)):
        if y_pred[i] > 5:
            y_pred[i] = 5
        if y_pred[i] < 0:
            y_pred[i] = 0

    fname = make_submission(y_pred, test_user_movie_pairs, submission_file)
    print('Submission file "{}" successfully written'.format(fname))

def submit_from_model(model, submission_file):
    """
    Predict the model on the provided test set and
    create a submission file

    Parameters
    ----------
    model : 
        Model to predict on
    submission_file: 
        Name to give to the submission file

    """
    prefix = 'Data/'

    # Load test data
    X_test = load_from_csv(os.path.join(prefix, 'data_train.csv'))
    test_user_movie_pairs = load_from_csv(os.path.join(prefix, 'data_test.csv'))

    y_pred = model.predict(X_test)

    for i in range(len(y_pred)):
        if y_pred[i] > 5:
            y_pred[i] = 5
        if y_pred[i] < 0:
            y_pred[i] = 0

    fname = make_submission(y_pred, test_user_movie_pairs, submission_file)
    print('Submission file "{}" successfully written'.format(fname))
