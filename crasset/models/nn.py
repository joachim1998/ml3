import os
import pandas as pd
import numpy as np
import base_methods as base
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from matplotlib import pyplot as plt


# ------------------------------- Hyperparameters tuning ------------------------------- #
def tuning():
    # Setting the range of parameters
    hidden_layer_sizes = [
                          (10,),
                          (50,),
                          (100,),
                          (200,),
                          (500,)]
                        
    activation = ['logistic','tanh','relu']
    alpha = [1e-5]
    learning_rate = ['constant', 'adaptive']
    learning_rate_init = [0.0005,0.001,0.003]
    early_stopping = [True]
    
    grid = {'hidden_layer_sizes' : hidden_layer_sizes,
                        'activation' : activation,
                        'alpha':alpha,
                        'learning_rate':learning_rate,
                        'learning_rate_init':learning_rate_init,
                        'early_stopping': early_stopping
                        }

    model = MLPRegressor(random_state = 42)
    optimal_parameters = base.hyper_tuning(model, grid)
    print('Optimal parameters: ', optimal_parameters)


# Playing with the number of neurons
def neuralNetNeurons():
    prefix = 'Data/'

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_train = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    y_train = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    nb_neurons_list = range(20, 500, 20)

    accuracies = []
    for nb_neurons in nb_neurons_list:
        filename = "NNModel_neurons_{}.pkl".format(nb_neurons)

        # Skip if the model has already been trained at this number of neurons
        if(os.path.isfile(filename)):
            print("NNModel with {} neurons already trained. Import filename {}".format(nb_neurons,filename))
            accuracies.append(joblib.load(filename)[1])
        else:
            model = MLPRegressor(hidden_layer_sizes = (nb_neurons,), 
                                activation = 'logistic', 
                                learning_rate = 'adaptive', 
                                learning_rate_init = 0.003, 
                                alpha = 1e-05, 
                                early_stopping = True, 
                                random_state = 42)
            cv_model = MLPRegressor(hidden_layer_sizes = (nb_neurons,),
                                activation = 'logistic', 
                                learning_rate = 'adaptive', 
                                learning_rate_init = 0.003, 
                                alpha = 1e-05, 
                                early_stopping = True, 
                                random_state = 42)               
            # Training the model on the whole training data
            with base.measure_time('Training'):
                print('Training neural net...')
                model.fit(X_train, y_train)

            # Performing cross validation
            with base.measure_time('Cross validation'):
                print('Cross validation...')
                cv_score = base.cross_validation(cv_model, X_train, y_train, 5)
                accuracies.append(cv_score)

            print("Number of neurons : {} Loss : {} CV score: {}".format(nb_neurons, model.loss_, cv_score))

            # Save estimator to file so that we train once
            joblib.dump((model, cv_score), filename)

    # Plot accuracy for different number of neurons
    print(accuracies)
    plt.plot(nb_neurons_list, accuracies)
    plt.xlabel("number_of_neurons")
    plt.ylabel("mean_squared_error")
    
    plt.savefig("NNneurons.svg")

# Playing with the number of layers
def neuralNetLayers():
    prefix = 'Data/'

    # Load training data
    X_train = base.load_from_csv(os.path.join(prefix, 'data_train.csv'))
    y_train = base.load_from_csv(os.path.join(prefix, 'output_train.csv'))

    layers = []
    nb_layers = range(1, 11)
    for i in nb_layers:
        layers.append(tuple([50] * i))

    accuracies = []
    for layer in layers:
        filename = "NNModel_layers_{}.pkl".format(len(layer))

        # Skip if the model has already been trained at this number of layers
        if(os.path.isfile(filename)):
            print("NNModel with {} layers already trained. Import filename {}".format(len(layer),filename))
            accuracies.append(joblib.load(filename)[1])
        else:
            model = MLPRegressor(hidden_layer_sizes = layer, 
                                activation = 'logistic', learning_rate = 'adaptive', 
                                learning_rate_init = 0.003, 
                                alpha = 1e-05, 
                                early_stopping = True, 
                                random_state = 42)
            cv_model = MLPRegressor(hidden_layer_sizes = layer, 
                                activation = 'logistic', learning_rate = 'adaptive', 
                                learning_rate_init = 0.003, 
                                alpha = 1e-05, 
                                early_stopping = True, 
                                random_state = 42)
            # Training the model on the whole training data
            with base.measure_time('Training'):
                print('Training neural net...')
                model.fit(X_train, y_train)

            # Performing cross validation
            with base.measure_time('Cross validation'):
                print('Cross validation...')
                cv_score = base.cross_validation(cv_model, X_train, y_train, 5)
                accuracies.append(cv_score)

            print("Number of layers : {} Loss : {} CV score: {}".format(len(layer), model.loss_, cv_score))

            # Save estimator to file so that we train once
            joblib.dump((model, cv_score), filename)

    # Plot accuracy for different number of layers
    print(accuracies)
    plt.plot(nb_layers, accuracies)
    plt.xlabel("number_of_layers")
    plt.ylabel("mean_squared_error")
    plt.savefig("NNlayers.svg")


if __name__ == '__main__':
    tuning()
    # base.submit_from_file("NNModel_neurons_320.pkl","MLP")
