'''
1. You need to implement

Analytical solution for L2-regularized linear regression that minimizes mean squared error function. (taught in lecture 4b)
An iterative solution using gradient descent for L2-regularized linear regression that minimizes mean squared error function (taught in lecture 4c)

2. Feel free to transform original features using basis functions, feature scaling etc.
Tip: You will find it useful to apply some form of feature scaling to your inputs before running gradient descent on your loss function.
Also tuning the hyperparameters like learning rate, regularization weight etc. should improve your results.
'''
import pandas as pd
import numpy as np
import matplotlib
from dateutil.rrule import weekday
from nltk.metrics.aline import feature_matrix

matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)


class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self):
        raise NotImplementedError

    def __call__(self, features, is_train=False):
        raise NotImplementedError


'''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''

'''
Arguments:
csv_path: path to csv file
is_train: True if using training data (optional)
scaler: a class object for doing feature scaling (optional)
'''

'''
help:
useful links: 
    * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
    * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
'''


def get_features(csv_path, is_train=False, scaler=None):
    df_feature = pd.read_csv(csv_path, sep=r'\s*,\s*', engine='python')
    df_feature.drop('shares',axis='columns', inplace=True)
    '''feature_bias=np.zeros(len(df_feature.index))
    feature_bias.fill(1)
    df_feature.insert(loc=0, column='bias', value=feature_bias)'''
    # if(is_train):
    # df_train=pd.read_csv('train.csv',sep=r'\s*,\s*',engine='python')

    return df_feature.to_numpy()


def get_targets(csv_path):
    df_feature = pd.read_csv(csv_path, sep=r'\s*,\s*', engine='python')
    return df_feature['shares'].to_numpy()
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''


def analytical_solution(feature_matrix, targets, C=0.0):

    x=np.linalg.inv(feature_matrix.transpose().dot( feature_matrix))
    x=x+C*np.identity(x.shape[0])
    w= x.dot(feature_matrix.T).dot(targets)
    return w
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 4b
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape m x 1
    '''


def get_predictions(feature_matrix, weights):
    return feature_matrix.dot(weights)
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''


def mse_loss(feature_matrix, weights, targets):
    prediction=get_predictions(feature_matrix, weights)
    a=targets-prediction
    return a.T.dot(a)

    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''


def l2_regularizer(weights):
    return weights.dot(weights)
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''


def loss_fn(feature_matrix, weights, targets, C=0.0):
    return mse_loss(feature_matrix,weights,targets)+C*l2_regularizer(weights)
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''

def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    raise NotImplementedError


def sample_random_batch(feature_matrix, targets, batch_size):
    batch=np.random.choice(feature_matrix.shape[0], batch_size, replace=False)
    sampled_feature_matrix=feature_matrix[batch, :]
    sampled_targets =targets[batch]
    return (sampled_feature_matrix, sampled_targets)

    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''

    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''


def initialize_weights(n):
    weights = np.zeros(n)
    #weights.fill(1)
    return weights
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''

    '''
    Arguments
    n: int
    '''


def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    '''

    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''

    raise NotImplementedError


def early_stopping(arg_1=None, arg_2=None, arg_3=None, arg_n=None):
    # allowed to modify argument list as per your need
    # return True or False
    raise NotImplementedError


def do_gradient_descent(train_feature_matrix,
                        train_targets,
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=5):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights

    a sample code is as follows --
    '''
    weights = initialize_weights(train_feature_matrix.shape[1])
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)

    print("step {} \t dev loss: {} \t train loss: {}".format(0, dev_loss, train_loss))
    for step in range(1, max_steps + 1):

        # sample a batch of features and gradients
        features, targets = sample_random_batch(train_feature_matrix, train_targets, batch_size)

        # compute gradients
        gradients = compute_gradients(features, weights, targets, C)

        # update weights
        weights = update_weights(weights, gradients, lr)

        if step % eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            print("step {} \t dev loss: {} \t train loss: {}".format(step, dev_loss, train_loss))

        '''
        implement early stopping etc. to improve performance.
        '''

    return weights


def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error
    predictions = get_predictions(feature_matrix, weights)
    loss = mse_loss(feature_matrix, weights, targets)
    return loss


if __name__ == '__main__':
    scaler = Scaler()  # use of scaler is optional
    train_features, train_targets = get_features('data/train.csv', True, scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv', False, scaler), get_targets('data/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=1e-8)
    print('evaluating analytical_solution...')
    dev_loss = do_evaluation(dev_features, dev_targets, a_solution)
    train_loss = do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features,
                                                train_targets,
                                                dev_features,
                                                dev_targets,
                                                lr=1.0,
                                                C=0.0,
                                                batch_size=32,
                                                max_steps=2000000,
                                                eval_steps=5)

    print('evaluating iterative_solution...')
    dev_loss = do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss = do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))



