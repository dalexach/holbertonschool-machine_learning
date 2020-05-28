#!/usr/bin/env python3
"""
Put it all together
"""


import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """
    Function that shuffles the data points in two matrices the same way:
    Arguments:
     - X: is the first numpy.ndarray of shape (m, nx) to shuffle
        * m is the number of data points
        * nx is the number of features in X
     - Y: is the second numpy.ndarray of shape (m, ny) to shuffle
        * m is the same number of data points as in X
        * ny is the number of features in Y
    Returns:
     The shuffled X and Y matrices
    """
    perm = np.random.permutation(len(X))

    shuffled_X = X[perm]
    shuffled_Y = Y[perm]

    return shuffled_X, shuffled_Y


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Function that creates the training operation for a NN in tensorflow
    using the Adam optimization algorithm
    Arguments:
     - loss is the loss of the network
     - alpha is the learning rate
     - beta1 is the weight used for the first moment
     - beta2 is the weight used for the second moment
     - epsilon is a small number to avoid division by zero
    Returns:
    The Adam optimization operation
    """

    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                                       beta2=beta2, epsilon=epsilon)
    train = optimizer.minimize(loss)

    return train


def create_layer(prev, n, activation):
    """
    Function that creates the layer
    Arguments:
     - prev: tensor output of the previus layer
     - n: numer of nodes in the new layer
     - activation: activation function to use
    Return:
     The new layer
    """

    kernel_ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initialzier=kernel_ini, name='layer')


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Function that builds, trains, and saves a NN model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization:
    Arguments:
     - Data_train is a tuple containing the training inputs and training
        labels, respectively
     - Data_valid is a tuple containing the validation inputs and
        validation labels, respectively
     - layers is a list containing the number of nodes in each layer of
        the network
     - activation is a list containing the activation functions used
        for each layer of the network
     - alpha is the learning rate
     - beta1 is the weight for the first moment of Adam Optimization
     - beta2 is the weight for the second moment of Adam Optimization
     - epsilon is a small number used to avoid division by zero
     - decay_rate is the decay rate for inverse time decay of the
        learning rate (the corresponding decay step should be 1)
     - batch_size is the number of data points that should be in a mini-batch
     - epochs is the number of times the training should pass through
        the whole dataset
     - save_path is the path where the model should be saved to
    Returns:
     The path where the model was saved
    """

    
