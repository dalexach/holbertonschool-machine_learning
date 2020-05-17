#!/usr/bin/env python3
"""
Function create_train_op
"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """
    Function that creates the training operation for the network
    Arguments:
     - loss is the loss of the network’s prediction
     - alpha is the learning rate
    Returns:
    An operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

    return optimizer
