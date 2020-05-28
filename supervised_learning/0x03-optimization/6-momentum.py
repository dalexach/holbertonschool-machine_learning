#!/usr/bin/env python3
"""
Momentum upgraded
"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Function that creates the training operation for a NN in tensorflow
    using the gradient descent with momentum optimization algorithm
    Arguments:
     - loss is the loss of the network
     - alpha is the learning rate
     - beta1 is the momentum weight
    Returns:
     The momentum optimization operation
    """

    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    train = optimizer.minimize(loss)

    return train
