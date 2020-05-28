#!/usr/bin/env python3
"""
Batch normalization upgraded
"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Function that creates a batch normalization layer for a NN in tensorflow:
    Arguments:
     - prev is the activated output of the previous layer
     - n is the number of nodes in the layer to be created
     - activation is the activation function that should be used on
        the output of the layer
    Returns:
     A tensor of the activated output for the layer
    """

    kernel_ini = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, kernel_initialzier=kernel_ini)

    z = lay(prev)

    m, v = tf.nn.moments(z, [0])
    beta = tf.Variable(tf.zeros([z.get_shape()[-1]]))
    gamma = tf.Variable(tf.ones([z.get_shape()[-1]]))
    z_n = tf.nn.batch_normalization(z, m, v, beta, gamma, 1e-8)
    y_pred = activation(z_n)

    return y_pred
