#!/usr/bin/env python3
"""
Function calculate_loss
"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Function that calculates the softmax cross.entropy loss of a prediction
    Arguments:
     - y:  is a placeholder for the labels of the input data
     - y_pred: is a tensor containing the network’s predictions
    Returns:
    A tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)

    return loss
