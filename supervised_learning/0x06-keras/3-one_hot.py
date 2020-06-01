#!/usr/bin/env python3
"""
One hot with Keras
"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Function that converts a label vector into a one-hot matrix
    Arguments:
     - labels
     - classes is the number of classes
    Returns:
     The one-hot matrix
    """
    onehot = K.utils.to_categorical(labels, classes)

    return onehot
