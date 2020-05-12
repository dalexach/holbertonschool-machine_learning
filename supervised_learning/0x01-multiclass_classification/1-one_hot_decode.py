#!/usr/bin/env python3
"""
Function One-Hot Decode
"""


import numpy as np


def one_hot_decode(one_hot):
    """
    Function that converts a one-hot matrix into a vector of labels
    Arguments:
     - one_hot (numpy.ndarray): is a one-hot encoded with shape (classes, m)
       * classes is the maximum number of classes
       * m is the number of examples
    Returns:
     A numpy.ndarray with shape (m, ) containing the numeric labels for each
     example, or None on failure
    """
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot.shape) != 2:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None

    return np.argmax(one_hot, axis=0)
