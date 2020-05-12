#!/usr/bin/env python3
"""
Function One-Hot Encode
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    Function that converts a numeric label vector into a one-hot matrix
    Arguments:
     - Y (numpy.ndarray): with shape (m,) containing numeric class labels
        * m is the number of examples
     - classes(int): is the maximum number of classes found in Y
    Returns:
     A one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if type(Y) is not np.ndarray:
        return None
    if len(Y) == 0:
        return None
    if type(classes) is not int or classes <= Y.max():
        return None

    one_hot = np.zeros((classes, Y.shape[0]))
    for c, m in enumerate(Y):
        one_hot[m][c] = 1

    return one_hot
