#!/usr/bin/env python3
"""
Batch Normalization
"""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Function that normalizes an unactivated output of a NN using
    batch normalization
    Arguments:
     - Z is a numpy.ndarray of shape (m, n) that should be normalized
         * m is the number of data points
         * n is the number of features in Z
     - gamma is a numpy.ndarray of shape (1, n) containing the scales
        used for batch normalization
     - beta is a numpy.ndarray of shape (1, n) containing the offsets
        used for batch normalization
     - epsilon is a small number used to avoid division by zero
    Returns:
    The normalized Z matrix
    """

    m = Z.mean(0)
    v = Z.var(0)

    Z_n = (Z - m) / (v + epsilon) ** (1/2)
    NZ = gamma * Z_n + beta

    return NZ
