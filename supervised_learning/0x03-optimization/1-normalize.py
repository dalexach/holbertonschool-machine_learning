#!/usr/bin/env python3
"""
Normalize
"""


import numpy as np


def normalize(X, m, s):
    """
    Function that normalizes (standardizes) a matrix
    Arguments:
     - X: is the numpy.ndarray of shape (d, nx) to normalize
        * d is the number of data points
        * nx is the number of features
     - m: is a numpy.ndarray of shape (nx,) that contains the mean of
          all features of X
     - s: is a numpy.ndarray of shape (nx,) that contains the standard
          deviation of all features of X
    Returns:
     The normalized X matrix
    """
    standardizes = (X - m) / s

    return standardizes
