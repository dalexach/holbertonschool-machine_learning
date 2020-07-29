#!/usr/bin/env python3
"""
Class MultiNormal
"""
import numpy as np


class MultiNormal(object):
    """
    Class that represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Constructor funtcion

        Arguments:
         - data is a numpy.ndarray of shape (d, n) containing the data set:
         - n is the number of data points
         - d is the number of dimensions in each data point

        Public instance variables:
         - mean - a numpy.ndarray of shape (d, 1) containing the mean of data
         - cov - a numpy.ndarray of shape (d, d) containing
            the covariance matrix data
        """

        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        n = data.shape[1]
        if n < 2:
            raise ValueError('data must contain multiple data points')

        d = data.shape[0]
        self.mean = (np.mean(data, axis=1)).reshape(d, 1)

        X = data - self.mean
        self.cov = (np.dot(X, X.T)) / (n - 1)
