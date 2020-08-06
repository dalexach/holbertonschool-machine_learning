#!/usr/bin/env python3
"""
Expectation function
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Function that calculates the expectation step in the EM algorithm for a GMM

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the data set
     - pi is a numpy.ndarray of shape (k,) containing the priors for each
        cluster
     - m is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
     - S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster

    Returns:
     g, l, or None, None on failure
        - g is a numpy.ndarray of shape (k, n) containing the posterior
        probabilities for each data point in each cluster
    """
