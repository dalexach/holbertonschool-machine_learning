#!/usr/bin/env python3
"""
Maximization function
"""
import numpy as np


def maximization(X, g):
    """
    Function that calculates the maximization step in the EM algorithm
    for a GMM

    Arguments:
     - X is a numpy.ndarray of shape (n, d) containing the data set
     - g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster

    Returns:
     pi, m, S, or None, None, None on failure
        - pi is a numpy.ndarray of shape (k,) containing the updated priors
        for each cluster
        - m is a numpy.ndarray of shape (k, d) containing the updated centroid
        means for each cluster
        - S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """
