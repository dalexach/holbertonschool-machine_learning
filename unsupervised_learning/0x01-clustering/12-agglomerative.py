#!/usr/bin/env python3
"""
Agglomerative function
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Function that performs agglomerative clustering on a dataset

    Argumetns:
     - X is a numpy.ndarray of shape (n, d) containing the dataset
     - dist is the maximum cophenetic distance for all clusters

    Returns:
     clss, a numpy.ndarray of shape (n,) containing the cluster indices
     for each data point
    """
