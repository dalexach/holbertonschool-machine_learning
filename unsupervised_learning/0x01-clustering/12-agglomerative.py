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

    link = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(link,
                                            t=dist,
                                            criterion='distance')

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(link,
                                       color_threshold=dist)
    plt.show()

    return clss
