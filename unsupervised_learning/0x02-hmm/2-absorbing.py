#!/usr/bin/env python3
"""
Absorbing Chains
"""
import numpy as np


def absorbing(P):
    """
    Function that determines if a markov chain is absorbing:

    Arguments:
     - P is a is a square 2D numpy.ndarray of shape (n, n) representing
        the standard transition matrix
        * P[i, j] is the probability of transitioning from state i to state j
        * n is the number of states in the markov chain

    Returns:
     True if it is absorbing, or False on failure
    """

    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    n1, n2 = P.shape
    if n1 != n2:
        return None
    if np.sum(P, axis=1).all() != 1:
        return None

    d = np.diag(P)
    if (d == 1).all():
        return True
    if not (d == 1).any():
        return False

    for i in range(n1):
        for j in range(n2):
            if i == j and i + 1 < len(P):
                if P[i + 1][j] == 0 and P[i][j + 1] == 0:
                    return False

    return True
