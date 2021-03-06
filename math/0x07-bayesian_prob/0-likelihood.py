#!/usr/bin/env python3
"""
Likelihood

You are conducting a study on a revolutionary cancer drug and are looking
to find the probability that a patient who takes this drug will develop severe
side effects. During your trials, n patients take the drug and x patients
develop severe side effects. You can assume that x follows
a binomial distribution.
"""
import numpy as np


def likelihood(x, n, P):
    """
    Function that calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects

    Arguments:
     - x is the number of patients that develop severe side effects
     - n is the total number of patients observed
     - P is a 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects

    Returns:
     A 1D numpy.ndarray containing the likelihood of obtaining
     the data, x and n, for each probability in P, respectively
    """

    if not isinstance(n, int) or (n <= 0):
        raise ValueError('n must be a positive integer')

    if not isinstance(x, int) or (x < 0):
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError('All values in P must be in the range [0, 1]')

    nu = (np.math.factorial(n))
    d = (np.math.factorial(x) * np.math.factorial(n - x))
    factorial = nu / d
    total = factorial * (np.power(P, x)) * (np.power((1 - P), (n - x)))

    return total
