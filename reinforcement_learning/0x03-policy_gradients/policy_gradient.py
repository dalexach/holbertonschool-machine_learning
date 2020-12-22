#!/usr/bin/env python3
"""
Simple Policy function
"""
import numpy as np


def policy(matrix, weight):
    """
    Function that computes to policy with a weight of a matrix.

    Arguments:
     - matrix: the current observation of the environment.
     - weight: random weight.

    Returns:
     The calculated policy
    """
    z = matrix.dot(weight)
    exp = np.exp(z)

    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """
    Function that computes the Monte-Carlo policy gradient
    based on a state and a weight matrix.

    Arguemnts:
     - state matrix representing the current observation of the environment
     - weight matrix of random weight

    Return:
     The action and the gradient.
    """

    def softmax_grad(softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    prb = policy(state, weight)
    action = np.argmax(prb)
    dsoftmax = softmax_grad(prb)[action, :]
    dlog = dsoftmax / prb[0, action]
    gradient = state.T.dot(dlog[None, :])

    return (action, gradient)
