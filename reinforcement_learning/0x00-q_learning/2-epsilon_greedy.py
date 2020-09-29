#!/usr/bin/env python3
"""
Epsilon Greedy
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Function that uses epsilon-greedy to determine the next action

    Arguments:
     - Q is a numpy.ndarray containing the q-table
     - state is the current state
     - epsilon is the epsilon to use for the calculation

    Returns:
     The next action index
    """

    e = np.random.uniform(0, 1)

    if e > epsilon:
        action = np.argmax(Q[state, :])
    else:
        action = np.random.randint(0, 3, None)

    return action
