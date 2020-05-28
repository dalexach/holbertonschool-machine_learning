#!/usr/bin/env python3
"""
Momentum
"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Function that updates a variable using the gradient descent with momentum
    optimization algorithm:
    Arguments:
     - alpha is the learning rate
     - beta1 is the momentum weight
     - var is a numpy.ndarray containing the variable to be updated
     - grad is a numpy.ndarray containing the gradient of var
     - v is the previous first moment of var
    Returns:
        The updated variable and the new moment, respectively
    """
    V = np.multiply(beta1, v) + np.multiply((1 - beta1), grad)
    Var = var - np.multiply(alpha, V)

    return Var, V
