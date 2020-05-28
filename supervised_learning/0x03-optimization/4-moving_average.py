#!/usr/bin/env python3
"""
Moving average
"""


import numpy as np


def moving_average(data, beta):
    """
    Function that calculates the weighted moving average of a data set:
    Arguments:
     - data (list): is the list of data to calculate the moving average of
     - beta (list): is the weight used for the moving average
    Returns:
     A list containing the moving averages of data
    """

    moving_average = []
    V = 0
    t = 1

    for d in data:
        V = beta * V + (1 - beta) * d
        correct = V / 1 - (beta ** t)
        moving_average.append(correct)
        t += 1

    return moving_average
