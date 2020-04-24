#!/usr/bin/env python3
"""
    Module to calculate the summation
"""


def summation_i_squared(n):
    """
     Function that calculates the summation of i squared
    Arguments:
    - n (int): the stopping condition
    Return:
     The sum of power to 2, if is not a valid number, None
    """

    if n > 0:
        return int(n*(n+1)*((2*n)+1)/6)
    else:
        return None
