#!/usr/bin/env python3
"""
    Calculating the derivate of a polynomial
"""


def poly_derivative(poly):
    """
     Function that find the derivate of a polynomial
    Arguments:
     - poly(list of integers): polynomial to calculate the derivate
    Return:
     List of coefficients representing the derivative of the polynomial
    """

    if poly is None or poly == 0:
        return ([0])

    derivate = []
    i = 1


    while i < len(poly):
        derivate.append(poly[i]*i)
        i += 1
    return derivate
