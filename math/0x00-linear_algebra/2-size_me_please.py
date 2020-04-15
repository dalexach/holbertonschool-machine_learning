#!/usr/bin/env python3
"""
This file contains matrix_shape function
"""


def matrix_shape(matrix):
    """
    Function that receives a matrix and calculate the shape of it

    Parameters:
     - matrix (list of lists): matrix to calculate the shape

    Return:
     The shape of the matrix as a list of integers
    """
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
