#!/usr/bin/env python3
"""
This file contains matrix_transpose function
"""


def matrix_transpose(matrix):
    """
    Function that transpose a 2D matrix

    Parameters:
    - matrix (list of lists): matrix to transpose

    Return:
     The matrix transpose (list of lists)
    """

    T_matrix = []
    # cycle for rows
    for r in range(len(matrix[0])):
        # list for the colums
        col = []
        # cycle for columns
        for c in range(len(matrix)):
            # changing the row for columns
            col.append(matrix[c][r])
        # adding to the transpose matrix
        T_matrix.append(col)

    return T_matrix
