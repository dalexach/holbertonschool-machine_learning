#!/usr/bin/env python3
"""
This file contains np_cat function
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis

    Parameters:
    - mat1 (numpy.ndarray): never empty
    - mat2 (numpy.ndarray): never empty
    - axis (int): axis to concatenate

    Return:
     A new numpy.ndarray with the matrices concatenated
    """

    return np.concatenate((mat1, mat2), axis)
