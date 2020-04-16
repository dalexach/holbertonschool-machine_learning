#!/usr/bin/env python3
"""
This file contains cat_arrays function
"""


def cat_arrays(arr1, arr2):
    """
    Function that concatenates two arrays

    Parameters:
    - arr1 (list of ints/floats): first array to concatenate
    - arr2 (list of ints/floats): second array to concatenate

    Return:
     New list with the arrays concatenated
    """

    new_arr = arr1[:] + arr2[:]

    return new_arr
