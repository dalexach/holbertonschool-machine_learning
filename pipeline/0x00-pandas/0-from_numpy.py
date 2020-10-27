#!/usr/bin/env python3
import pandas as pd
"""
From Numpy
"""


def from_numpy(array):
    """
    Function that creates a pd.DataFrame from a np.ndarray

    Arguments:
     - array is the np.ndarray from which you should create the pd.DataFrame

    Note:
    The columns of the pd.DataFrame should be labeled in alphabetical order
     and capitalized.
    There will not be more than 26 columns.

    Returns:
     The newly created pd.DataFrame
    """

    alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    if array.shape[1] > 26:
        df = pd.DataFrame(array)
        df = df.iloc[:, 0:26]
        df.columns = alpha

    else:
        col = alpha[:array.shape[1]]
        df = pd.DataFrame(array, columns=col)

    return df
