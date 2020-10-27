#!/usr/bin/env python3
"""
From File
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Function that loads data from a file as a pd.DataFrame

    Arguments:
     - filename is the file to load from
     - delimiter is the column separator

    Returns:
     The loaded pd.DataFrame
    """

    df = pd.read_csv(filename, sep=delimiter)

    return df
