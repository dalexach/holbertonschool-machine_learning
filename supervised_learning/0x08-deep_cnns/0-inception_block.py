#!/usr/bin/env python3
"""
Inception Block
"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Function that that builds an inception block
    as described in Going Deeper with Convolutions (2014).

    Arguments:
     - A_prev is the output from the previous layer
     - filters is a tuple or list containing:
        * F1 is the number of filters in the 1x1 convolution
        * F3R is the number of filters in the 1x1 convolution before
            the 3x3 convolution
        * F3 is the number of filters in the 3x3 convolution
        * F5R is the number of filters in the 1x1 convolution before
            the 5x5 convolution
        * F5 is the number of filters in the 5x5 convolution
        * FPP is the number of filters in the 1x1 convolution after
            the max pooling

    Returns:
     The concatenated output of the inception block
    """
