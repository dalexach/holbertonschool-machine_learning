#!/usr/bin/env python3
"""
Identity Block
"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Function that builds an identity block as described
    in Deep Residual Learning for Image Recognition (2015):

    Arguments:
     - A_prev is the output from the previous layer
     - filters is a tuple or list containing:
        * F11 is the number of filters in the first 1x1 convolution
        * F3 is the number of filters in the 3x3 convolution
        * F12 is the number of filters in the second 1x1 convolution

    Returns:
     The activated output of the identity block
    """
