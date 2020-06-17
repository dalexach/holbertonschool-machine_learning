#!/usr/bin/env python3
"""
Projection Block
"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Function that builds a projection block as described
    in Deep Residual Learning for Image Recognition (2015):

    Arguments:
     - A_prev is the output from the previous layer
     - filters is a tuple or list containing:
        * F11 is the number of filters in the first 1x1 convolution
        * F3 is the number of filters in the 3x3 convolution
        * F12 is the number of filters in the second 1x1 convolution
            as well as the 1x1 convolution in the shortcut connection
     - s is the stride of the first convolution in both
            the main path and the shortcut connection

    Returns:
     The activated output of the projection block
    """
