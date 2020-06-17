#!/usr/bin/env python3
"""
Dense Block
"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Function that builds a dense block as described
    in Densely Connected Convolutional Networks:

    Arguments:
     - X is the output from the previous layer
     - nb_filters is an integer representing the number of filters in X
     - growth_rate is the growth rate for the dense block
     - layers is the number of layers in the dense block

    Returns:
     The concatenated output of each layer within the Dense Block
     and the number of filters within the concatenated outputs, respectively
    """
