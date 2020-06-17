#!/usr/bin/env python3
"""
Transition Layer
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Function that builds a transition layer as described
    in Densely Connected Convolutional Networks:

    Arguments:
     - X is the output from the previous layer
     - nb_filters is an integer representing the number of filters in X
      compression is the compression factor for the transition layer

    Returns:
     The output of the transition layer
     and the number of filters within the output, respectively
    """
