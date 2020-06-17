#!/usr/bin/env python3
"""
DenseNet-121
"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Function that builds the DenseNet-121 architecture as described
    in Densely Connected Convolutional Networks:

    Arguments:
     - growth_rate is the growth rate
     - compression is the compression factor

    Returns:
     The keras model
    """
