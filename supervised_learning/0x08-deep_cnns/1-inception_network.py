#!/usr/bin/env python3
"""
Inception Network
"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Function that builds the inception network as described
    in Going Deeper with Convolutions (2014)

    Returns:
     The keras model
    """
