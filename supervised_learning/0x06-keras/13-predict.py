#!/usr/bin/env python3
"""
Prediction using NN
"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Function that makes a prediction using a NN

    Arguments:
     - network is the network model to make the prediction with
     - data is the input data to make the prediction with
     - verbose is a boolean that determines if output should be printed
        during the prediction process

    Returns:
     The prediction for the data
    """
    predicted = network.predict(x=data, verbose=verbose)

    return predicted
