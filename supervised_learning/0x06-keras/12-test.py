#!/usr/bin/env python3
"""
Test a NN
"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Function that tests a neural network

    Arguments:
     - network is the network model to test
     - data is the input data to test the model with
     - labels are the correct one-hot labels of data
     - verbose is a boolean that determines if output should be printed
        during the testing process

    Returns:
     The loss and accuracy of the model with the testing data, respectively
    """
    testm = network.evaluate(data, labels, verbose=verbose)

    return testm
