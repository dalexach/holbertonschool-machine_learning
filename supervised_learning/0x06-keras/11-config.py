#!/usr/bin/env python3
"""
Save and load Configuration
"""


import tensorflow.keras as K


def save_config(network, filename):
    """
    Function that saves a model’s configuration in JSON format

    Arguments:
     - network is the model whose configuration should be saved
     - filename is the path of the file that the configuration
        should be saved to

    Returns:
     None
    """
    jmodel = network.to_json()
    with open(filename, 'w') as jfile:
        jfile.write(jmodel)

    return None


def load_config(filename):
    """
    Function that loads a model with a specific configuration

    Arguments:
     - filename is the path of the file containing the model’s configuration
        in JSON format

    Returns:
     The loaded model
    """
    with open(filename, 'r') as jfile:
        jmodel = K.models.model_from_json(jfile.read())

    return jmodel
