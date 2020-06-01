#!/usr/bin/env python3
"""
Buiding a model with Keras using Input
"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Function that builds a neural network with the Keras library
    Arguments:
     - nx is the number of input features to the network
     - layers is a list containing the number of nodes in each layer
        of the network
     - activations is a list containing the activation functions used for
        each layer of the network
     - lambtha is the L2 regularization parameter
     - keep_prob is the probability that a node will be kept for dropout
    Returns:
     The keras model
    """
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)
    outputs = K.layers.Dense(layers[0], input_shape=(nx,),
                             activation=activations[0],
                             kernel_regularizer=regularizer,
                             name='dense')(inputs)

    for layer in range(1, len(layers)):
        dname = 'dense_' + str(layer)
        dropout = K.layers.Dropout(rate=(1 - keep_prob))(outputs)
        outputs = K.layers.Dense(layers[layer],
                                 activation=activations[layer],
                                 kernel_regularizer=regularizer,
                                 name=dname)(dropout)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model
