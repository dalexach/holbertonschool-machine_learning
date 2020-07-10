#!/usr/bin/env python3
"""
Custom layer class TripletLoss
"""

import tensorflow as tf
import tensorflow.keras as K


class TripletLoss(K.layers.Layer):
    """
    TripletLoss class that inherits from tensorflow.keras.layers.Layer
    """

    def __init__(self, alpha, **kwargs):
        """
        Class constructor

        Arguments:
         - alpha is the alpha value used to calculate the triplet loss
         - sets the public instance attribute alpha
        """
        super(TripletLoss, self).__init__(**kwargs)
        self.alpha = alpha

    # Public instance method
    def triplet_loss(self, inputs):
        """
        Public instance method that calculate Triplet Loss

        Arguments:
         - inputs is a list containing the anchor,
            positive and negative output tensors from the last layer
            of the model, respectively

        Returns:
         A tensor containing the triplet loss values
        """

        A, P, N = inputs

        d1 = K.layers.Subtract()([A, P])
        d2 = K.layers.Subtract()([A, N])

        d1sqrt = K.backend.square(d1)
        d2sqrt = K.backend.square(d2)

        d1sum = K.backend.sum(d1sqrt, axis=1)
        d2sum = K.backend.sum(d2sqrt, axis=1)

        total = K.layers.Subtract()([d1sum, d2sum])

        loss = K.backend.maximum(total + self.alpha, 0)

        return loss

    # Public instance method
    def call(self, inputs):
        """
        Public instance method that call Triplet Loss

        Arguments:
         - inputs is a list containing
            the anchor, positive, and negative output tensors from the last
            layer of the model, respectively
         - adds the triplet loss to the graph

        Returns:
         The triplet loss tensor
        """

        loss = self.triplet_loss(inputs)
        self.add_loss(loss)

        return loss
