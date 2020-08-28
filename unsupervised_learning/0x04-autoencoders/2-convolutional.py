#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Function that creates a convolutional autoencoder

    Arguments:
     - input_dims is a tuple of integers containing the dimensions
        of the model input
     - filters is a list containing the number of filters for each
        convolutional layer in the encoder, respectively
        * the filters should be reversed for the decoder
     - latent_dims is a tuple of integers containing the dimensions
        of the latent space representation

    Returns:
     encoder, decoder, auto
         - encoder is the encoder model
         - decoder is the decoder model
         - auto is the full autoencoder model
    """
