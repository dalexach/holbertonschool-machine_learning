#!/usr/bin/env python3
"""
Sparse Autoencoder
"""
import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Function that creates a sparse autoencoder

    Arguments:
     - input_dims is an integer containing the dimensions of the model input
     - hidden_layers is a list containing the number of nodes for each
        hidden layer in the encoder, respectively
        * the hidden layers should be reversed for the decoder
     - latent_dims is an integer containing the dimensions of
        the latent space representation
     - lambtha is the regularization parameter used for L1 regularization
        on the encoded output

    Returns:
     encoder, decoder, auto
        - encoder is the encoder model
        - decoder is the decoder model
        - auto is the sparse autoencoder model
    """
