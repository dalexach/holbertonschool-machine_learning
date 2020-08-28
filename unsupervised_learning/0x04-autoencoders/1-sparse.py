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
    # Encoder
    iencoder = keras.Input(shape=(input_dims, ))

    output = keras.layers.Dense(hidden_layers[0],
                                activation='relu', )(iencoder)

    for i in range(1, len(hidden_layers)):
        output = keras.layers.Dense(hidden_layers[i],
                                    activation='relu')(output)
    reg = keras.regularizers.l1(lambtha)
    oencoder = keras.layers.Dense(latent_dims,
                                  activation='relu',
                                  activity_regularizer=reg)(output)

    # Decoder
    idecoder = keras.Input(shape=(latent_dims, ))
    output2 = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(idecoder)

    for i in range(len(hidden_layers)-2, -1, -1):
        output2 = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(output2)
    odecoder = keras.layers.Dense(input_dims,
                                  activation='sigmoid',)(output2)

    encoder = keras.models.Model(inputs=iencoder, outputs=oencoder)
    decoder = keras.models.Model(inputs=idecoder, outputs=odecoder)

    out_encoder = encoder(iencoder)
    out_decoder = decoder(out_encoder)

    # Autoencoder
    auto = keras.models.Model(inputs=iencoder, outputs=out_decoder)
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
