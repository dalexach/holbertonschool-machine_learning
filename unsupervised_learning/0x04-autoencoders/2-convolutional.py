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
    # Encoder
    iencoder = keras.Input(shape=input_dims)

    output = keras.layers.Conv2D(filters=filters[0],
                                 kernel_size=3,
                                 padding='same',
                                 activation='relu')(iencoder)
    output = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(output)

    for i in range(1, len(filters)):
        output = keras.layers.Conv2D(filters=filters[i],
                                     kernel_size=3,
                                     padding='same',
                                     activation='relu')(output)
        output = keras.layers.MaxPool2D(pool_size=(2, 2),
                                        padding='same')(output)

    oencoder = output

    # Decoder
    idecoder = keras.Input(shape=latent_dims)

    output2 = keras.layers.Conv2D(filters=filters[-1],
                                  kernel_size=3,
                                  padding='same',
                                  activation='relu')(idecoder)
    output2 = keras.layers.UpSampling2D(2)(output2)

    for i in range(len(filters)-2, 0, -1):
        output2 = keras.layers.Conv2D(filters=filters[i],
                                      kernel_size=3,
                                      padding='same',
                                      activation='relu')(output2)
        output2 = keras.layers.UpSampling2D(2)(output2)

    output2 = keras.layers.Conv2D(filters=filters[0],
                                  kernel_size=3,
                                  padding='valid',
                                  activation='relu')(output2)
    output2 = keras.layers.UpSampling2D(2)(output2)

    odecoder = keras.layers.Conv2D(filters=input_dims[-1],
                                   kernel_size=3,
                                   padding='same',
                                   activation='sigmoid')(output2)

    encoder = keras.models.Model(inputs=iencoder, outputs=oencoder)
    decoder = keras.models.Model(inputs=idecoder, outputs=odecoder)

    encoder.summary()
    decoder.summary()
    input_auto = keras.Input(shape=input_dims)
    out_encoder = encoder(input_auto)
    out_decoder = decoder(out_encoder)

    # Autoencoder
    auto = keras.models.Model(inputs=input_auto, outputs=out_decoder)
    auto.compile(optimizer='Adam',
                 loss='binary_crossentropy')

    return encoder, decoder, auto
