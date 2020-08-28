#!/usr/bin/env python3
"""
Variational Autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates a variational autoencoder

    Arguments:
     - input_dims is an integer containing the dimensions of the model input
     - hidden_layers is a list containing the number of nodes for each
        hidden layer in the encoder, respectively
        * the hidden layers should be reversed for the decoder
     - latent_dims is an integer containing the dimensions of the latent
        space representation

    Returns:
     encoder, decoder, auto
        - encoder is the encoder model, which should output
            the latent representation, the mean, and the log variance,
            respectively
        - decoder is the decoder model
        - auto is the full autoencoder model
    """
    # Encoder
    iencoder = keras.Input(shape=(input_dims, ))

    output = keras.layers.Dense(hidden_layers[0], activation='relu')(iencoder)

    for i in range(1, len(hidden_layers)):
        output = keras.layers.Dense(hidden_layers[i],
                                    activation='relu')(output)

    z_mean = keras.layers.Dense(latent_dims)(output)
    z_var = keras.layers.Dense(latent_dims)(output)

    def sampling(args):
        """
        Sampling the data from the data set using the z_mean and z_stand_dev
        """
        z_mean, z_var = args
        m = keras.backend.shape(z_mean)[0]
        dims = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(m, dims))

        return z_mean + keras.backend.exp(0.5 * z_var) * epsilon

    z = keras.layers.Lambda(sampling,
                            output_shape=(latent_dims,))([z_mean, z_var])

    # Decoder
    idecoder = keras.Input(shape=(latent_dims, ))
    output2 = keras.layers.Dense(hidden_layers[-1],
                                 activation='relu')(idecoder)

    for i in range(len(hidden_layers)-2, -1, -1):
        output2 = keras.layers.Dense(hidden_layers[i],
                                     activation='relu')(output2)
    odecoder = keras.layers.Dense(input_dims, activation='sigmoid')(output2)

    encoder = keras.models.Model(inputs=iencoder, outputs=[z, z_mean, z_var])
    decoder = keras.models.Model(inputs=idecoder, outputs=odecoder)

    out_encoder = encoder(iencoder)[0]
    out_decoder = decoder(out_encoder)
    # Autoencoder
    auto = keras.models.Model(inputs=iencoder, outputs=out_decoder)

    encoder.summary()
    decoder.summary()
    auto.summary()

    def loss(y_in, y_out):
        """
        Custom loss function
        """
        y_loss = keras.backend.binary_crossentropy(y_in, y_out)
        y_loss = keras.backend.sum(y_loss, axis=1)
        kl_loss = (1 + z_var - keras.backend.square(z_mean)
                   - keras.backend.exp(z_var))
        kl_loss = -0.5 * keras.backend.sum(kl_loss, axis=1)

        return y_loss + kl_loss

    auto.compile(optimizer='Adam', loss=loss)

    return encoder, decoder, auto
