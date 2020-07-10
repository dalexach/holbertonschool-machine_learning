#!/usr/bin/env python3
"""
FaceVerification class
"""

import numpy as np
import tensorflow as tf


class FaceVerification:
    """
    FaceVerification class
    """

    def __init__(self, model_path, database, identities):
        """
        Class constructor

        Argumetns:
         - model_path is the path to where the face verification embedding
            model is stored
         - database is a numpy.ndarray of shape (d, e) containing all
            the face embeddings in the database
           * d is the number of images in the database
           * e is the dimensionality of the embedding
         - identities is a list of length d containing the identities
            corresponding to the embeddings in database
        """

        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.model = tf.keras.models.load_model(model)

        self.database = database
        self.identities = identities

    # Public instance method
    def embedding(self, images):
        """
        Public instance method that calculates the face embedding of images

        Arguments:
         - images is a numpy.ndarray of shape (i, n, n, 3)
            containing the aligned images
           * i is the number of images
           * n is the size of the aligned images

        Returns:
         A numpy.ndarray of shape (i, e)
         containing the embeddings where e is the dimensionality
         of the embeddings
        """

        em = np.zeros((images.shape[0], 128))

        for i, img in enumerate(images):
            em[i] = self.model.predict(np.expand_dims(img, axis=0))[0]

        return np.array(em)

    # Public instance method
    def verify(self, image, tau=0.5):
        """
        Public instance method

        Arguments:
         - image is a numpy.ndarray of shape (n, n, 3)
            containing the aligned image of the face to be verify
           * n is the shape of the aligned image
         - tau is the maximum euclidean distance used for verification

        Returns:
         (identity, distance), or (None, None) on failure
            - identity is a string containing the identity of the verified face
            - distance is the euclidean distance between the verified face
                embedding and the identified database embedding
        """

        em = self.model.predict(np.expand_dims(image, axis=0))[0]
        dist = []
        lenght = len(self.identities)

        for i in range(lenght):
            dist.append(np.sum(np.square(em, self.database[i])))

        dist = np.array(dist)
        idx = np.argmin(dist)

        if dist[idx] < tau:
            return self.identities[idx], dist[idx]
        else:
            return None, None
