#!/usr/bin/env python3
"""
Class TrainModel that trains a model
for face verification using triplet loss
"""


import numpy as np
import tensorflow as tf
from triplet_loss import TripletLoss


class TrainModel:
    """
    Class TrainModel that trains a model
    for face verification using triplet loss
    """

    def __init__(self, model_path, alpha):
        """
        Class constructor

        Arguments:
         - model_path is the path to the base face verification embedding model
            * loads the model using with
                tf.keras.utils.CustomObjectScope({'tf': tf})
            * saves this model as the public instance method base_model
         - alpha is the alpha to use for the triplet loss calculation
        """

        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = tf.keras.models.load_model(model_path)

        A = tf.keras.Input(shape=(96, 96, 3))
        P = tf.keras.Input(shape=(96, 96, 3))
        N = tf.keras.Input(shape=(96, 96, 3))

        netw1 = self.base_model(A)
        netw2 = self.base_model(P)
        netw3 = self.base_model(N)

        tl = TripletLoss(alpha)

        nall = [netw1, netw2, netw3]
        out = tl(nall)

        model = tf.keras.models.Model([A, P, N], out)
        model.compile(optimizer='adam')
        self.training_model = model

    # Public instance method
    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        """
        Public instance method that trains self.training_model

        Arguments:
         - triplets is a list of numpy.ndarrayscontaining
            the inputs to self.training_model
         - epochs is the number of epochs to train for
         - batch_size is the batch size for training
         - validation_split is the validation split for training
         - verbose is a boolean that sets the verbosity mode

        Returns:
         The History output from the training
        """

        history = self.training_model.fit(triplets,
                                          batch_size=batch_size,
                                          verbose=verbose,
                                          validation_split=validation_split)

        return history

    # Public instance method

    def save(self, save_path):
        """
        Public instance method that saves the base embedding model:

        Arguments:
         - save_path is the path to save the model

        Returns:
         The saved model
        """

        tf.keras.models.save_model(self.base_model, save_path)

        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Static method that calculates the F1 score of predictions

        Arguments:
         - y_true - a numpy.ndarray of shape (m,) containing the correct labels
            * m is the number of examples
         - y_pred- a numpy.ndarray of shape (m,)
            containing the predicted labels

        Returns:
         The f1 score
        """

        TP = np.count_nonzero(y_pred * y_true)
        TN = np.count_nonzero((y_pred - 1) * (y_true - 1))
        FP = np.count_nonzero(y_pred * (y_true - 1))
        FN = np.count_nonzero((y_pred - 1) * y_true)

        if TP + FP == 0:
            return 0
        else:
            presicion = TP / (TP + FP)

        if TP + FN == 0:
            return 0
        else:
            recall = TP / (TP + FN)

        F1 = 2 * presicion * recall / (presicion + recall)

        return F1

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Static method that calculates the accuracy

        Arguments:
         - y_true - a numpy.ndarray of shape (m,) containing the correct labels
            * m is the number of examples
         - y_pred- a numpy.ndarray of shape (m,)
            containing the predicted labels

        Returns:
         The accuracy
        """

        TP = np.count_nonzero(y_pred * y_true)
        TN = np.count_nonzero((y_pred - 1) * (y_true - 1))
        FP = np.count_nonzero(y_pred * (y_true - 1))
        FN = np.count_nonzero((y_pred - 1) * y_true)

        accur = (TP + TN) / (TP + FN + TN + FP)

        return accur

    # Public instance method
    def best_tau(self, images, identities, thresholds):
        """
        Public instance methid that calculates
        the best tau to use for a maximal F1 score

        Arguments:
         - images - a numpy.ndarray of shape (m, n, n, 3)
            containing the aligned images for testing
            * m is the number of images
            * n is the size of the images
         - identities - a list containing the identities
            of each image in images
         - thresholds - a 1D numpy.ndarray of distance thresholds (tau) to test

        Returns:
         (tau, f1, acc)
            - tau- the optimal threshold to maximize F1 score
            - f1 - the maximal F1 score
            - acc - the accuracy associated with the maximal F1 score
        """

        em = np.zeros((images.shape[0], 128))

        for i, img in em:
            em[i] = self.basemode.predict(np.expand_dims(img, axis=0))[0]

        dist = []
        identical = []

        length = len(identities)

        for i in range(length - 1):
            for j in range(i + 1, length):
                dist.append(np.sum(np.square(em[i] - em[j])))
                if identities[i] == identities[j]:
                    identical.append(1)
                else:
                    identical.append(0)

        dist = np.array(dist)
        identical = np.array(identical)

        F1 = [self.f1_score(identical, dist < t) for t in thresholds]
        acc = [self.accuracy(identical, dist < t) for t in thresholds]
        idx = np.argmax(F1)
        tau = thresholds[idx]
        opt_F1 = F1[idx]
        opt_acc = acc[idx]

        return tau, opt_F1, opt_acc
