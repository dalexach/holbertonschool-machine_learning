#!/usr/bin/env python3
"""
PCA Color Augmentation
"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Function that performs PCA color augmentation as described in the AlexNet paper

    Arguments:
     - image is a 3D tf.Tensor containing the image to change
     - alphas a tuple of length 3 containing the amount that each channel should change

    Returns:
     The augmented image
    """
