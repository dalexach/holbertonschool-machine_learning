#!/usr/bin/env python3
"""
Hue function
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Function that changes the hue of an image
    Arguments:
     - image is a 3D tf.Tensor containing the image to change
     - delta is the amount the hue should change

    Returns:
     The altered image
    """
