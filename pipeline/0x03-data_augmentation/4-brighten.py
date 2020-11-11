#!/usr/bin/env python3
"""
Brightness function
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Function that randomly changes the brightness of an image

    Arguments:
     - image is a 3D tf.Tensor containing the image to change
     - max_delta is the maximum amount the image
        should be brightened (or darkened)

    Returns:
     The altered image
    """

    brightness = tf.image.adjust_brightness(image, max_delta)

    return brightness
