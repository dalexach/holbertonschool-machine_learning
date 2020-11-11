#!/usr/bin/env python3
"""
Flip function
"""
import tensorflow as tf


def flip_image(image):
    """
    Function that flips an image horizontally

    Arguments:
     - image is a 3D tf.Tensor containing the image to flip

    Returns:
     The flipped image
    """

    fliped = tf.image.flip_left_right(image)

    return fliped
