#!/usr/bin/env python3
"""
PCA Color Augmentation
"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    Function that performs PCA color augmentation
    as described in the AlexNet paper

    Arguments:
     - image is a 3D tf.Tensor containing the image to change
     - alphas a tuple of length 3 containing the amount
        that each channel should change

    Returns:
     The augmented image
    """

    img = tf.keras.preprocessing.image.img_to_array(image)
    init_img = img.astype(float).copy()

    # Rescaling to 0 to 1 range
    img = img / 255.0

    re_img = img.reshape(-1, 3)

    center_img = re_img - np.mean(re_img, axis=0)
    cov_img = np.cov(center_img, rowvar=False)
    e_val, e_vect = np.linalg.eig(cov_img)

    # Sorting values and vector
    sort_perm = e_val[::-1].argsort()
    e_val[::-1].sort()
    e_vect = e_vect[:, sort_perm]

    m1 = np.column_stack((e_vect))
    m2 = np.zeros((3, 1))
    m2[:, 0] = alphas * e_val[:]
    vect = np.matrix(m1) * np.matrix(m2)

    for i in range(3):
        init_img[..., i] += vect[i]

    init_img = np.clip(init_img, 0.0, 255.0)
    init_img = init_img.astype(np.uint8)

    return init_img
