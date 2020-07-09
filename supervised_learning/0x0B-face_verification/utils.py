#!/usr/bin/env python3
"""
Loading images
"""

import csv
import cv2
import glob
import numpy as np
import os


def load_images(images_path, as_array=True):
    """
    Function that loads images from a directory or file

    Arguments:
     - images_path is the path to a directory from which to load images
     - as_array is a boolean indicating whether the images should be
        loaded as one numpy.ndarray
        If True, the images should be loaded as a numpy.ndarray
        of shape (m, h, w, c) where:
           * m is the number of images
           * h, w, and c are the height, width, and number of channels
           of all images, respectively
        If False, the images should be loaded as a list of
        individual numpy.ndarrays

    Returns:
     images, filenames
        - images is either a list/numpy.ndarray of all images
        - filenames is a list of the filenames associated with
            each image in images
    """

    images = []
    filenames = []
    image_paths = glob.glob(images_path + '/*')

    for image in sorted(image_paths):
        img_read = cv2.imread(image)
        img_rgb = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
        filename = image.split('/')[-1].strip()
        filenames.append(filename)

    if as_array is True:
        images = np.stack(images)

    return (images, filenames)
