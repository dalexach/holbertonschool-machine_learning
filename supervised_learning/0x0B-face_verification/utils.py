#!/usr/bin/env python3
"""
Some special functions
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

    img_path = glob.glob(images_path + '/*')
    img_path = sorted([i for i in img_path])

    filenames = [path.split('/')[-1] for path in img_path]

    img_bgr = [cv2.imread(image) for image in img_path]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_bgr]

    if as_array:
        images = np.array(images)

    return images, filenames


def load_csv(csv_path, params={}):
    """
    Function that loads the contents of a csv file as a list of lists

    Arguments:
     - csv_path is the path to the csv to load
     - params are the parameters to load the csv with

    Returns:
     A list of lists representing the contents found in csv_path
    """

    csv_list = []

    with open(csv_path, 'r') as csvf:
        csv_read = csv.reader(csvf, params)
        for row in csv_read:
            csv_list.append(row)

    return csv_list


def save_images(path, images, filenames):
    """
    Function that saves images to a specific path:

    Arguments:
     - path is the path to the directory in which the images should be saved
     - images is a list/numpy.ndarray of images to save
     - filenames is a list of filenames of the images to save

    Returns:
     True on success and False on failure
     """
    if os.path.exists(path):
        for img, filename in zip(images, filenames):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite('./' + path + '/' + filename, image)

        return True

    else:
        return False


def generate_triplets(images, filenames, triplet_names):
    """
    Function that generates triplets

    Arguments:
     - images is a numpy.ndarray of shape (i, n, n, 3) containing
        the aligned images in the dataset
        * i is the number of images
        * n is the size of the aligned images
     - filenames is a list of length i containing the corresponding
        filenames for images
     - triplet_names is a list of length m of lists where each sublist
        contains the filenames of an anchor, positive,
        and negative image, respectively
        * m is the number of triplets

    Returns:
     A list [A, P, N]:
        - A is a numpy.ndarray of shape (m, n, n, 3) containing
            the anchor images for all m triplets
        - P is a numpy.ndarray of shape (m, n, n, 3) containing
            the positive images for all m triplets
        - N is a numpy.ndarray of shape (m, n, n, 3) containing
            the negative images for all m triplets
    """

    A, P, N = [], [], []

    filename = [filename.split('.')[0] for filename in filenames]

    for tn in triplet_names:
        try:
            idx_A = filename.index(tn[0])
            idx_P = filename.index(tn[1])
            idx_N = filename.index(tn[2])

            A.append(images[idx_A])
            P.append(images[idx_P])
            N.append(images[idx_N])
        except ValueError:
            pass

    return [A, P, N]
