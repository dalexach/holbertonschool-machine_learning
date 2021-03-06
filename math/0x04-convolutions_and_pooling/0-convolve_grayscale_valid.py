#!/usr/bin/env python3
"""
Valid Convolution
"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images:

    Arguments:
     - images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images
         * m is the number of images
         * h is the height in pixels of the images
         * w is the width in pixels of the images
     - kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
        the convolution
        * kh is the height of the kernel
        * kw is the width of the kernel

    Returns:
    A numpy.ndarray containing the convolved images
    """

    m = images.shape[0]
    himage = images.shape[1]
    wimage = images.shape[2]
    hkernel = kernel.shape[0]
    wkernel = kernel.shape[1]

    hfinal = himage - hkernel + 1
    wfinal = wimage - wkernel + 1
    convoluted = np.zeros((m, hfinal, wfinal))

    mImage = np.arange(0, m)
    for i in range(hfinal):
        for j in range(wfinal):
            data = np.sum(np.multiply(images[mImage, i:hkernel+i, j:wkernel+j],
                                      kernel), axis=(1, 2))
            convoluted[mImage, i, j] = data

    return convoluted
