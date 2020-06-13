#!/usr/bin/env python3
"""
Convolutional Forward Prop
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function that performs forward propagation over a convolutional
    layer of a NN:

    Arguments:
     - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        * m is the number of examples
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
        * c_prev is the number of channels in the previous layer
     - W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        the kernels for the convolution
        * kh is the filter height
        * kw is the filter width
        * c_prev is the number of channels in the previous layer
        * c_new is the number of channels in the output
     - b is a numpy.ndarray of shape (1, 1, 1, c_new) containing
        the biases applied to the convolution
     - activation is an activation function applied to the convolution
     - padding is a string that is either same or valid, indicating
        the type of padding used
     - stride is a tuple of (sh, sw) containing the strides for
        the convolution
        * sh is the stride for the height
        * sw is the stride for the width

    Returns:
     The output of the convolutional layer
    """

    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = W.shape[0]
    kw = W.shape[1]
    c_prev = W.shape[2]
    c_new = W.shape[3]

    sh = stride[0]
    sw = stride[1]

    if padding == 'valid':
        ph = 0
        pw = 0

    if padding == 'same':
        ph = int(np.ceil(((h_prev * sh) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev * sw) - sw + kw - w_prev) / 2))

    hfinal = int(((h_prev - kh + (2 * ph)) / sh) + 1)
    wfinal = int(((w_prev - kw + (2 * pw)) / sw) + 1)
    convoluted = np.zeros((m, hfinal, wfinal, c_new))

    m_A_prev = np.arange(0, m)
    image = np.pad(A_prev,
                   pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode='constant', constant_values=0)

    for h in range(hfinal):
        for w in range(wfinal):
            for cn in range(c_new):
                bias = b[:, :, :, cn]
                data = np.sum(np.multiply(image[m_A_prev,
                                                h*sh:kh+(h*sh),
                                                w*sw:kw+(w*sw)],
                                          W[:, :, :, cn]),
                              axis=(1, 2, 3))
                convoluted[m_A_prev, h, w, cn] = activation((data + bias))

    return convoluted
