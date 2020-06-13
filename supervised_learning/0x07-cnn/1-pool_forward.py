#!/usr/bin/env python3
"""
Pooling Forward Prop
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs forward propagation over a pooling layer of a NN

    Arguments:
     - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
        * m is the number of examples
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
        * c_prev is the number of channels in the previous layer
     - kernel_shape is a tuple of (kh, kw) containing the size of
        the kernel for the pooling
        * kh is the kernel height
        * kw is the kernel width
     - stride is a tuple of (sh, sw) containing the strides for
        the convolution
        * sh is the stride for the height
        * sw is the stride for the width
     - mode is a string containing either max or avg, indicating
        whether to perform maximum or average pooling, respectively

    Returns:
     The output of the pooling layer
    """

    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    hfinal = int(((h_prev - kh) / sh) + 1)
    wfinal = int(((w_prev - kw) / sw) + 1)
    convoluted = np.zeros((m, hfinal, wfinal, c_prev))

    m_A_prev = np.arange(0, m)

    for h in range(hfinal):
        for w in range(wfinal):
            if mode == 'max':
                data = np.max(A_prev[m_A_prev,
                                     h*sh:kh+(h*sh),
                                     w*sw:kw+(w*sw)],
                              axis=(1, 2))
            if mode == 'avg':
                data = np.mean(A_prev[m_A_prev,
                                      h*sh:kh+(h*sh),
                                      w*sw:kw+(w*sw)],
                               axis=(1, 2))
            convoluted[m_A_prev, h, w] = data

    return convoluted
