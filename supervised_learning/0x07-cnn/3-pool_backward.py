#!/usr/bin/env python3
"""
Pooling Back Prop
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function 

    Arguments:
     - dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
        the partial derivatives with respect to the output of the pooling layer
        * m is the number of examples
        * h_new is the height of the output
        * w_new is the width of the output
        * c_new is the number of channels in the output
     - A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c)
        containing the output of the previous layer
        * h_prev is the height of the previous layer
        * w_prev is the width of the previous layer
     - kernel_shape is a tuple of (kh, kw) containing the size of the kernel
        for the pooling
        * kh is the kernel height
        * kw is the kernel width
     - stride is a tuple of (sh, sw) containing the strides for
        the convolution
        * sh is the stride for the height
        * sw is the stride for the width
     - mode is a string containing either max or avg, indicating whether
        to perform maximum or average pooling, respectively

    Returns:
     The partial derivatives with respect to the previous layer (dA_prev)
    """

    m = dA.shape[0]
    h_new = dA.shape[1]
    w_new = dA.shape[2]
    c_new = dA.shape[3]

    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c = A_prev.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    dx = np.zeros_like(A_prev)
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for cn in range(c_new):
                    if mode == 'max':
                        aux = A_prev[i,
                                     h*sh:kh+(h*sh),
                                     w*sw:kw+(w*sw),
                                     cn]
                        mask = (aux == np.max(aux))
                        dx[i,
                           h*sh:kh+(h*sh),
                           w*sw:kw+(w*sw),
                           cn] += dA[i, h, w, cn] * mask
                    if mode == 'avg':
                        dx[i,
                           h*sh:kh+(h*sh),
                           w*sw:kw+(w*sw),
                           cn] += (dA[i, h, w, cn])/kh/kw

    return dx
