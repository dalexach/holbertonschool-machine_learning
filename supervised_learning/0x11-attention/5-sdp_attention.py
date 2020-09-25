#!/usr/bin/env python3
"""
Scaled Dot Product Attention Function
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Function that calculates the scaled dot product attention

    Arguments:
    - Q is a tensor with its last two dimensions as (..., seq_len_q, dk)
        containing the query matrix
    - K is a tensor with its last two dimensions as (..., seq_len_v, dk)
        containing the key matrix
    - V is a tensor with its last two dimensions as (..., seq_len_v, dv)
        containing the value matrix
    - mask is a tensor that can be broadcast into(..., seq_len_q, seq_len_v)
        containing the optional mask, or defaulted to None
        * if mask is not None, multiply -1e9 to the mask and add it to
         the scaled matrix multiplication

    Returns:
     output, weights
     - output a tensor with its last two dimensions as (..., seq_len_q, dv)
        containing the scaled dot product attention
     - weights tensor with its last two dimensions as
        (..., seq_len_q, seq_len_v) containing the attention weights
    """

    qk_dot = tf.matmul(Q, K, transpose_b=True)
    dk_square = tf.cast((tf.math.square(K.shape[-1])), tf.float32)
    qk_scale = tf.math.divide(qk_dot, dk_square)

    if mask is not None:
        mask_multiply = tf.math.multiply(mask, -1e9)
        qk_scale += mask_multiply

    weights = tf.nn.softmax(qk_scale, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
