#!/usr/bin/env python3
"""
Extract Word2Vec
"""
import tensorflow.keras as K
from gensim.models import Word2Vec


def gensim_to_keras(model):
    """
    Function that converts a gensim word2vec model to a keras Embedding layer

    Arguments:
     - model is a trained gensim word2vec models

    Returns:
     The trainable keras Embedding
    Note: you need keras=2.2.2 to run get_keras_embedding()
    """

    return model.wv.get_keras_embedding(train_embeddings=True)
