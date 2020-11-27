#!/usr/bin/env python3
"""
Semantic Search
"""
import numpy as np
import os
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """
    Function that performs semantic search on a corpus of documents

    Arguments:
     - corpus_path is the path to the corpus of reference documents
        on which to perform semantic search
     - sentence is the sentence from which to perform semantic search

    Returns:
     The reference text of the document most similar to sentence
    """

    articles = [sentence]

    for fn in os.listdir(corpus_path):
        if not fn.endswith('.md'):
            continue
        with open(corpus_path + '/' + fn, 'r', encoding='utf-8') as file:
            articles.append(file.read())

    embed = hub.load(
        'https://tfhub.dev/google/universal-sentence-encoder-large/5')
    embeddings = embed(articles)

    corr = np.inner(embeddings, embeddings)
    close = np.argmax(corr[0, 1:])

    return articles[close + 1]
