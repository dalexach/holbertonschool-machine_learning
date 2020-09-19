#!/usr/bin/env python3
"""
Cumulative N-gram BLEU score
"""
import numpy as np


def precitions(references, sentence, n):
    """
    Function to calculate the presition for the n-gram BLEU score for a sentence

    Arguments:
     - references is a list of reference translations
        * each reference translation is a list of the words in the translation
     - sentence is a list containing the model proposed sentence
     - n is the size of the largest n-gram to use for evaluation

    Returns:
     The presition
    """

    len_refer = []
    clipped = {}

    N_sentence = [' '.join([str(jd) for jd in sentence[id:id + n]])
                  for id in range(len(sentence) - (n - 1))]
    len_Noutput = (len(N_sentence))

    for refs in references:
        N_reference = [' '.join([str(jd) for jd in refs[id:id + n]])
                       for id in range(len(sentence) - (n - 1))]

        len_refer.append(len(refs))

        for w in N_reference:
            if w in N_sentence:
                if not clipped.keys() == w:
                    clipped[w] = 1

    clipped_count = sum(clipped.values())

    return clipped_count / len_Noutput
    

def cumulative_bleu(references, sentence, n):
    """
    Function that calculates the cumulative n-gram BLEU score for a sentence

    Arguments:
     - references is a list of reference translations
        * each reference translation is a list of the words in the translation
     - sentence is a list containing the model proposed sentence
     - n is the size of the largest n-gram to use for evaluation

    Returns:
     The cumulative n-gram BLEU score
    """

    len_output = len(sentence)
    precition = [0] * n
    for x in range(0, n):
        precition[x] = precitions(references, sentence, x + 1)

    mean = np.sum(precition) / n
    closest_idx = np.argmin([abs(len(x) - len_output) for x in references])
    closest_len_refer = len(references[closest_idx])

    if len_output > closest_len_refer:
        bp = 1
    else:
        bp = np.exp(1 - (float(closest_len_refer) / len_output))

    BLEU_score = bp * mean

    return BLEU_score
