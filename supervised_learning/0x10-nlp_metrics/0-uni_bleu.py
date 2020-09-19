#!/usr/bin/env python3
"""
Unigram BLEU score
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Function that calculates the unigram BLEU score for a sentence

    Arguments:
     - references is a list of reference translations
        * each reference translation is a list of the words in the translation
     - sentence is a list containing the model proposed sentence

    Returns:
     The unigram BLEU score
    """

    len_output = len(sentence)
    counter = 0
    len_clipped = 0

    for w in sentence:
        counter += sentence.count(w)
        maxi = 0
        for ref in references:
            tmp = ref.count(w)
            if tmp > maxi:
                maxi = tmp
        len_clipped += maxi

    closest_idx = np.argmin([abs(len(x) - len_output) for x in references])
    closest_len_refer = len(references[closest_idx])

    if len_output > closest_len_refer:
        bp = 1
    else:
        bp = np.exp(1 - closest_len_refer / len_output)

    BLEU_score = bp * (len_clipped / counter)

    return BLEU_score
