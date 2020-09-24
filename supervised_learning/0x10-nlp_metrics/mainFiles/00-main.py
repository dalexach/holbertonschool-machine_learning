#!/usr/bin/env python3
import numpy as np
uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["hello", "there"]]
sentence = ["hello", "there"]
print(np.round(uni_bleu(references, sentence)), 10)
