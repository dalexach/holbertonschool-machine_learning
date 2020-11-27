#!/usr/bin/env python3
"""
Question Answerings
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Fucntion that finds a snippet of text
    within a reference document to answer a question

    Argumetns:
     - question is a string containing the question to answer
     - reference is a string containing the reference document
        from which to find the answer

    Returns:
     A string containing the answer. If no answer is found, return None
    """

    p = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(p)
    model = hub.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')
    t_question = tokenizer.tokenize(question)
    t_reference = tokenizer.tokenize(reference)

    s1 = ['[CLS]']
    s2 = ['[SEP]']
    tokens = s1 + t_question + s2 + t_reference + s2

    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(t_question) + 1) +\
                     [1] * (len(t_reference) + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
                             (input_word_ids, input_mask, input_type_ids))

    outputs = model([input_word_ids, input_mask, input_type_ids])
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1


    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer if len(answer) > 1 else None
