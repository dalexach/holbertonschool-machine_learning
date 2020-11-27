#!/usr/bin/env python3
"""
Answer Questions
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Function that answers questions from a reference text

    Arguments:
     - reference is the reference text

    Note:
    If the answer cannot be found in the reference text,
    respond with Sorry, I do not understand your question.
    """

    cases = ['exit', 'goodbye', 'bye']

    while True:
        question = input('Q: ')
        question = question.lower()

        if question in cases:
            print('A: Goodbye')
            exit(0)
        else:
            answer = question_answer(question, reference)
            if answer is None or answer == '':
                print('A: Sorry, I do not understand your question.')
            else:
                print('A: {}'.format(answer))
