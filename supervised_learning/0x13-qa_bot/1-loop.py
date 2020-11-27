#!/usr/bin/env python3
"""
Create the loop
Script that takes in input from the user with the prompt Q:
and prints A: as a response.
If the user inputs exit, quit, goodbye, or bye, case insensitive,
print A: Goodbye and exit
"""

cases = ['exit', 'goodbye', 'bye']

while True:
    ans = input('Q: ')
    ans = ans.lower()

    if ans in cases:
        print('A: Goodbye')
        exit(0)
    else:
        print('A: ')
