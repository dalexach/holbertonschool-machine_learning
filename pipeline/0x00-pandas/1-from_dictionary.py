#!/usr/bin/env python3
"""
Python script that created a pd.DataFrame from a dictionary:
- The first column should be labeled First and have
  the values 0.0, 0.5, 1.0, and 1.5
- The second column should be labeled Second and have
  the values one, two, three, four
- The rows should be labeled A, B, C, and D, respectively
- The pd.DataFrame should be saved into the variable df
"""
import pandas as pd

dic = {
    'First': [0.0, 0.5, 1.0, 1.5],
    'Second': ['one', 'two', 'three', 'four']
}
idx = list('ABCD')

df = pd.DataFrame(dic, index=idx)
