#!/usr/bin/env python3
"""
Indexing script
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

"""
Requirements:
Index the pd.DataFrame on the Timestamp column
"""
df = df.set_index('Timestamp')

print(df.tail())