#!/usr/bin/env python3
"""
Rename and display
"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
"""
Requirements:
- Rename the column Timestamp to Datetime
- Convert the timestamp values to datatime values
- Display only the Datetime and Close columns
"""
# Rename column
df = df.rename(columns={'Timestamp': 'Datetime'})
# Converting to datatime values
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
# Only displays the Datetime and Close columns
df = df.loc[:, ['Datetime', 'Close']]

print(df.tail())
