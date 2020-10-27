#!/usr/bin/env python3
"""
Fill script
"""
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
"""
Fill in the missing data points in the pd.DataFrame
- The column Weighted_Price should be removed
- missing values in High, Low, Open, and Close should be
  set to the previous row’s Close value
- missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
"""
# Removing the column
df = df.drop(columns=['Weighted_Price'])
# Setting the missing values to the previous Close values
cols = ['High', 'Low', 'Open']
df['Close'].fillna(method='ffill', inplace=True)

for i in cols:
    df[i].fillna(value=df.Close.shift(1, axis=0), inplace=True)
# Setting the missing values to 0
df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)

print(df.head())
print(df.tail())
