#!/usr/bin/env python3
"""
Visualize script to visualize the pd.DataFrame
"""
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

def open_d(array):
    return array[0]


def close_d(array):
    return array[-1]

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

"""
Requirements:
Visualize the pd.DataFrame:
- Plot the data from 2017 and beyond at daily intervals
- The column Weighted_Price should be removed
- Rename the column Timestamp to Date
- Convert the timestamp values to date values
- Index the data frame on Date
- Missing values in High, Low, Open, and Close should be set to
  the previous row’s Close value
- Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
"""
# Removing the column Weighted_Price
df = df.drop('Weighted_Price', axis=1)

# Rename column
df = df.rename(columns={'Timestamp': 'Date'})

# Converting to datatime values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Taking the data from 2017
df = df[df['Date'] >= '2017-01-01']

# Setting Date as index
df = df.set_index('Date')

# Setting the missing values to the previous Close values
cols = ['High', 'Low', 'Open']
df['Close'].fillna(method='ffill', inplace=True)

for i in cols:
    df[i].fillna(value=df.Close.shift(1, axis=0), inplace=True)

# Setting the missing values to 0
df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)

# Resampling seconds to days
rdf = pd.DataFrame()
rdf['Open'] = df.Open.resample('D').apply(open_d)
rdf['High'] = df.High.resample('D').max()
rdf['Low'] = df.Low.resample('D').min()
rdf['Close'] = df.Close.resample('D').apply(close_d)
rdf['Volume_(BTC)'] = df['Volume_(BTC)'].resample('D').sum()
rdf['Volume_(Currency)'] = df['Volume_(Currency)'].resample('D').sum()

# Plotting data
rdf.plot()
plt.show()
