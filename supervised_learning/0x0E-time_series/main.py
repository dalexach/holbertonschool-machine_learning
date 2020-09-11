#!/usr/bin/env python3
"""
Main file to test the training
"""

import numpy as np
import tensorflow.keras as K
import pandas as pd
import datetime as dt
import tensorflow as tf
preprocess = __import__('preprocess_data').preprocessing
forecast = __import__('forecast_btc').forecasting

if __name__ == "__main__":
    """
    Starting the process
    """
    file_path = '../data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    train, validation, test = preprocess(file_path)
    """
    print('Train values: ')
    print(train)
    print('Validation values:')
    print(validation)
    print('Test values')
    print(test)
    """
    forecast(train, validation, test)
