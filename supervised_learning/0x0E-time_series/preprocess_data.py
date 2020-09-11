#!/usr/bin/env python3
"""
Preprocessing the database
"""
import pandas as pd


def preprocessing(name_file):
    """
    Function to clean tha data from csv

    Arguments:
     - name_file is the name of the file that contains the data

    Returns:
     - train is the train values
     - validation is the validation values
     - test is the test values
    """

    db_data = pd.read_csv(name_file).dropna()
    db_data['Timestamp'] = pd.to_datetime(db_data['Timestamp'], unit='s')
    db_data = db_data[db_data['Timestamp'].dt.year >= 2017]
    db_data.reset_index(inplace=True, drop=True)
    db_data = db_data.drop(['Timestamp'], axis=1)
    db_data = db_data[0::60]

    n = len(db_data)

    # Split data
    train = db_data[0:int(n * 0.7)]
    validation = db_data[int(n * 0.7):int(n * 0.9)]
    test = db_data[int(n * 0.9):]

    # Normalize data
    train_mean = train.mean()
    train_std = train.std()
    train = (train - train_mean) / train_std
    validation = (validation - train_mean) / train_std
    test = (test - train_mean) / train_std

    return train, validation, test
