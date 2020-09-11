# [holbertonschool-machine_learning](https://github.com/dalexach/holbertonschool-machine_learning)

## 0x0E. Time Series Forecasting
### Description 
Bitcoin (BTC) became a trending topic after its price peaked in 2018. Many have sought to predict its value in order to accrue wealth. Let’s attempt to use our knowledge of RNNs to attempt just that.

Given the coinbase and bitstamp datasets, write a script, forecast_btc.py, that creates, trains, and validates a keras model for the forecasting of BTC:

- Your model should use the past 24 hours of BTC data to predict the value of BTC at the close of the following hour (approximately how long the average transaction takes):
- The datasets are formatted such that every row represents a 60 second time window containing:
    - The start time of the time window in Unix time
    - The open price in USD at the start of the time window
    - The high price in USD within the time window
    - The low price in USD within the time window
    - The close price in USD at end of the time window
    - The amount of BTC transacted in the time window
    - The amount of Currency (USD) transacted in the time window
    - The volume-weighted average price in USD for the time window
- Your model should use an RNN architecture of your choosing
- Your model should use mean-squared error (MSE) as its cost function
- You should use a tf.data.Dataset to feed data to your model


### Files

| File | Description |
| ------ | ------ |
| [forecast_btc.py](forecast_btc.py) | Script that creates, trains, and validates a keras model for the forecasting of BTC. |
| [preprocess_data.py](preprocess_data.py) | Script to preprocess this data. |


### Build with
- Python (python 3.5)
- Numpy (numpy 1.15)
- Ubuntu 16.04 LTS 

## Author

[Daniela Chamorro](https://www.linkedin.com/in/dalexach/) [:octocat:](https://github.com/dalexach)

[Twitter](https://twitter.com/dalexach)