# [holbertonschool-machine_learning](https://github.com/dalexach/holbertonschool-machine_learning)

## 0x00. Pandas
### Description 

- What is pandas?
- What is a pd.DataFrame? How do you create one?
- What is a pd.Series? How do you create one?
- How to load data from a file
- How to perform indexing on a pd.DataFrame
- How to use hierarchical indexing with a pd.DataFrame
- How to slice a pd.DataFrame
- How to reassign columns
- How to sort a pd.DataFrame
- How to use boolean logic with a pd.DataFrame
- How to merge/concatenate/join pd.DataFrames
- How to get statistical information from a pd.DataFrame
- How to visualize a pd.DataFrame


### Files
#### Mandatory Tasks

| File | Description |
| ------ | ------ |
| [0-from_numpy.py](0-from_numpy.py) | Function from_numpy that creates a pd.DataFrame from a np.ndarray. |
| [1-from_dictionary.py](1-from_dictionary.py) | Script that created a pd.DataFrame from a dictionary. |
| 2-from_file.py](2-from_file.py) | Function from_file that loads data from a file as a pd.DataFrame. |
| [3-rename.py](3-rename.py) | Complete the script below to perform the following: a)Rename the column Timestamp to Datetime. b)Convert the timestamp values to datatime values. c)Display only the Datetime and Close columns. |
| [4-array.py](4-array.py) | Complete the following script to take the last 10 rows of the columns High and Close and convert them into a numpy.ndarray. |
| [5-slice.py](5-slice.py) | Complete the following script to slice the pd.DataFrame along the columns High, Low, Close, and Volume_BTC, taking every 60th row. |
| [6-flip_switch.py](6-flip_switch.py) | Complete the following script to alter the pd.DataFrame such that the rows and columns are transposed and the data is sorted in reverse chronological order. |
| [7-high.py](7-high.py) | Complete the following script to sort the pd.DataFrame by the High price in descending order. |
| [8-prune.py](8-prune.py) | Complete the following script to remove the entries in the pd.DataFrame where Close is NaN. |
| [9-fill.py](9-fill.py) | Complete the following script to fill in the missing data points in the pd.DataFrame: a)The column Weighted_Price should be removed. b)missing values in High, Low, Open, and Close should be set to the previous row’s Close value. c)missing values in Volume_(BTC) and Volume_(Currency) should be set to 0. |
| [10-index.py](10-index.py) | Complete the following script to index the pd.DataFrame on the Timestamp column. |
| [11-concat.py](11-concat.py) | Complete the following script to index the pd.DataFrames on the Timestamp columns and concatenate them: a)Concatenate the start of the bitstamp table onto the top of the coinbase table. b)Include all timestamps from bitstamp up to and including timestamp 1417411920. c)Add keys to the data labeled bitstamp and coinbase respectively. |
| [12-hierarchy.py](12-hierarchy.py) | Based on 11-concat.py, rearrange the MultiIndex levels such that timestamp is the first level: a)Concatenate th bitstamp and coinbase tables from timestamps 1417411980 to 1417417980, inclusive. b)Add keys to the data labeled bitstamp and coinbase respectively. c)Display the rows in chronological order. |
| [13-analyze.py](13-analyze.py) | Complete the following script to calculate descriptive statistics for all columns in pd.DataFrame except Timestamp. |
| [14-visualize.py](14-visualize.py) | Complete the following script to visualize the pd.DataFrame: a)Plot the data from 2017 and beyond at daily intervals. b)The column Weighted_Price should be removed. c)Rename the column Timestamp to Date. d)Convert the timestamp values to date values. e)Index the data frame on Date. f)Missing values in High, Low, Open, and Close should be set to the previous row’s Close value. g)Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0. |


### Build with
- Python (python 3.5)
- Numpy (numpy 1.15)
- Pandas (pandas 0.24.x)
- Ubuntu 16.04 LTS 

## Author

[Daniela Chamorro](https://www.linkedin.com/in/dalexach/) [:octocat:](https://github.com/dalexach)

[Twitter](https://twitter.com/dalexach)
