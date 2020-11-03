#!/usr/bin/env python3
"""
Log stats
"""
from pymongo import MongoClient


if __name__ == "__main__":
    """
    Python script that provides some stats about Nginx logs stored in MongoDB:

    Database: logs
    Collection: nginx
    Display:
     - first line: x logs where x is the number of documents in this collection
     - second line: Methods:
     - 5 lines with the number of documents with the method
        ["GET", "POST", "PUT", "PATCH", "DELETE"] in this order
     - one line with the number of documents with:
        - method=GET
        - path=/status
    """

    client = MongoClient('mongodb://127.0.0.1:27017')
    logs_collection = client.logs.nginx
    count_docs = logs_collection.count_documents({})
    print('{} logs'.format(count_docs))
    print('Methods:')
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for met in methods:
        num_met = logs_collection.count_documents({'method': met})
        print('\tmethod {}: {}'.format(met, num_met))

    filter_path = {'method': 'GET', 'paht': '/status'}
    num_path = logs_collection.count_documents(filter_path)
    print("{} status check".format(num_path))
