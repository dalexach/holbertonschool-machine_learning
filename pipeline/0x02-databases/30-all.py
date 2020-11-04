#!/usr/bin/env python3
"""
List all socuments in Python
"""


def list_all(mongo_collection):
    """
    Function that lists all documents in a collection

    Arguments:
     - mongo_collection will be the pymongo collection object

    Return:
     An empty list if no document in the collection
    """

    docs = []
    collection = mongo_collection.find()
    for doc in collection:
        docs.append(doc)

    return docs
