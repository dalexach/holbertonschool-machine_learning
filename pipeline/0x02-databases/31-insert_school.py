#!/usr/bin/env python3
"""
Inset a document in Python
"""


def insert_school(mongo_collection, **kwargs):
    """
    Function that inserts a new document in a colletion based on kwargs

    Arguments:
    mongo_collection will be the pymongo collection object

    Returns:
     The new _id
    """

    id_ = mongo_collection.insert_one(kwargs).inserted_id

    return id_
