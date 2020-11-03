#!/usr/bin/env python3
"""
Where can I learn Python?
"""


def schools_by_topic(mongo_collection, topic):
    """
    Function that returns the list of school having a specific topic

    Arguments:
     - mongo_collection will be the pymongo collection object
     - topic (string) will be topic searched
    """

    research = []
    result = mongo_collection.find({'topics': {'$all': [topic]}})
    for res in result:
        research.append(res)

    return research
