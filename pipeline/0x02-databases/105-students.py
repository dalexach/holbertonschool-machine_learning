#!/usr/bin/env python3
"""
Top students
"""


def top_students(mongo_collection):
    """
    Function that returns all students sorted by average score

    Arguments:
     - mongo_collection will be the pymongo collection object

    Note:
     - The top must be ordered
     - The average score must be part of each item returns with
        key = averageScore

    Returns:
     All students sorted by average score
    """
    students = mongo_collection.find()
    top_students = []
    for student in students:
        topics = student["topics"]
        score = 0
        for topic in topics:
            score += topic["score"]
        score /= len(topics)
        student["averageScore"] = score
        top_students.append(student)

    return sorted(top_students, key=lambda i: i["averageScore"], reverse=True)
