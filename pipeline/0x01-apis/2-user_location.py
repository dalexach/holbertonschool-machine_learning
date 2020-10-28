#!/usr/bin/env python3
"""
Rate me is you can!
Printing the location of a specific user
"""
import sys
import requests
import time


if __name__ == '__main__':

    url = sys.argv[1]
    payload = {'Accept': 'application/vnd.github.v3+json'}
    r = requests.get(url, params=payload)

    # Status: 403 Forbidden
    if r.status_code == 403:
        limit = r.headers["X-Ratelimit-Reset"]
        x = (int(limit) - int(time.time())) / 60
        print("Reset in {} min".format(int(x)))

    # Status: 200 OK
    if r.status_code == 200:
        location = r.json()["location"]
        print(location)

    # Status: 404 Not Found
    if r.status_code == 404:
        print("Not found")
