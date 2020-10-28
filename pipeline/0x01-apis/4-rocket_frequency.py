#!/usr/bin/env python3
"""
How many by rocket?
Displays the number of launches per rocket
"""
import requests


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/"
    rockets = {}
    res = requests.get(url)
    launches = res.json()

    for idx, launch in enumerate(launches):
        rId = launch['rocket']
        rocketUrl = "https://api.spacexdata.com/v4/rockets/{}".format(rId)
        rData = requests.get(rocketUrl).json()
        rName = rData['name']

        if rName in rockets.keys():
            rockets[rName] += 1
        else:
            rockets[rName] = 1

    rocket = sorted(rockets.items(), key=lambda kv: kv[0])
    rocket = sorted(rocket, key=lambda kv: kv[1], reverse=True)

    for r in rocket:
        print('{}: {}'.format(r[0], r[1]))
