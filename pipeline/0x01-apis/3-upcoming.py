#!/usr/bin/env python3
"""  What will be next? """
import requests


if __name__ == '__main__':

    url = "https://api.spacexdata.com/v4/launches/upcoming"
    res = requests.get(url)
    launches = res.json()
    date = float('inf')
    for i, launch in enumerate(launches):
        if date > launch["date_unix"]:
            date = launch["date_unix"]
            index = i
    lname = launches[index]["name"]
    ldate = launches[index]["date_local"]
    lrocket = launches[index]["rocket"]
    url = "https://api.spacexdata.com/v4/rockets/{}".format(lrocket)
    rocketname = requests.get(url).json()["name"]
    lidx = launches[index]["launchpad"]
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(lidx)
    lp = requests.get(url).json()
    launchname = lp["name"]
    launclocale = lp["locality"]
    # Format
    # <launch name> (<date>) <rocket name> - <launchpad name>
    # (<launchpad locality>)
    data = '{} ({}) {} - {} ({})'.format(lname, ldate, rocketname,
                                         launchname, launclocale)

    print(data)
