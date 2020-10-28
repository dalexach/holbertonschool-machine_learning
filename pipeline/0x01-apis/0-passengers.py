#!/usr/bin/env python3
"""
Can I join?
"""
import requests


def availableShips(passengerCount):
    """
    Method that returns the list of ships that can hold a
    given number of passengers

    Returns:
     List of the ships, if no ship available, return an empty list.
    """

    url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []

    while url is not None:
        res = requests.get(url)
        results = res.json()['results']
        for ship in results:
            pas = ship['passengers']
            pas = pas.replace(',', '')
            if pas.isnumeric() and int(pas) >= passengerCount:
                ships.append(ship['name'])
        url = res.json()['next']

    return ships
