#!/usr/bin/env python3
"""
Where I am?
"""
import requests


def sentientPlanets():
    """
    Method that returns the list of names of the home planets
    of all sentient species

    Returns:
     The list of names of the home planets
    """

    url = "https://swapi-api.hbtn.io/api/species/"
    planets = []

    while url is not None:
        req = requests.get(url)
        results = req.json()['results']

        for specie in results:
            if (specie['designation'] == 'sentient' or
                    specie['classification'] == 'sentient'):

                urlplanet = specie['homeworld']
                if urlplanet is not None:
                    planet = requests.get(urlplanet).json()
                    planets.append(planet['name'])
        url = req.json()['next']

    return planets
