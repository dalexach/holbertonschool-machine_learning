#!/usr/bin/env python3
"""
Class Poisson that represents a poisson distribution
"""


class Poisson:
    """
    Representing a Poisson distribution
    """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor
        Arguments:
        - data (list): is a list of the data to be used to estimate the
        distribution
        - lambtha (int/float): is the expected number of occurences
        in a given time frame
        """
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Instance method that calculates the value of the PMF for a
        given number of successes
        Arguments:
         - k (int/float): number of "successes"
        """
        if k <= 0:
            return 0
        k = int(k)
        fact_k = 1
        for x in range(1, k+1):
            fact_k = x * fact_k

        pmf = (pow(self.e, self.lambtha * -1)
               * (pow(self.lambtha, k))) / fact_k

        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        Arguments:
         - k (int/float): number of "successes"
        """
        if k <= 0:
            return 0

        k = int(k)

        cdf = 0

        for n in range(1, k+1):
            cdf += (self.pmf(n))

        return cdf
