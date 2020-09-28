#!/usr/bin/env python3
"""
Initialize Q-table
"""
import gym


def q_init(env):
    """
    Function that initializes the Q-table

    Arguments:
     - env is the FrozenLakeEnv instance

    Returns:
     The Q-table as a numpy.ndarray of zeros
    """
