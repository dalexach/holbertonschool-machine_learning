#!/usr/bin/env python3
"""
Play
"""
import gym


def play(env, Q, max_steps=100):
    """
    Function that has the trained agent play an episode

    Arguments:
     - env is the FrozenLakeEnv instance
     - Q is a numpy.ndarray containing the Q-table
     - max_steps is the maximum number of steps in the episode

    Returns:
     The total rewards for the episode
    """
