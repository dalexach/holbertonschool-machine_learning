#!/usr/bin/env python3
"""
Q-learning
"""
import gym


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Function that performs Q-learning

    Arguments:
     - env is the FrozenLakeEnv instance
     - Q is a numpy.ndarray containing the Q-table
     - episodes is the total number of episodes to train over
     - max_steps is the maximum number of steps per episode
     - alpha is the learning rate
     - gamma is the discount rate
     - epsilon is the initial threshold for epsilon greedy
     - min_epsilon is the minimum value that epsilon should decay to
     - epsilon_decay is the decay rate for updating epsilon between episodes

    Returns:
     Q, total_rewards
        - Q is the updated Q-table
        - total_rewards is a list containing the rewards per episode
    """
