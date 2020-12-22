#!/usr/bin/env python3
"""
Training reinforcement model
"""
import numpy as np
import matplotlib.pyplot as plt


def policy(matrix, weight):
    """
    Function that computes to policy with a weight of a matrix.

    Arguments:
     - matrix: the current observation of the environment.
     - weight: random weight.

    Returns:
     The calculated policy
    """
    z = matrix.dot(weight)
    exp = np.exp(z)

    return exp / np.sum(exp)


def policy_gradient(state, prb, action):
    """
    Function that computes the Monte-Carlo policy gradient
    based on a state and a weight matrix.
    This function change because of the implementation
    the nA env.action_space.n

    Arguemnts:
     - state matrix representing the current observation of the environment
     - weight matrix of random weight

    Return:
     The gradient.
    """

    def softmax_grad(softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    dsoftmax = softmax_grad(prb)[action, :]
    dlog = dsoftmax / prb[0, action]
    gradient = state.T.dot(dlog[None, :])

    return gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Function that implements a full training

    Argumetns:
     - env initial environment
     - nb_episodes number of episodes used for training
     - alpha the learning rate
     - gamma the discount factor

    Return:
     All values of the score
     (sum of all rewards during one episode loop)
    """
    weight = np.random.rand(4, 2)
    nA = env.action_space.n
    episode_rewards = []
    for e in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0

        while True:
            if show_result and not e % 1000:
                env.render()
            probs = policy(state, weight)
            action = np.random.choice(nA, p=probs[0])
            next_state, reward, done, _ = env.step(action)
            next_state = next_state[None, :]

            grad = policy_gradient(state, probs, action)
            grads.append(grad)

            grads.append(grad)
            rewards.append(reward)

            score += reward
            state = next_state

            if done:
                break

        for i in range(len(grads)):
            aux = sum([r * (gamma ** r) for t, r in enumerate(rewards[i:])])
            weight += alpha * grads[i] * aux

        episode_rewards.append(score)
        print("EP: " + str(e) + " Score: " + str(score) + "        ",
              end="\r", flush=False)

    return episode_rewards
