#!/usr/bin/env python3
"""
Train an agent that can play Atari’s Breakout
"""
import gym
import h5py
import keras as K
from keras import layers
from keras.optimizers import Adam
import numpy as np
from PIL import Image
from rl.core import Processor
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


class AtariProcessor(Processor):
    """
    Class that prepocesses data based on Deep Learning
    Quick Reference by Mike Bernico
    """
    def process_observation(self, observation):
        """
        Function to preprocess the state
        """
        obs = observation
        assert obs.ndim == 3
        img = Image.fromarray(obs)
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')
        img = np.array(img)
        assert img.shape == (84, 84)

        return img.astype('uint8')

    def process_state_batch(self, batch):
        """
        Function to preprocesses the state of a batch
        """

        processed = batch.astype('float32') / 255

        return processed

    def process_reward(self, reward):
        """
        Function to processes the reward
        """

        return np.clip(reward, -1., 1.)


def create_q_model(actions):
    """
    Function that creates a CNN
    """
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(4, 84, 84))
    layer0 = layers.Permute((2, 3, 1))(inputs)
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu",
                           data_format="channels_last")(layer0)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu",
                           data_format="channels_last")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu",
                           data_format="channels_last")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(actions, activation="linear")(layer5)

    return K.Model(inputs=inputs, outputs=action)


# Enviroment
env = gym.make('BreakoutNoFrameskip-v4')
state = env.reset()
actions = env.action_space.n
model = create_q_model(actions)
memory = SequentialMemory(limit=1000000, window_length=4)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                              value_max=1., value_min=.1,
                              value_test=.05, nb_steps=1000000)
stateprocess = AtariProcessor()
dqn = DQNAgent(model=model, nb_actions=actions, memory=memory,
               nb_steps_warmup=50000, target_model_update=10000,
               policy=policy, processor=stateprocess, train_interval=4,
               gamma=.99, delta_clip=1.)
dqn.compile(optimizer=Adam(lr=0.00025), metrics=['mae'])
dqn.fit(env, nb_steps=17500, log_interval=10000, visualize=False, verbose=2)
dqn.save_weights('policy.h5', overwrite=True)
