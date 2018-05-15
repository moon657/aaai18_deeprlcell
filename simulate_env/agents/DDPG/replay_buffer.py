# UHANA CONFIDENTIAL
# __________________
#
# Copyright (C) Uhana, Incorporated
# [2015] - [2016] Uhana Incorporated
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Uhana Incorporated and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Uhana Incorporated
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or Reproduction of this material
# or Unauthorized copying of this file, via any medium is strictly
# prohibited and strictly forbidden.

from collections import deque
from scipy.signal import lfilter
import numpy as np
import cPickle
import random

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def sample_batch(self, batch_size):
        # Randomly sample batch_size examples
        minibatch = random.sample(self.buffer, batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def save_buffer(self, path):
        save_file = open(path, 'wb')
        cPickle.dump(self.buffer, save_file, -1)
        save_file.close()

    def load_buffer(self, path):
        save_file = open(path, 'rb')
        data = cPickle.load(save_file)
        self.buffer += data
        self.num_experiences += len(data)
        save_file.close()

def discount(x, gamma):
    return lfilter([1],[1,-gamma],x[::-1])[::-1]

class RolloutBuffer(ReplayBuffer):
    def __init__(self, norm_len):
        self.buffer_size = 0
        self.buffer = {'state':[], 'action':[], 'reward':[]}
        self.all_rewards = []
        self.norm_len = norm_len

    def add(self, state, action, reward):
        self.buffer['state'].append(state)
        self.buffer['action'].append(action)
        self.buffer['reward'].append(reward)
        self.buffer_size += 1

    def erase(self):
        self.buffer_size = 0
        self.buffer = {'state':[], 'action':[], 'reward':[]}

    def rollout(self, gamma):
        """
        Gathering a single episode of trasitions.
        """
        rewards = np.array(self.buffer['reward'])
        states = np.array(self.buffer['state'])
        actions = np.array(self.buffer['action'])
        discounted_rewards = discount(rewards, gamma)
        self.all_rewards += list(discounted_rewards)
        self.all_rewards = self.all_rewards[:self.norm_len]
        discounted_rewards = (discounted_rewards-np.mean(self.all_rewards))/np.std(self.all_rewards)
        return states, actions, discounted_rewards 

