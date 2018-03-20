from __future__ import print_function
from __future__ import division
import sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import itertools
import argparse
import ConfigParser
import pandas
import collections
from collections import OrderedDict
import operator
import h5py
import joblib
import time

# plotting utils
AAAI_ROOT_DIR = os.environ['AAAI_ROOT_DIR']
util_dir = AAAI_ROOT_DIR + '/utils/'
sys.path.append(util_dir)
from textfile_utils import list_from_textfile, remove_and_create_dir
from discretization_utils import *
from IOT_DP_utils import *

# run experiments in parallel
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

"""
helper utils for doing DP with discretized states, actions, finite horizon T
    - operate on a Q table of Q: S x A x T
    - compute optimal Q table

requires:
    - problem_params_dict: info on discretization
    - reward_params_dict: info on reward structure
    - trans_prob_params: p(s' | s, a)
"""

"""
    compute optimal value and action for a specific state
    return Q^{*}(s,a), a^{*} = argmax Q^{s,a} for a specific state s
"""
def get_single_state_optimal_value(single_state_index = None, horizon = None,  Q_table = None, problem_params_dict = None):

    # looking for terminal value at T
    if horizon == Q_table.shape[-1]: 
        optimal_value = 0.0
        optimal_action_index = None
    else:
        Q_t = Q_table[:,:,horizon]
        single_action_value_function = Q_t[single_state_index,:]

        assert(len(single_action_value_function) == problem_params_dict['action_space_dim'])

        optimal_action_index = np.argmax(single_action_value_function)
        optimal_value = single_action_value_function[optimal_action_index]

    return optimal_value, optimal_action_index

"""
    pre-caches a table of R(S, A, S') for all (s, a, s') tuples
    need to pass in env specific info to get problem specific reward values
"""

def get_reward_table(problem_params_dict = None, reward_params_dict = None, print_mode = False, trans_prob_params = None, env_type = 'PPC'):

    state_space_dim = problem_params_dict['state_space_dim']
    action_space_dim = problem_params_dict['action_space_dim']
    states = problem_params_dict['states']
    actions = problem_params_dict['actions']

    # init Q table
    reward_table = np.zeros((state_space_dim, action_space_dim, state_space_dim))

    stateIndex_to_state = trans_prob_params['stateIndex_to_state']
    actionIndex_to_action = trans_prob_params['actionIndex_to_action'] 

    for state_index, state in enumerate(states):
        for action_index, action in enumerate(actions):

            for next_possible_state_index, next_possible_state in enumerate(states):
              
                continuous_state = stateIndex_to_state[state_index]
                continuous_action = actionIndex_to_action[action_index]
                new_continuous_state = stateIndex_to_state[next_possible_state_index]

                reward = get_single_transition_reward(continuous_state = continuous_state, continuous_action = continuous_action, new_continuous_state = new_continuous_state, reward_params_dict = reward_params_dict, print_mode = print_mode, env_type = 'PPC')
                
                reward_table[state_index, action_index, next_possible_state_index] = reward 

            if print_mode:
                print(reward_table)
                print(' ')
                print(' ')
    return reward_table 


"""
select between different reward fncs based on env
"""
def get_single_transition_reward(continuous_state = None, continuous_action = None, new_continuous_state = None, reward_params_dict = None, print_mode = None, env_type = 'PPC'):

    if env_type == 'PPC':
        reward, IOT_scheduled_data_MB, user_lost_data_MB, hard_thpt_limit_MB, orig_cell_thpt, new_cell_thpt = PPC_env_get_single_transition_reward(congestion = continuous_state, action = continuous_action, new_congestion = new_continuous_state, reward_params_dict = reward_params_dict, print_mode = print_mode)
    else:
        raise NotImplementedError('env not supported')

    return reward

"""
select between different trans prob vectors based on env
"""
def get_trans_prob_vector(continuous_state = None, action_index = None, horizon = None, trans_prob_params = None, print_mode = None, env_type = 'PPC'):

    if env_type == 'PPC':
        next_state_probs, next_state_list, next_continuous_state = PPC_env_get_trans_prob_vector(continuous_state = continuous_state, action_index = action_index, horizon = horizon, trans_prob_params = trans_prob_params, print_mode = print_mode)
    else:
        raise NotImplementedError('env not supported')

    return next_state_probs, next_state_list, next_continuous_state


def get_reward_vector(env_type = 'PPC', reward_table = None, state_index = None, action_index = None, reward_params_dict = None, print_mode = None, trans_prob_params = None, horizon = None):

    if env_type == 'PPC':
        if reward_params_dict['CE_mode'] == 'INSTANTANEOUS':
            reward_vector, next_state_list = PPC_env_get_reward_vector(state_index = state_index, action_index = action_index, reward_params_dict = reward_params_dict, print_mode = print_mode, trans_prob_params = trans_prob_params, horizon = t-1)
        else:
            reward_vector = reward_table[state_index, action_index,:]
    else:
        raise NotImplementedError('env not supported')

    return reward_vector

"""
    state action value loop
    compute Q^{*}_{t-1}(s, a) for a specific state s, action a from Q^{*}_t
    return a 1 X S array, called Q_state_action_element
"""

def compute_specific_state_action_value_loop(state_index = None, action_index = None, t = None, trans_prob_params = None, print_mode = None, reward_table = None, Q_table = None, problem_params_dict = None, reward_params_dict = None, env_type = 'PPC'):
    
    stateIndex_to_state = trans_prob_params['stateIndex_to_state']
    state = stateIndex_to_state[state_index]
    states = trans_prob_params['states']

    next_state_probs, next_state_list, next_continuous_state = get_trans_prob_vector(continuous_state = state, action_index = action_index, horizon = t, trans_prob_params = trans_prob_params, print_mode = print_mode, env_type = env_type)

    # REWARD VECTOR
    # 1 X S numpy array
    reward_vector = get_reward_vector(reward_table = reward_table, state_index = state_index, action_index = action_index, reward_params_dict = reward_params_dict, print_mode = print_mode, trans_prob_params = trans_prob_params, horizon = t-1, env_type = env_type)

    if print_mode:
        print('next_state_probs ', next_state_probs)
        print(' ')
        print('reward_vector ', reward_vector)
        print(' ')

    # NEXT OPTIMAL VALUE
    # 1 X S array
    next_state_value_vector = [get_single_state_optimal_value(single_state_index = x, horizon = t, Q_table = Q_table, problem_params_dict = problem_params_dict)[0] for x in range(len(states))]

    if print_mode:
        print('next_state_value_vector ', next_state_value_vector)
        print(' ')

    # 1 x S
    future_discounted_cumulative_reward_vec = np.array(reward_vector) + problem_params_dict['GAMMA'] * np.array(next_state_value_vector)

    Q_state_action_element = np.dot(np.array(next_state_probs), future_discounted_cumulative_reward_vec)

    return Q_state_action_element

def compute_specific_state_VECTOR_value_loop(state_index = None, t = None, trans_prob_params = None, print_mode = None, reward_table = None, Q_table = None, problem_params_dict = None, reward_params_dict = None, env_type = 'PPC'):
    
    stateIndex_to_state = trans_prob_params['stateIndex_to_state']
    state = stateIndex_to_state[state_index]
    states = trans_prob_params['states']
    actions = problem_params_dict['actions']
    action_space_dim = problem_params_dict['action_space_dim']

    Q_specific_state_vec = []

    for action_index, action in enumerate(actions):

        next_state_probs, next_state_list, next_continuous_state = get_trans_prob_vector(continuous_state = state, action_index = action_index, horizon = t, trans_prob_params = trans_prob_params, print_mode = print_mode, env_type = env_type)

        reward_vector = get_reward_vector(reward_table = reward_table, state_index = state_index, action_index = action_index, reward_params_dict = reward_params_dict, print_mode = print_mode, trans_prob_params = trans_prob_params, horizon = t-1, env_type = env_type)

        if print_mode:
            print('next_state_probs ', next_state_probs)
            print(' ')
            print('reward_vector ', reward_vector)
            print(' ')

        # NEXT OPTIMAL VALUE
        # 1 X S array
        next_state_value_vector = [get_single_state_optimal_value(single_state_index = x, horizon = t, Q_table = Q_table, problem_params_dict = problem_params_dict)[0] for x in range(len(states))]

        if print_mode:
            print('next_state_value_vector ', next_state_value_vector)
            print(' ')

        # 1 x S
        future_discounted_cumulative_reward_vec = np.array(reward_vector) + problem_params_dict['GAMMA'] * np.array(next_state_value_vector)

        Q_state_action_element = np.dot(np.array(next_state_probs), future_discounted_cumulative_reward_vec)
        Q_specific_state_vec.append(Q_state_action_element)

    return np.array(Q_specific_state_vec)

"""
load a Q table which is a h5 file that has been pre-computed
"""

def load_Q_table(problem_params_dict = None):

    print('load Q table from hd5 file')
    save_Q_table = problem_params_dict['base_results_dir'] + '/Qtable.' + str(problem_params_dict['problem_number']) + '.' + problem_params_dict['cell_day'] + '.h5'
    
    with h5py.File(save_Q_table, 'r') as hf:
        Q_table = hf['Q_table'][:]

    return Q_table

"""
    write a Q table to hd5 file
"""
def save_Q_table(problem_params_dict = None, Q_table = None):
    print('write Q table to hd5 file')
    save_Q_table = problem_params_dict['base_results_dir'] + '/Qtable.' + str(problem_params_dict['problem_number']) + '.' + problem_params_dict['cell_day'] + '.h5'
    
    with h5py.File(save_Q_table, 'w') as hf:
        hf.create_dataset("Q_table",  data=Q_table)

