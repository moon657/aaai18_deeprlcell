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
from Q_DP_utils import *
from IOT_DP_utils import *

# run experiments in parallel
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

"""
these are problem agnostic utils for discretized DP
"""

def compute_Q_table_finite_horizon(problem_params_dict = None,  print_mode = False, trans_prob_params = None, reward_params_dict = None, save_Q_table_mode = True, progress_print_mode = True):

    # problem params
    state_space_dim = problem_params_dict['state_space_dim']
    action_space_dim = problem_params_dict['action_space_dim']
    T = problem_params_dict['T']
    GAMMA = problem_params_dict['GAMMA']
    states = problem_params_dict['states']
    actions = problem_params_dict['actions']
    print_interval = problem_params_dict['print_interval']
    reward_table = reward_params_dict['reward_table']

    # init Q table
    Q_table = np.zeros((state_space_dim, action_space_dim, T))
    horizons = range(1,T+1)
    horizons.reverse()

    if print_mode:
        print('horizon : ', T)
        print('Q_table : ', Q_table.shape)
        print(' ')
    elapsed = 0.0

    if progress_print_mode:
        print('RUNNING ', problem_params_dict['compute_mode'])
        print('REWARD MODE ', reward_params_dict['CE_mode'])

    # init DP loop
    for t in horizons:

        start = time.time()
        if t % print_interval == 0:
            fraction_done = float(T-t)/float(T)
            
            next_state_value_vector = [get_single_state_optimal_value(single_state_index = x, horizon = t, Q_table = Q_table, problem_params_dict = problem_params_dict)[0] for x in range(len(states))]
            optimal_value_across_states = np.max(next_state_value_vector)
            
            if progress_print_mode:
                print('fraction done ', fraction_done, ' t ', t, ' time elapsed ', elapsed)
                print('optimal value: ', optimal_value_across_states)

        if print_mode:
            print('horizon : ', t)
            print(' ')

        # construct Q_t-1
        local_Q_t_minus1 = np.zeros((state_space_dim, action_space_dim))
        
        if problem_params_dict['compute_mode'] == 'SEQUENTIAL':
            for state_index, state in enumerate(states):
                for action_index, action in enumerate(actions):
                    Q_state_action_element = compute_specific_state_action_value_loop(state_index = state_index, action_index = action_index, t = t, trans_prob_params = trans_prob_params, print_mode = print_mode, reward_table = reward_table, Q_table = Q_table, problem_params_dict = problem_params_dict, reward_params_dict = reward_params_dict)
                    # get the element
                    local_Q_t_minus1[state_index, action_index] = Q_state_action_element 

            # update next Q table based on Bellman backup
            Q_table[:,:,(t-1)] = local_Q_t_minus1

        else:
            Q_specific_state_vec_list = Parallel(n_jobs=num_cores)(delayed(compute_specific_state_VECTOR_value_loop)(state_index = state_index, t = t, trans_prob_params = trans_prob_params, print_mode = print_mode, reward_table = reward_table, Q_table = Q_table, problem_params_dict = problem_params_dict, reward_params_dict = reward_params_dict) for state_index in range(len(states))) 

            for state_index, Q_state_row in enumerate(Q_specific_state_vec_list):
                Q_table[state_index, :, (t-1)] = Q_state_row

        end = time.time()
        elapsed = end-start

        # also get the best value and action for the current stage, print this for debugging

    if save_Q_table_mode:
        save_Q_table(problem_params_dict = problem_params_dict, Q_table = Q_table)

    return Q_table

def execute_control_strategy(Q_table = None, problem_params_dict = None,  print_mode = False, trans_prob_params = None, reward_params_dict = None, action_sequence_vec = None, control_strategy = 'Q_policy', override_init_state_mode = False, override_cts_init_state = None, env_type = 'PPC'):

    state_space_dim = problem_params_dict['state_space_dim']
    action_space_dim = problem_params_dict['action_space_dim']
    T = problem_params_dict['T']
    GAMMA = problem_params_dict['GAMMA']

    congestion_var = trans_prob_params['congestion_var']
    stateIndex_to_state = trans_prob_params['stateIndex_to_state']
    actionIndex_to_action = trans_prob_params['actionIndex_to_action']
    discretized_y = trans_prob_params['discretized_state_values']
    states = trans_prob_params['states']
    
    discretized_controlled_congestion_var = 'DISCRETIZED_CONTROLLED_' + congestion_var
    controlled_congestion_var = 'CONTINUOUS_CONTROLLED_' + congestion_var
    action_var = 'ACTION'
    action_index_var = 'ACTION_INDEX'
    time_index_var = 'TIME_INDEX'
    reward_var = 'REWARD'

    master_cell_records = trans_prob_params['master_cell_records'].copy()

    for t in range(T):

        if t == 0:
            
            if override_init_state_mode:
                curr_state = override_cts_init_state 
            else:
                curr_state = list(master_cell_records[master_cell_records[time_index_var] == t][congestion_var])[0]      

        # propogate from the controlled state onwards
        else:
            curr_state = list(master_cell_records[master_cell_records[time_index_var] == t][controlled_congestion_var])[0]        

        # get optimal action
        curr_state_index, closest_curr_cont_state = get_discrete_state_from_cont_query(query_y = curr_state, discretized_y = discretized_y)
        curr_congestion_discretized = stateIndex_to_state[curr_state_index]

        if print_mode:
            print('eval Q debug, override_mode', override_init_state_mode)
            print('starting Q table at cont state : ', curr_state)
            print('starting Q table at discrete state : ', curr_congestion_discretized)


        if control_strategy == 'Q_policy':
            optimal_value, optimal_action_index = get_single_state_optimal_value(single_state_index = curr_state_index, horizon = t, Q_table = Q_table, problem_params_dict = problem_params_dict)
        else:
            optimal_action_index = action_sequence_vec[t]

        if print_mode:
            print(' t ', t)
            print('optimal value ', optimal_value)
            print('optimal action_index ', optimal_action_index)

        # get the next state based on dynamics, sample stochastically, from trans probs
        next_t = t + 1
        optimal_action = actionIndex_to_action[optimal_action_index]
        next_state_probs, next_state_list, next_continuous_state = get_trans_prob_vector(continuous_state = curr_congestion_discretized, action_index = optimal_action_index, horizon = next_t, trans_prob_params = trans_prob_params, print_mode = False)

        next_controlled_state = np.random.choice(next_state_list, p = next_state_probs)
   
        reward = get_single_transition_reward(continuous_state = curr_congestion_discretized, continuous_action = optimal_action, new_continuous_state = next_controlled_state, reward_params_dict = reward_params_dict, env_type = 'PPC')

        if print_mode:
            print('curr_state ', curr_state)
            print('optimal_action ', optimal_action)
            print('next continuous state', next_continuous_state)
            print('next_controlled_state ', next_controlled_state)
            print('next_state_probs ', next_state_probs)
            print('next_state_list ', next_state_list)
            print('state_list ', states)
            print('reward ', reward)
            print(' ')

        # append optimal actions and states to master_cell_records
        #####################   
        master_cell_records.set_value(t, action_var, optimal_action)
        master_cell_records.set_value(t, reward_var, reward)
        master_cell_records.set_value(t+1, controlled_congestion_var, next_continuous_state)
        master_cell_records.set_value(t+1, discretized_controlled_congestion_var, next_controlled_state)
        master_cell_records.set_value(t, action_index_var, optimal_action_index)

    optimization_results_df = master_cell_records.iloc[0:T]

    reward_vec = np.array(optimization_results_df[reward_var])

    discounted_gamma_vec = [GAMMA**t for t in range(T)]

    total_discounted_reward = np.dot(discounted_gamma_vec, reward_vec)

    if print_mode:
        print(master_cell_records.iloc[0:T+1])
        print('total discounted reward', total_discounted_reward)
   
    total_reward = np.sum(reward_vec)
    return optimization_results_df, total_discounted_reward, total_reward 

