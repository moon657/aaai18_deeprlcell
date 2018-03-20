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

# run experiments in parallel
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

# probability from t-1 to t
"""
    p_t(s' | s, a)
    based on time variant congestion shifts
"""

def PPC_env_get_trans_prob_vector(continuous_state = None, action_index = None, horizon = None, trans_prob_params = None, print_mode = False):

    master_cell_records = trans_prob_params['master_cell_records']
    congestion_var = trans_prob_params['congestion_var']
    time_index_var = trans_prob_params['time_index_var']
    stateIndex_to_state = trans_prob_params['stateIndex_to_state']
    actionIndex_to_action = trans_prob_params['actionIndex_to_action'] 
    discretized_state_values = trans_prob_params['discretized_state_values']
    states = trans_prob_params['states']
    tol = trans_prob_params['numerical_tolerance']

    if print_mode:
        print(trans_prob_params)

    congestion_t = list(master_cell_records[master_cell_records[time_index_var] == horizon][congestion_var])[0]
    congestion_tminus1 = list(master_cell_records[master_cell_records[time_index_var] == horizon-1][congestion_var])[0]

    continuous_action = actionIndex_to_action[action_index]
    
    gaussian_noise = 0.0

    historical_commute_delta = congestion_t - congestion_tminus1

    # do piecewise dynamics
    if continuous_action == 0.0:
        next_continuous_state = congestion_t + gaussian_noise
    
    else:
        next_continuous_state = continuous_state + trans_prob_params['M'] * continuous_action + historical_commute_delta + gaussian_noise

    # get a prob distro over this next state if non-gaussian noise
    # based on this see where next states could be binned
    next_discrete_state, discretized_next_continuous_state = get_discrete_state_from_cont_query(query_y = next_continuous_state, discretized_y = discretized_state_values)

    next_state_list = []
    next_state_probs = []

    for next_possible_state_index, next_possible_state in enumerate(states):
        next_state_list.append(next_possible_state)
      
        if next_possible_state_index == next_discrete_state:
            next_state_probs.append(1.0 - trans_prob_params['epsilon'])
        else:
            next_state_probs.append(trans_prob_params['epsilon']/(len(states) - 1))

    assert( (np.sum(next_state_probs) >= 1.0-tol) )

    if print_mode:
        print('continuous state ', continuous_state)
        print('continuous action ', continuous_action)
        print('next continuous state ', next_continuous_state)
        print('next state probs ', next_state_probs)
        print(' ')
        print(' ')
    return next_state_probs, next_state_list, next_continuous_state

"""
compute cell thpt from the congestion value
"""
def throughput_model(congestion = None, reward_params_dict = None):

    if reward_params_dict['RF_mode'] == False:
        thpt = float(1.0)/float(congestion)

    else:
        rf = reward_params_dict['RF_model']

        if reward_params_dict['CE_mode'] == 'MEDIAN': 
            single_state_list = [] 
            for feature in reward_params_dict['RF_features']:
                if feature == reward_params_dict['congestion_var']:
                    single_state_list.append(congestion)
                else:
                    single_state_list.append(reward_params_dict['MEDIAN_VALUE_' + feature])
            thpt_vec = rf.predict(np.array(single_state_list).reshape([1,-1]))

        else:
            single_state_list = [] 
            for feature in reward_params_dict['RF_features']:
                if feature == reward_params_dict['congestion_var']:
                    single_state_list.append(congestion)
                else:
                    single_state_list.append(reward_params_dict['INSTANTANEOUS_VALUE_' + feature])
            thpt_vec = rf.predict(np.array(single_state_list).reshape([1,-1]))

        thpt = thpt_vec[0]
    return thpt

"""
    r(s, a, s')
    weighted sum of IOT traffic and loss to users
    returns a scalar from a specific (s, a, s') tuple
"""

def PPC_env_get_single_transition_reward(congestion = None, action = None, new_congestion = None, reward_params_dict = None, print_mode = False):
    # modulate various terms in reward computation
    alpha = reward_params_dict['alpha']
    beta = reward_params_dict['beta']
    kappa = reward_params_dict['kappa']
    burst_prob_user_selector = reward_params_dict['burst_prob_user_selector']
    control_interval_seconds = reward_params_dict['control_interval_seconds']
    avg_user_burst_prob = reward_params_dict['avg_user_burst_prob']
    KB_MB_converter = reward_params_dict['KB_MB_converter']
    hard_thpt_limit_flag = reward_params_dict['hard_thpt_limit_flag']
    hard_thpt_limit = reward_params_dict['hard_thpt_limit']

    # compute IOT data scheduled
    burst_prob = action

    orig_cell_thpt = throughput_model(congestion = congestion, reward_params_dict = reward_params_dict)
    new_cell_thpt = throughput_model(congestion = new_congestion, reward_params_dict = reward_params_dict)

    IOT_scheduled_data_MB = float(
        new_cell_thpt * burst_prob *
        control_interval_seconds) / float(KB_MB_converter)

    # do IOT user and regular user have same burst probs?
    if (burst_prob_user_selector == 'same_as_IOT'):
        burst_prob_user = burst_prob
    else:
        burst_prob_user = avg_user_burst_prob

    user_original_data_MB = (orig_cell_thpt * burst_prob_user *
                             control_interval_seconds) / float(KB_MB_converter)
    user_new_data_MB = (new_cell_thpt * burst_prob_user *
                        control_interval_seconds) / float(KB_MB_converter)
    user_lost_data_MB = user_original_data_MB - user_new_data_MB

    # only if flag is set do we add a penalty for thpts that are below a hard limit, eg (K-B')
    if (hard_thpt_limit_flag):
        # get a penalty if below the thpt limit
        if (new_cell_thpt < reward_params_dict['hard_thpt_limit']):
            hard_thpt_limit_MB = (
                burst_prob *
                (reward_params_dict['hard_thpt_limit'] - new_cell_thpt) *
                control_interval_seconds) / float(KB_MB_converter)
            #print('penalty ', new_cell_thpt, reward_params_dict['hard_thpt_limit'], hard_thpt_limit_MB)
        else:
            hard_thpt_limit_MB = 0.0
    else:
        hard_thpt_limit_MB = 0

    # REWARD COMPUTATION
    reward = alpha * IOT_scheduled_data_MB - beta * user_lost_data_MB - kappa * hard_thpt_limit_MB

    #reward = reward / reward_scale
    if (print_mode):
        print('cell_thpt', orig_cell_thpt)
        print('new_cell_thpt', new_cell_thpt)
        print('burst_prob_IOT', burst_prob)
        print('burst_prob_user', burst_prob_user)
        print('IOT_scheduled_data_MB', IOT_scheduled_data_MB)
        print('data_MB lost user', user_lost_data_MB)
        print('thpt penalty', hard_thpt_limit_MB)
        print('reward', reward)
        print(' ')
        print(' ')

    return reward, IOT_scheduled_data_MB, user_lost_data_MB, hard_thpt_limit_MB, orig_cell_thpt, new_cell_thpt

def PPC_env_get_reward_vector(state_index = None, action_index = None, reward_params_dict = None, print_mode = False, trans_prob_params = None, horizon = None):

    master_cell_records = trans_prob_params['master_cell_records']
    congestion_var = trans_prob_params['congestion_var']
    time_index_var = trans_prob_params['time_index_var']
    stateIndex_to_state = trans_prob_params['stateIndex_to_state']
    actionIndex_to_action = trans_prob_params['actionIndex_to_action'] 
    discretized_state_values = trans_prob_params['discretized_state_values']
    states = trans_prob_params['states']
    tol = trans_prob_params['numerical_tolerance']

    next_state_list = []
    reward_vector = []

    # get the N, E for the current timestep from real timeseries
    if reward_params_dict['RF_mode']:
        if reward_params_dict['CE_mode'] == 'INSTANTANEOUS':
            curr_ts_df = master_cell_records[master_cell_records[time_index_var] == horizon]
            reward_params_dict['INSTANTANEOUS_VALUE_' + reward_params_dict['specf_var']] = list(curr_ts_df[reward_params_dict['specf_var']])[0]
            reward_params_dict['INSTANTANEOUS_VALUE_' + reward_params_dict['num_sess_var']] = list(curr_ts_df[reward_params_dict['num_sess_var']])[0]

            if print_mode: 
                print('horizon: ', horizon)
                print('specf: ', reward_params_dict['INSTANTANEOUS_VALUE_' + reward_params_dict['specf_var']])
                print('nsess: ', reward_params_dict['INSTANTANEOUS_VALUE_' + reward_params_dict['num_sess_var']])
                print(' ')

    for next_possible_state_index, next_possible_state in enumerate(states):
        next_state_list.append(next_possible_state)
      
        continuous_state = stateIndex_to_state[state_index]
        continuous_action = actionIndex_to_action[action_index]
        new_congestion = stateIndex_to_state[next_possible_state_index]

        reward, IOT_scheduled_data_MB, user_lost_data_MB, hard_thpt_limit_MB, orig_cell_thpt, new_cell_thpt = PPC_env_get_single_transition_reward(congestion = continuous_state, action = continuous_action, new_congestion = new_congestion, reward_params_dict = reward_params_dict, print_mode = print_mode)
        reward_vector.append(reward)

    if print_mode:
        print('reward vector ', reward_vector)
        print(' ')
        print(' ')

    return reward_vector, next_state_list 
