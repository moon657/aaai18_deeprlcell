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
from main_DP_utils import *

# run experiments in parallel
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_results_dir', type=str, required=False, help="directory of reward plot")
    parser.add_argument('--config_file', type=str, required=False, help="config params")
    parser.add_argument('--delete_dir', type=bool, required=False, help="config params")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    file_path_config_file = args.config_file
    base_results_dir = args.base_results_dir
    delete_dir = args.delete_dir

    if delete_dir:
        remove_and_create_dir(base_results_dir)

    nominal_congestion_trace = [3.5,.2,.4,.5,.7,1,2,5,7,6,3,2,1,4,7,10,12]

    # perturb c a bit to see distro of rewards
    ########################################
    noise_offset = 0.5
    congestion_trace = [np.max([0.01, c - noise_offset]) for c in nominal_congestion_trace]

    congestion_var = 'CONGESTION'
    thpt_var = 'THROUGHPUT'
    num_sess_var = 'CELLT_AVG_NUM_SESS'
    specf_var = 'CELLT_AGG_SPECF_DL'
    
    master_cell_records = pandas.DataFrame()
    master_cell_records['TIME_INDEX'] = range(len(congestion_trace))
    master_cell_records[congestion_var] = congestion_trace

    # discretize the state space
    ########################################
    min_y = np.min(congestion_trace)*0.9
    max_y = np.max(congestion_trace)*1.1

    NUM_STATE_BINS = 15
    #NUM_STATE_BINS = 5

    discretized_y = np.linspace(min_y, max_y, NUM_STATE_BINS) 

    stateIndex_to_state, state_to_stateIndex = discrete_state_cont_y_conversion_dicts(discretized_y = discretized_y, print_mode = True)

    states = state_to_stateIndex.keys()
    print('states ', states)

    for query_y in [0.5, 1.3, 15.2, 13.0, 7.4]:
        closest_bin_index, closest_y = get_discrete_state_from_cont_query(query_y = query_y, discretized_y = discretized_y, print_mode = True)


    # discretize the action space
    ########################################
    min_action = 0.0
    max_action = 1.0

    NUM_ACTION_BINS = 10

    discretized_action = np.linspace(min_action, max_action, NUM_ACTION_BINS) 
    actionIndex_to_action, action_to_actionIndex = discrete_state_cont_y_conversion_dicts(discretized_y = discretized_action, print_mode = True)

    actions = actionIndex_to_action.keys()
    print('actions ', actions)

    # problem parameters for DP
    ########################################
    # horizon
    T = len(congestion_trace) - 1

    # discrete, state, actions
    GAMMA = 0.99

    TEST_EXHAUSTIVE = False

    if TEST_EXHAUSTIVE == True:
        T = 5
        NUM_ACTION_BINS = 4

    # parameters
    problem_params_dict = {}
    problem_params_dict['states'] = states
    state_space_dim = len(states)
    problem_params_dict['state_space_dim'] = state_space_dim

    problem_params_dict['actions'] = actions
    action_space_dim = len(actions)
    problem_params_dict['action_space_dim'] = action_space_dim

    problem_params_dict['T'] = T
    problem_params_dict['GAMMA'] = GAMMA
    problem_params_dict['base_results_dir'] = base_results_dir
    problem_params_dict['problem_number'] = 0.0
    problem_params_dict['cell'] = 'test'
    problem_params_dict['day'] = 'fakeDay'
    cell_day = '.'.join(['cell', problem_params_dict['cell'], 'day', problem_params_dict['day']])

    problem_params_dict['cell_day'] = cell_day
    #problem_params_dict['compute_mode'] = 'SEQUENTIAL'
    problem_params_dict['compute_mode'] = 'PARALLEL'
    problem_params_dict['print_interval'] = 1

    state_action_pairs_index_list = []
    for state_index, state in enumerate(problem_params_dict['states']):
        for action_index, action in enumerate(problem_params_dict['actions']):
            state_action_pairs_index_list.append((state_index, action_index))
    problem_params_dict['state_action_pairs_index_list'] = state_action_pairs_index_list



    # trans prob params 
    trans_prob_params = {}
    trans_prob_params['epsilon'] = 0.0
    trans_prob_params['M'] = 1.0
    trans_prob_params['master_cell_records'] = master_cell_records
    trans_prob_params['congestion_var'] = 'CONGESTION'
    trans_prob_params['time_index_var'] = 'TIME_INDEX'
    trans_prob_params['stateIndex_to_state'] = stateIndex_to_state
    trans_prob_params['actionIndex_to_action']  = actionIndex_to_action
    trans_prob_params['discretized_state_values'] = discretized_y
    trans_prob_params['states'] = states
    trans_prob_params['numerical_tolerance'] = 0.001

    reward_params_dict = {}
    reward_params_dict['alpha'] = 5.0
    reward_params_dict['beta'] = 1.0
    reward_params_dict['kappa'] = 1.0
    reward_params_dict['burst_prob_user_selector'] = 'same_as_IOT'
    reward_params_dict['control_interval_seconds'] = 1
    reward_params_dict['avg_user_burst_prob'] = 0.10
    reward_params_dict['KB_MB_converter'] = 1
    reward_params_dict['RF_mode'] = False
    reward_params_dict['hard_thpt_limit_flag'] = True
    reward_params_dict['CE_mode'] = 'NONE'
    reward_params_dict['specf_var'] = specf_var
    reward_params_dict['num_sess_var'] = num_sess_var

    thpt_trace = [throughput_model(congestion = c, reward_params_dict = reward_params_dict) for c in congestion_trace]
    master_cell_records[thpt_var] = thpt_trace
    reward_params_dict['hard_thpt_limit'] = master_cell_records[thpt_var].quantile(0.50)

    reward_table = get_reward_table(problem_params_dict = problem_params_dict, reward_params_dict = reward_params_dict, print_mode = False, trans_prob_params = trans_prob_params)
    reward_params_dict['reward_table'] = reward_table

    print_mode = False
    compute_Q_table_finite_horizon(problem_params_dict = problem_params_dict,  print_mode = print_mode, trans_prob_params = trans_prob_params, reward_params_dict = reward_params_dict)

    Q_table = load_Q_table(problem_params_dict = problem_params_dict)

    # problem params
    #########################################################
    reward_params_pkl = problem_params_dict['base_results_dir'] + '/reward_params.pkl'
    joblib.dump(reward_params_dict, reward_params_pkl)

    trans_prob_params_pkl = problem_params_dict['base_results_dir'] + '/trans_prob_params.pkl'
    joblib.dump(trans_prob_params, trans_prob_params_pkl)

    problem_params_pkl = problem_params_dict['base_results_dir'] + '/problem_params.pkl'
    joblib.dump(problem_params_dict, problem_params_pkl)

    # problem params
    # BEGIN HERE
    #########################################################
    N_trials = 1

    # now execute the control program
    for trial in range(N_trials):
        print('trial', trial)
        
        print('OPTIMAL STRATEGY')
        print('######################')
        Q_optimization_results_df, Q_total_discounted_reward, Q_total_reward = execute_control_strategy(Q_table = Q_table, problem_params_dict = problem_params_dict,  print_mode = True, trans_prob_params = trans_prob_params, reward_params_dict = reward_params_dict, action_sequence_vec = None, control_strategy = 'Q_policy')
        print('optimization_results_df', Q_optimization_results_df)
        print('total_discounted_reward', Q_total_discounted_reward)

        Q_results_ts_csv = problem_params_dict['base_results_dir'] + '/ts.Q_table.csv'
        Q_optimization_results_df.to_csv(Q_results_ts_csv)

        print(' ')
        print('######################')

        print('SUBOPTIMAL STRATEGY [ALL 0]')
        print('######################')
        action_sequence_vec = [0 for t in range(problem_params_dict['T'])]
        optimization_results_df, total_discounted_reward, total_reward = execute_control_strategy(Q_table = Q_table, problem_params_dict = problem_params_dict,  print_mode = print_mode, trans_prob_params = trans_prob_params, reward_params_dict = reward_params_dict, action_sequence_vec = action_sequence_vec, control_strategy = 'suboptimal')
        print('optimization_results_df', optimization_results_df)
        print('total_discounted_reward', total_discounted_reward)
    
        print(' ')
        print('######################')

        print('SUBOPTIMAL STRATEGY [ALL MAX]')
        print('######################')
        action_sequence_vec = [actions[-1] for t in range(problem_params_dict['T'])]
        optimization_results_df, total_discounted_reward, total_reward = execute_control_strategy(Q_table = Q_table, problem_params_dict = problem_params_dict,  print_mode = print_mode, trans_prob_params = trans_prob_params, reward_params_dict = reward_params_dict, action_sequence_vec = action_sequence_vec, control_strategy = 'suboptimal')
        print('optimization_results_df', optimization_results_df)
        print('total_discounted_reward', total_discounted_reward)
    
        print(' ')
        print('######################')




    # now exhaustively search over all actions to test them
    ##################################################################
    if TEST_EXHAUSTIVE:

        all_action_vec_iterator = itertools.product(actions, repeat = problem_params_dict['T'])

        action_vector_to_value_dict = {}

        print('######################')
        print(' OPTIMAL ACTION STR ')
        print('######################')

        total_num_action_vecs = len(actions)**(problem_params_dict['T']) 
        print(' total sequences to enumerate ', total_num_action_vecs)
        print(' ')

        for idx, action_sequence_vec in enumerate(all_action_vec_iterator):

            action_sequence_vec_str = '_'.join([str(x) for x in action_sequence_vec])
          
            optimization_results_df, total_discounted_reward, total_reward = execute_control_strategy(Q_table = None, problem_params_dict = problem_params_dict,  print_mode = print_mode, trans_prob_params = trans_prob_params, reward_params_dict = reward_params_dict, action_sequence_vec = action_sequence_vec, control_strategy = 'empirical')

            action_vector_to_value_dict[action_sequence_vec_str] = total_discounted_reward

        max_dict_element = max(action_vector_to_value_dict.iteritems(), key=operator.itemgetter(1))

        optimal_action_sequence_vec_str = max_dict_element[0]
        optimal_discounted_reward = max_dict_element[1]
        Q_optimal_action_sequence_vec_str = '_'.join([str(int(x)) for x in list(Q_optimization_results_df['ACTION_INDEX'])])

        print('empirical optimal action seq vec str', optimal_action_sequence_vec_str)
        print('optimal discounted reward', optimal_discounted_reward)
        print('Q optimal policy', Q_optimal_action_sequence_vec_str)
        print('Q total discounted reward', Q_total_discounted_reward)
        print('value for Q policy check', action_vector_to_value_dict[Q_optimal_action_sequence_vec_str])

        #print('action_vector_to_value_dict')        
        print('action_vector_to_value_dict')        

        sorted_actions = sorted(action_vector_to_value_dict.items(), key=operator.itemgetter(1))

        sorted_actions.reverse()

        N = 10
        print('top N actions', N)
        for sorted_action_tuple in sorted_actions[0:N]:
            print(sorted_action_tuple[0], sorted_action_tuple[1])
            print(' ')
       
        Q_results_ts_csv = problem_params_dict['base_results_dir'] + '/ts.Q_table.csv'
        Q_optimization_results_df.to_csv(Q_results_ts_csv)
