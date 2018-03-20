# Author: Sandeep Chinchali
import sys, os
import pandas
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import collections
import datetime, time
from datetime import timedelta
import joblib

import pylab as pl
import numpy as np
import seaborn
import ConfigParser

# run experiments in parallel
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

# plotting utils
AAAI_ROOT_DIR = os.environ['AAAI_ROOT_DIR']
sys.path.append(AAAI_ROOT_DIR + '/plot/')
import argparse

from plotting_utils import overlaid_ts

"""
compute discounted rewards
"""
def compute_discounted_reward(reward_vec = None, GAMMA = None):
    DP_reward = np.nansum(np.array(reward_vec))
    print('reward ', DP_reward)

    cumulative_reward_vec = [reward_vec[t]*GAMMA**t for t in range(len(reward_vec))]
    
    DP_cumulative_reward = np.nansum(np.array(cumulative_reward_vec))
    
    mean_DP_cumulative_reward = np.nanmean(cumulative_reward_vec)
    
    mean_DP_reward = np.nanmean(reward_vec)
    
    print('cumulative reward ', DP_cumulative_reward)

    reward_results_vector = [DP_reward, DP_cumulative_reward, mean_DP_reward, mean_DP_cumulative_reward, len(reward_vec)]

    return reward_results_vector


def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config_file', type=str, required=False, help="config params")
    parser.add_argument('--base_results_dir', type=str, required=False, help="directory of reward plot")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    saved_Q_results_dir = args.base_results_dir
    plot_saved_Q_results_dir = args.base_results_dir
    #file_path_config_file = args.config_file
    #config = ConfigParser.ConfigParser()
    #config.read(file_path_config_file)

    # key things to have in title: horizon T, name of environment, variable names

    # get the experiment data dir
    ###########################################################
    #base_plot_dir = config.get('PATHS', 'base_plot_dir')
    #saved_Q_results_dir = config.get('PATHS', 'saved_Q_results_dir')
   
    ## now see how well the quantized version did
    ######################################################################################
    #algo_type = config.get('DP_PARAMS', 'algo_type')
    #env_type = config.get('DP_PARAMS', 'env_type')
    #plot_time_var = config.get('DP_PARAMS', 'plot_time_var')
    plot_time_var = 'TIME_INDEX'

    algo_type = 'DP' 
    env_type = 'PPC'

    ###########################################################
    reward_params_pkl = saved_Q_results_dir + '/' + '.'.join(['reward_params.pkl'])
    reward_params_dict = joblib.load(reward_params_pkl)
    print('reward_params_dict', reward_params_dict)

    problem_params_pkl = saved_Q_results_dir + '/' + '.'.join(['problem_params.pkl'])
    problem_params_dict = joblib.load(problem_params_pkl)
    print('problem_params_dict', problem_params_dict)

    trans_prob_params_pkl = saved_Q_results_dir + '/' + '.'.join(['trans_prob_params.pkl'])
    trans_prob_params = joblib.load(trans_prob_params_pkl)
    print('reward_params_dict', reward_params_dict)

    original_master_cell_records = trans_prob_params['master_cell_records'].copy()
    print(original_master_cell_records.head())

    Q_results_ts_csv = saved_Q_results_dir + '/' + '.'.join(['ts.Q_table.csv'])
    Q_optimization_results_df = pandas.read_csv(Q_results_ts_csv)

    # plot actual congestion, DP congestion vs time
    #####################################################
    congestion_var = trans_prob_params['congestion_var']
    discretized_controlled_congestion_var = 'DISCRETIZED_CONTROLLED_' + congestion_var
    continuous_controlled_congestion_var = 'CONTINUOUS_CONTROLLED_' + congestion_var
   
    DP_RL_dict = {}
    reward_vec = list(Q_optimization_results_df['REWARD'])
    reward_results_vector = compute_discounted_reward(reward_vec = reward_vec, GAMMA = problem_params_dict['GAMMA'])
    DP_RL_dict['DP'] = reward_results_vector

    GLOBAL_LW = 1.0
    GLOBAL_LS = '-'
    GLOBAL_MARKER = '.'

    ## plot congestion overlaid
    ################################################
    action_var = 'ACTION'
    reward_var = 'REWARD'
    congestion_ts_dict = collections.OrderedDict()
    for col_to_plot in [action_var, discretized_controlled_congestion_var]:
        ts_data_dict = collections.OrderedDict()
        ts_data_dict['xvec'] = list(Q_optimization_results_df[plot_time_var])
        ts_data_dict['ts_vector'] = list(Q_optimization_results_df[col_to_plot])
        ts_data_dict['lw'] = GLOBAL_LW
        ts_data_dict['linestyle'] = GLOBAL_LS
        ts_data_dict['marker'] = GLOBAL_MARKER
        congestion_ts_dict[col_to_plot] = ts_data_dict

    # plot the columns
    display_label = lambda a, b, k, M, K, algo_type, cumulative_reward, experiment_num: r'$\alpha = %d, \beta = %d, \kappa = %d $, M = %d, Limit = %d, Algo = %s, Discounted R = %.3f, env = %s' % (a, b, k, M, K, algo_type, DP_RL_dict['DP'][1], env_type)

    title_str = display_label(reward_params_dict['alpha'], reward_params_dict['beta'], reward_params_dict['kappa'], trans_prob_params['M'], int(reward_params_dict['hard_thpt_limit']), algo_type, DP_RL_dict['DP'][1], env_type)
   
    overlaid_congestion_plot_file = plot_saved_Q_results_dir + '/' + '.'.join(['overlaid.DP', congestion_var, 'pdf'])
    overlaid_ts(normalized_ts_dict = congestion_ts_dict, title_str = title_str, plot_file = overlaid_congestion_plot_file, ylabel = congestion_var, xlabel = 'MINUTE_OF_DAY_LOCAL', fontsize = 10)

    ## plot individual ts
    ################################################
    for col_to_plot in [action_var, congestion_var, discretized_controlled_congestion_var, continuous_controlled_congestion_var, reward_var]:
        action_ts_dict = collections.OrderedDict()
        ts_data_dict = collections.OrderedDict()
        ts_data_dict['xvec'] = list(Q_optimization_results_df[plot_time_var])
        ts_data_dict['ts_vector'] = list(Q_optimization_results_df[col_to_plot])
        ts_data_dict['lw'] = GLOBAL_LW
        ts_data_dict['linestyle'] = GLOBAL_LS
        ts_data_dict['marker'] = GLOBAL_MARKER
        action_ts_dict[col_to_plot] = ts_data_dict

        overlaid_action_plot_file = plot_saved_Q_results_dir + '/' + '.'.join(['individual.DP', col_to_plot, 'pdf'])
        overlaid_ts(normalized_ts_dict = action_ts_dict, title_str = title_str, plot_file = overlaid_action_plot_file, ylabel = col_to_plot, xlabel = 'MINUTE_OF_DAY_LOCAL', fontsize = 10)
    
    ## what were the total gains for the RL timeseries?
    ################################################
    DP_results_df = pandas.DataFrame()
    DP_results_df['total_reward'] = [DP_RL_dict['DP'][0]]
    DP_results_df['total_discounted_reward'] = [DP_RL_dict['DP'][1]]
    DP_results_df['num_pts'] = [DP_RL_dict['DP'][-1]]
    DP_results_df['state_space_dim'] = [problem_params_dict['state_space_dim']]
    DP_results_df['action_space_dim'] = [problem_params_dict['action_space_dim']]

    DP_results_file = plot_saved_Q_results_dir + '/' + '.'.join(['onlyDP.results.txt'])
    DP_results_df.to_csv(DP_results_file, sep = '\t')
