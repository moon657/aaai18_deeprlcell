import gym
from gym import spaces
from gym.utils import seeding
import sys,os
import numpy as np
from os import path
import cPickle
import json
import pandas

# plotting utils
RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + 'utils/'
sys.path.append(util_dir)

# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

# simulators
simulators_dir = RL_ROOT_DIR + '/simulate_env/simulators/'
sys.path.append(simulators_dir)

from evaluate_DDPG_utils import load_DDPG_dict, wrapper_run_DDPG_experiment, adjust_timestamp_CELX
from textfile_utils import list_from_textfile, remove_and_create_dir

def parse_args():
        parser = argparse.ArgumentParser(description='call simple cell simulator')
	parser.add_argument('--base_results_dir', type=str, required=False, default='/Users/csandeep/Documents/work/uhana/work/20161128/simple_cell_sim/')
	parser.add_argument('--experiment_prefix', type=str, required=False, default='RL_TELSTRA')
	parser.add_argument('--print_mode', type=bool, required=False, default=False)
	parser.add_argument('--KPI_to_plot_file', type=str, required=False, default='/Users/csandeep/Documents/work/uhana/work/20161128/simple_cell_sim/KPI.txt')
	return parser.parse_args()

if __name__ == "__main__":
  	args = parse_args()

	# basic args
	# where plots go
	base_results_dir = args.base_results_dir
	# how to identify experiment
	experiment_prefix = args.experiment_prefix
	print_mode = args.print_mode
	# KPIs of interest
	KPI_to_plot_file = args.KPI_to_plot_file

	cell_id = 'test'

	###############################################
	# load the burst probs action space
	burst_discretization = .025
	burst_prob_params_dict = load_burst_prob_params_dict(burst_discretization=burst_discretization, max_burst_prob=1.0, min_burst_prob = .1, optimal_burst_prob=.25)	

	# for the reward computation
	reward_params_dict =  load_reward_params_dict(alpha=1.0, beta=1.0, k=1, control_interval_seconds=60, avg_user_burst_prob=.05)
	# B = B_0/C
	reward_params_dict['B_0'] = 1.0
	# modulate hard thpt penalty: -activity*kappa*(K-B')
	reward_params_dict['kappa'] = 1.0
	# if thpt goes below this for N timepoints, abort
	reward_params_dict['hard_thpt_limit'] = .15

	# state is just collision metric
	min_state_vector = np.array([1])
	max_state_vector = np.array([6])

	# to initialize the cell simulator
	env_params_dict = {}
	env_params_dict['continuous_action_mode'] = True
	env_params_dict['deterministic_reset_mode'] = False
	env_params_dict['print_mode'] = False
	env_params_dict['reward_params_dict'] = reward_params_dict
	env_params_dict['burst_prob_params_dict'] = burst_prob_params_dict
	env_params_dict['min_state_vector'] = min_state_vector
	env_params_dict['max_state_vector'] = max_state_vector
	env_params_dict['reset_state_vector'] = min_state_vector
	env_params_dict['thpt_var'] = 'THPT'

	# episode is at least 15 steps before termination
	env_params_dict['min_iterations_before_done'] = 15
	# if THPT below bad_thpt_threshold for num_last_entries AND we have surpassed min_iterations, env can abort prematurely
        env_params_dict['num_last_entries'] = 10
	env_params_dict['bad_thpt_threshold'] = .75* reward_params_dict['hard_thpt_limit']
	env_params_dict['hard_thpt_limit_flag'] = False

	# DDPG function
	######################################################################
	# how many episodes for DDPG algorithm
	agent_type = 'DDPG'
	DDPG_dict = load_DDPG_dict(MAX_STEP=75, EPISODES=1500, TEST_EPISODES=5, TEST_TERM=50, PRINT_LEN=40, SAVE_TERM=1)
	experiment_params = experiment_prefix

	# info for plotting
	#####################
	plotting_info_dict = {}
	plotting_info_dict['experiment_params'] = 'simpleCell_early'
	plotting_info_dict['KPI_to_plot_file'] = KPI_to_plot_file 
	plotting_info_dict['base_results_dir'] = base_results_dir
	plotting_info_dict['cell_id'] = cell_id
	plotting_info_dict['time_var'] = 'ITERATION_INDEX'

	# run the DDPG agent, record results, write plots
	reward_history_df, DDPG_rewards_df, env, DDPG_rewards, DDPG_wt, algo = simple_evaluate_DDPG(env_params_dict = env_params_dict, DDPG_dict = DDPG_dict, plotting_info_dict= plotting_info_dict, agent_type = agent_type)
