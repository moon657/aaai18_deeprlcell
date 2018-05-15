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

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import cPickle
import json
import pandas
import sys,os


# load utils from the transition matrix
RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + '/learning_suite/python/control_theory_code/state_space_generation'
sys.path.append(util_dir)

util_dir = RL_ROOT_DIR + '/learning_suite/python/utils/'
sys.path.append(util_dir)

actor_critic_utils = RL_ROOT_DIR + '/working_dir/tchu'
sys.path.append(actor_critic_utils)

print(sys.path)
print(RL_ROOT_DIR)

from algorithm import *
import train_util as tu

from external_Q_agent_implementation import *
from panel_plot_timeseries import plot_loop_KPI
from double_panel_plot_timeseries import panel_KPI_plot
from textfile_utils import list_from_textfile
from external_Q_agent_implementation import QLearner
from reward_computation_utils import *
from time_variant_transition_matrix import state_to_vector, continuous_state_to_state_num 
from plot_reward_across_agents import *
from simple_cell_simulator import *

def init_Q_learner(state_value_map_df, num_actions):
	num_states = state_value_map_df.shape[0]
	learner = QLearner(num_states= num_states,
		       num_actions= num_actions,
		       alpha=0.2,
		       gamma=1,
		       random_action_rate=0.5,
		       random_action_decay_rate=0.99)

	return learner

# done
def write_Q_table(learner, burst_prob_vector, num_trials, base_results_dir, experiment_params_str):
	# if discrete Q, save the Q table to a file
	Q_df = pandas.DataFrame()
	for row in range(learner.qtable.shape[0]):
		print('state_num', 'best_action_index', 'max_value', 'optimal_burst_prob')

		best_action_index = learner.qtable[row].argmax()
		max_reward = learner.qtable[row].max()
		optimal_burst_prob = burst_prob_vector[best_action_index]
		print(row, best_action_index, max_reward, optimal_burst_prob)
		#print (row, learner.qtable[row].argmax(), learner.qtable[row].max(), burst_prob_vector[learner.qtable[row].argmax()])

		state_Q_table_dict = {}
		state_Q_table_dict['NUM_TRIALS'] = [str(num_trials)]
		state_Q_table_dict['STATE_INDEX'] = [row]
		state_Q_table_dict['OPTIMAL_BURST_PROB'] = [optimal_burst_prob] 
		state_Q_table_dict['MAX_REWARD'] = [max_reward]

		state_Q_table_df = pandas.DataFrame(state_Q_table_dict)
		Q_df = Q_df.append(state_Q_table_df)

	Q_csv = base_results_dir + '/Qtable.' + experiment_params_str + '.csv'
	Q_df.to_csv(Q_csv, inplace=True)
	return Q_df

def init_cell_environment(env_dict):
        
	CELX_df = env_dict['CELX_df']
	transition_matrix_fname = env_dict['transition_matrix_fname']
	burst_prob_params_dict = env_dict['burst_prob_params_dict']
	cell_id = env_dict['cell_id']
	print_mode = env_dict['print_mode']
	reset_state_vector = env_dict['reset_state_vector']
	deterministic_reset_mode = env_dict['deterministic_reset_mode']
	RF_model_pkl_file = env_dict['RF_model_pkl_file']
	sampling_mode = env_dict['sampling_mode']
	reward_params_dict = env_dict['reward_params_dict']
	KPI_list = env_dict['KPI_list']
	min_state_vector = env_dict['min_state_vector']
	max_state_vector = env_dict['max_state_vector']
	time_var = env_dict['time_var']
	date_var = env_dict['date_var']
	thpt_var = env_dict['thpt_var']
	delay_reward_interval = env_dict['delay_reward_interval']
	delayed_reward_mode = env_dict['delayed_reward_mode']
	continuous_action_mode = env_dict['continuous_action_mode']

	state_value_map_df = env_dict['state_value_map_df']

	max_thpt_value = 100
	env = CellSimulatorEnv(CELX_df = CELX_df, transition_matrix_fname = transition_matrix_fname, burst_prob_params_dict = burst_prob_params_dict, cell_id = cell_id, print_mode = print_mode, reset_state_vector = reset_state_vector, deterministic_reset_mode = deterministic_reset_mode, RF_model_pkl_file = RF_model_pkl_file, sampling_mode = sampling_mode, reward_params_dict = reward_params_dict, KPI_list = KPI_list, min_state_vector = min_state_vector, max_state_vector = max_state_vector, time_var = time_var, thpt_var = thpt_var, delay_reward_interval = delay_reward_interval, delayed_reward_mode = delayed_reward_mode, date_var=date_var, continuous_action_mode = continuous_action_mode, state_value_map_df= state_value_map_df, max_thpt_value=max_thpt_value)

	observation = env.reset()
	#print('observation', observation)
	action_space = env.action_space
	#print('action_space', action_space)
	return observation, action_space, env

def get_init_action(observation, action_space, env, agent_type, Q_agent_dict):
	# create a random action
	action = action_space.sample()


	# for Q agents set the state
	if(agent_type == 'discreteQ'):
		learner = Q_agent_dict['learner']
		KPI_list = Q_agent_dict['KPI_list']
		RF_cut_KPI_list = Q_agent_dict['RF_cut_KPI_list']
		state_value_map_df = Q_agent_dict['state_value_map_df']
		
		# how an observation looks
		# array([3.0178959196556701, 1.27858301041608, 55.513216662406904,
		# 	       145.204088394823, 3577.1780667469611], dtype=object)

		# state is a number
		# state = f(observation)
		state_df = extract_state_dict(observation, KPI_list)				
		state = continuous_state_to_state_num(state_value_map_df, state_df, RF_cut_KPI_list)
		action = learner.set_initial_state(state)
	return action

def simple_evaluate_DDPG(env = None, DDPG_dict=None, plotting_info_dict=None, agent_type=None):
    algo = DDPG(env.observation_space, env.action_space, WARMUP=DDPG_dict['WARMUP'])
    
    # get the action mode correct
    DDPG_rewards, DDPG_wt = tu.train(algo, env, MAX_STEP=DDPG_dict['MAX_STEP'], EPISODES=DDPG_dict['EPISODES'], TEST_EPISODES=DDPG_dict['TEST_EPISODES'], TEST_TERM=DDPG_dict['TEST_TERM'], SAVE_TERM=DDPG_dict['SAVE_TERM'], exp_mode='ou_noise', env_name='cell')
    
    experiment_params = plotting_info_dict['experiment_params']
    KPI_to_plot_file = plotting_info_dict['KPI_to_plot_file']
    base_results_dir = plotting_info_dict['base_results_dir']
    cell_id = plotting_info_dict['cell_id'] 
    time_var = plotting_info_dict['time_var']
    
    rmeans = DDPG_rewards[:1][0]
    rstds = DDPG_rewards[1:][0]
    train_reward_plot = base_results_dir + '/' + experiment_params + '.batchRewards.png'
    tu.plot_train_rewards(rmeans, rstds, fig_path=train_reward_plot, xlabel='Training episodes', ylabel='Total rewards', x=None)
    
    DDPG_rewards_df = pandas.DataFrame({'mean_reward': list(DDPG_rewards[0]), 'std_reward': list(DDPG_rewards[1])})
    experiment_reward_results_csv = base_results_dir + '/' + experiment_params + '.batchRewards.csv'
    DDPG_rewards_df.to_csv(experiment_reward_results_csv, index=False)
    
    batch_var = 'BATCH_NUM'
	# get data for unconverged/converged neural net performance
	#####################
    reward_history_df = env.reward_history_df
    reward_history_df.set_index(time_var, inplace=True)
    
    early_batch_index = reward_history_df[batch_var].min()
    late_batch_index = reward_history_df[batch_var].max()
    
    early_batch_df = reward_history_df[reward_history_df[batch_var] == early_batch_index]
    late_batch_df = reward_history_df[reward_history_df[batch_var] == late_batch_index]

	# start the plots
	#####################

	# plot first batch KPIs
    plotting_info_dict['experiment_params'] = 'simpleCell_early'
    no_datetime_overlay_KPI_plot(early_batch_df, plotting_info_dict)

	# plot last batch converged KPIs
    plotting_info_dict['experiment_params'] = 'simpleCell_late'
    no_datetime_overlay_KPI_plot(late_batch_df, plotting_info_dict)
    return reward_history_df, DDPG_rewards_df, env, DDPG_rewards, DDPG_wt, algo, late_batch_df, early_batch_df

def evaluate_DDPG(env_dict, DDPG_dict=None, plotting_info_dict = None, agent_type=None):

	plotting_info_dict['agent_type'] = agent_type

	# PARAMETERS FOR PLOTTING
	train_test_str = plotting_info_dict['TRAIN_TEST']
	experiment_prefix = plotting_info_dict['experiment_prefix']
	delay_reward_interval = plotting_info_dict['delay_reward_interval']
	base_results_dir = plotting_info_dict['base_results_dir']
	KPI_to_plot_file = plotting_info_dict['KPI_to_plot_file']
	cell_id = plotting_info_dict['cell_id']
	datetime_mode = plotting_info_dict['datetime_mode']
	time_var = plotting_info_dict['time_var']
	date_var = plotting_info_dict['date_var']
	fmt = plotting_info_dict['fmt']
	marker_plot_mode = plotting_info_dict['marker_plot_mode']
	start_time = plotting_info_dict['start_time']
	end_time = plotting_info_dict['end_time']
	cumulative_results_dir = plotting_info_dict['cumulative_results_dir']

	train_day_list = plotting_info_dict['train_day_list']
	test_day_list = plotting_info_dict['test_day_list']
	train_test_day_str = plotting_info_dict['train_test_day_str']

	experiment_params = '_'.join([experiment_prefix, cell_id, agent_type, 'delay', str(delay_reward_interval), train_test_str])

	print('evaluate ', agent_type)
	observation, action_space, env = init_cell_environment(env_dict)

	if(agent_type == 'DDPG'):
		algo = DDPG(env.observation_space, env.action_space, WARMUP=6000, GAMMA=0)
	elif(agent_type == 'LRQ'):
		algo = LRQ(env.observation_space, env.action_space)

	
	#DDPG_rewards = tu.train_daily_episode(algo, env, DDPG_dict)
	DDPG_rewards, DDPG_wt = tu.train(algo, env, MAX_STEP=DDPG_dict['MAX_STEP'], EPISODES=DDPG_dict['EPISODES'], TEST_EPISODES=DDPG_dict['TEST_EPISODES'], TEST_TERM=DDPG_dict['TEST_TERM'], SAVE_TERM=DDPG_dict['SAVE_TERM'], exp_mode='ou_noise', env_name='cell')

	#DDPG_rewards = tu.train_daily_episode(algo, env, DDPG_dict)
	DDPG_rewards_df = pandas.DataFrame({'mean_reward': list(DDPG_rewards[0]), 'std_reward': list(DDPG_rewards[1])})

	experiment_reward_results_csv = base_results_dir + '/' + experiment_params + '.batchRewards.csv'
	DDPG_rewards_df.to_csv(experiment_reward_results_csv, index=False)

	rmeans = DDPG_rewards[:1][0] 
	rstds = DDPG_rewards[1:][0]
	
	DDPG_dict['TRAIN_REWARD_DF'] = DDPG_rewards_df
	DDPG_dict['DDPG_REWARDS'] = DDPG_rewards

	train_reward_plot = base_results_dir + '/' + experiment_params + '.batchRewards.png'
	tu.plot_train_rewards(rmeans, rstds, fig_path=train_reward_plot, xlabel='Training episodes', ylabel='Total rewards', x=None)
	###################################################################################
	# now save the reward results to a file
	experiment_reward_results_csv = base_results_dir + '/' + experiment_params + '.rewards.csv'
	reward_history_df = env.reward_history_df
	reward_history_df.to_csv(experiment_reward_results_csv, index=False)

	# now do a plot of reward vs time, C vs time, all metrics vs time
	batch_var = 'BATCH_NUM'
	last_batch_df, complete_experiments_df = get_last_batch_data(reward_history_df)

	# plot reward vs BATCH_NUM
	agent_list = [agent_type]
	agent_results_dir = base_results_dir + '/' + agent_type
	#plot_all_agent_reward(base_results_dir, agent_list, plotting_info_dict)

	# update this to have the dataframe of interest
	#panel_KPI_plot(last_batch_df, plotting_info_dict)

	return DDPG_dict

def evaluate_learning_agent(num_batches, batch_size, agent_type, env_dict, burst_prob_params_dict, Q_agent_dict=None, plotting_info_dict = None):


	# PARAMETERS FOR PLOTTING
	train_test_str = plotting_info_dict['TRAIN_TEST']
	experiment_prefix = plotting_info_dict['experiment_prefix']
	delay_reward_interval = plotting_info_dict['delay_reward_interval']
	base_results_dir = plotting_info_dict['base_results_dir']
	KPI_to_plot_file = plotting_info_dict['KPI_to_plot_file']
	cell_id = plotting_info_dict['cell_id']
	agent_type = plotting_info_dict['agent_type']
	datetime_mode = plotting_info_dict['datetime_mode']
	time_var = plotting_info_dict['time_var']
	date_var = plotting_info_dict['date_var']
	fmt = plotting_info_dict['fmt']
	marker_plot_mode = plotting_info_dict['marker_plot_mode']
	start_time = plotting_info_dict['start_time']
	end_time = plotting_info_dict['end_time']
	experiment_params = plotting_info_dict['experiment_params']
	cumulative_results_dir = plotting_info_dict['cumulative_results_dir']

	train_day_list = plotting_info_dict['train_day_list']
	test_day_list = plotting_info_dict['test_day_list']
	train_test_day_str = plotting_info_dict['train_test_day_str']

	# start the env
	observation, action_space, env = init_cell_environment(env_dict)

	action = get_init_action(observation, action_space, env, agent_type, Q_agent_dict)
	
	# action space params
	action_space = burst_prob_params_dict['action_space'] 
	num_actions = burst_prob_params_dict['num_actions'] 
	burst_prob_to_action = burst_prob_params_dict['burst_prob_to_action'] 
	action_to_burst_prob = burst_prob_params_dict['action_to_burst_prob']
	burst_prob_vector = burst_prob_params_dict['burst_prob_vector']
	optimal_discretized_burst = burst_prob_params_dict['optimal_discretized_burst']

	num_trials = num_batches * batch_size

	for trial in range(num_trials):
		# create a step
		if(trial % batch_size == 0):
			progress_str = ' '.join(['agent', agent_type, 'batch', str(np.floor(trial/batch_size))])
			print(progress_str) 

		observation, reward, done, info = env.step(action)

		if(agent_type == 'random'):
			action = action_space.sample()
			#print('executing agent ', agent_type) 
		elif(agent_type == 'minBurst'):
			if(env.continuous_action_mode):
				action = np.array([min_burst_prob])
				#print('executing agent ', agent_type) 
			else:
				action = burst_prob_to_action[burst_prob_vector[0]]
				#print('executing agent ', agent_type) 

		elif(agent_type == 'maxBurst'):
			if(env.continuous_action_mode):
				action = np.array([max_burst_prob])
				#print('executing agent ', agent_type) 
			else:
				action = burst_prob_to_action[burst_prob_vector[-1]]
				#print('executing agent ', agent_type) 

		elif(agent_type == 'optimal'):
			# map the optimal burst prob vector here
			action = burst_prob_to_action[optimal_discretized_burst]

		elif(agent_type == 'medianBurst'):
			# map the optimal burst prob vector here
			median_burst_prob = burst_prob_vector[num_actions/2]
			action = burst_prob_to_action[median_burst_prob]

		elif(agent_type == 'discreteQ'):
			# tabular Q agent
			# action = agent.act(ob)
			# action = learner.move(state_prime, reward)

			# external agent key functions
			# discretized Q
			RF_cut_KPI_list = Q_agent_dict['RF_cut_KPI_list']
			KPI_list = Q_agent_dict['KPI_list']
			learner = Q_agent_dict['learner']

			state_value_map_df = Q_agent_dict['state_value_map_df']
			state_prime_df = extract_state_dict(observation, KPI_list)	
			state_prime = continuous_state_to_state_num(state_value_map_df, state_prime_df, RF_cut_KPI_list)

			action = learner.move(state_prime, reward)
			# print('executing agent ', agent_type)
		else:
			action = action_space.sample()

	# save the trained learner for future use
	if(agent_type == 'discreteQ'):
		Q_agent_dict['learner'] = learner
		Q_agent_dict['Q_table'] = learner.qtable

	# now save the reward results to a file
	experiment_reward_results_csv = base_results_dir + '/' + experiment_params + '.rewards.csv'
	reward_history_df = env.reward_history_df
	reward_history_df.to_csv(experiment_reward_results_csv, index=False)

	# now do a plot of reward vs time, C vs time, all metrics vs time
	batch_var = 'BATCH_NUM'
	last_batch_df, complete_experiments_df = get_last_batch_data(reward_history_df)

	# plot reward vs BATCH_NUM
	agent_list = [agent_type]
	agent_results_dir = base_results_dir + '/' + agent_type
	plot_all_agent_reward(base_results_dir, agent_list, plotting_info_dict)

	# update this to have the dataframe of interest
	panel_KPI_plot(last_batch_df, plotting_info_dict)

	return Q_agent_dict
