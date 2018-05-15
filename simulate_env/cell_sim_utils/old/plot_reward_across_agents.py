import sys,os
import pandas
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import collections
import datetime,time
from datetime import timedelta
import bisect
import pylab as pl
import numpy as np
import argparse

# Author: Sandeep Chinchali
# Description: Given a dataframe csv, plot a timeseries for a KPI of interest and date range of interest
# load utils from the transition matrix
RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + '/utils/'
sys.path.append(util_dir)
from textfile_utils import list_from_file,list_from_textfile
from double_panel_plot_timeseries import *

def get_last_batch_data(experiment_rewards_df):

	# last batch number
	last_complete_batch = experiment_rewards_df['BATCH_NUMBER'].max()-1
	# cumulative reward by batch
	all_complete_experiments_df = experiment_rewards_df[experiment_rewards_df['BATCH_NUMBER'] <= last_complete_batch]
	last_complete_batch_df = experiment_rewards_df[experiment_rewards_df['BATCH_NUMBER'] == last_complete_batch]

	return last_complete_batch_df, all_complete_experiments_df


# plots cumulative reward across batches
def plot_all_agent_reward(simulation_results_dir, agent_list, plotting_info_dict):

  # PARAMETERS FOR PLOTTING
  train_test_str = plotting_info_dict['TRAIN_TEST']
  experiment_prefix = plotting_info_dict['experiment_prefix']
  delay_reward_interval = plotting_info_dict['delay_reward_interval']
  base_results_dir = plotting_info_dict['base_results_dir']
  KPI_to_plot_file = plotting_info_dict['KPI_to_plot_file']
  cell_id = plotting_info_dict['cell_id']
  datetime_mode = plotting_info_dict['datetime_mode']
  time_var = plotting_info_dict['time_var']
  fmt = plotting_info_dict['fmt']
  marker_plot_mode = plotting_info_dict['marker_plot_mode']
  start_time = plotting_info_dict['start_time']
  end_time = plotting_info_dict['end_time']
  experiment_params = plotting_info_dict['experiment_params']
  cumulative_results_dir = plotting_info_dict['cumulative_results_dir']

  #####################################################
  reward_columns = ['REWARD', 'DELAYED_REWARD']

  print('performance by batches')
  extract_columns = [time_var] + reward_columns
  for reward_var in reward_columns:
	  for agent_type in agent_list:
		  print(agent_type)
		  experiment_params = '_'.join([experiment_prefix, cell_id, agent_type, 'delay', str(delay_reward_interval), train_test_str])
		  experiment_reward_results_csv = simulation_results_dir + '/' + experiment_params + '.rewards.csv'
		  experiment_rewards_df = pandas.read_csv(experiment_reward_results_csv)

		  # last batch number
		  last_complete_batch = experiment_rewards_df['BATCH_NUMBER'].max()-1
		  # cumulative reward by batch
		  all_complete_experiments_df = experiment_rewards_df[experiment_rewards_df['BATCH_NUMBER'] <= last_complete_batch]
		  cumulative_batch_rewards_df = all_complete_experiments_df.groupby('BATCH_NUMBER').sum()

		  # plot batch number on x, cumulative reward var on y per agent
		  #cumulative_batch_rewards_df.set_index('BATCH_NUMBER')
		  bp = cumulative_batch_rewards_df[reward_var].plot(linestyle = '--', marker='o')
		  fig = bp.get_figure()
		  plt.ylabel('BATCH_SUM_' + reward_var)
		  # reward vs time for last epoch across agents


	  if(len(agent_list) > 1):
		  experiment_params = '_'.join([experiment_prefix, cell_id, 'ALL_AGENT', 'delay', str(delay_reward_interval), train_test_str])
	  title_str = experiment_params
	  plt.title(title_str + '\n')
	  batch_vs_reward_plot_name = cumulative_results_dir + '/batch.' + reward_var + '.' + experiment_params + '.png'
	  plt.legend(agent_list)
	  fig.savefig(batch_vs_reward_plot_name)
	  plt.close()

  print('print latest batch only')
  for reward_var in reward_columns:
	  for agent_type in agent_list:
		  print(agent_type)
		  experiment_params = '_'.join([experiment_prefix, cell_id, agent_type, 'delay', str(delay_reward_interval), train_test_str])
		  experiment_reward_results_csv = simulation_results_dir + '/' + experiment_params + '.rewards.csv'
		  experiment_rewards_df = pandas.read_csv(experiment_reward_results_csv)

		  # last batch number
		  last_complete_batch = experiment_rewards_df['BATCH_NUMBER'].max()-1
		  last_complete_batch_df = experiment_rewards_df[experiment_rewards_df['BATCH_NUMBER'] == last_complete_batch]
		  cumulative_batch_rewards_df = last_complete_batch_df

		  if(datetime_mode):
			  cumulative_batch_rewards_df.set_index(pandas.to_datetime(cumulative_batch_rewards_df[time_var]), inplace = True)
		  else:
			  cumulative_batch_rewards_df.set_index(time_var, inplace = True)


		  # plot batch number on x, cumulative reward var on y per agent
		  #cumulative_batch_rewards_df.set_index('BATCH_NUMBER')
		  bp = cumulative_batch_rewards_df.cumsum()[reward_var].plot(linestyle = '--', marker='o')
		  fig = bp.get_figure()
		  title_str = experiment_params
		  plt.title(title_str + '\n')
		  plt.ylabel(reward_var)

	  if(len(agent_list) > 1):
		  experiment_params = '_'.join([experiment_prefix, cell_id, 'ALL_AGENT', 'delay', str(delay_reward_interval), train_test_str])

	  batch_vs_reward_plot_name = cumulative_results_dir + '/lastBatch.' + reward_var + '.' + experiment_params + '.png'
	  plt.legend(agent_list)
	  fig.savefig(batch_vs_reward_plot_name)
	  plt.close()

def parse_args():
        parser = argparse.ArgumentParser(description='file plot')
        parser.add_argument('--simulation_results_dir', type=str, required=False, default='/Users/csandeep/Documents/work/uhana/work/20161128/RL_unit_test/output/results_RL_TELSTRA/')
        parser.add_argument('--base_results_dir', type=str, required=False, default='/Users/csandeep/Documents/work/uhana/work/20161128/RL_unit_test/output/results_RL_TELSTRA/test_cumulative/')
	parser.add_argument('--agent_list_file', type=str, required=False, default='/Users/csandeep/Documents/work/uhana/work/20161128/RL_unit_test/conf/agent_list.txt')
        parser.add_argument('--time_var', type=str, required=False, default='TIMESTAMP')
        parser.add_argument('--cell_id', type=str, required=False, default='136046093.160428')
	parser.add_argument('--experiment_params', type=str, required=False, default='RL_TELSTRA')
	parser.add_argument('--datetime_mode', type=bool, required=False, default=False)
	return parser.parse_args()

if __name__ == "__main__":
  # key assumption: DATETIME is in a string as in MAST.CELL records
  args = parse_args()
  
  simulation_results_dir = args.simulation_results_dir
  base_results_dir = args.base_results_dir
  agent_list_file = args.agent_list_file
  time_var = args.time_var
  cell_id = args.cell_id
  experiment_params = args.experiment_params
  datetime_mode = args.datetime_mode
  marker_plot_mode=True
  
  agent_list = list_from_textfile(agent_list_file)

  plot_all_agent_reward(experiment_params, cell_id, agent_list, time_var, simulation_results_dir, base_results_dir)
