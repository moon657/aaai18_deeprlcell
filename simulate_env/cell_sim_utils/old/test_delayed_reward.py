import numpy as np
from os import path
import pandas
import sys,os

RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

from simple_cell_utils import *

####################################
# base_results_dir = '/Users/csandeep/Documents/work/uhana/code/deepRL_cell/deeprl_cell/example_data/example_reward_dataframe/test'
base_results_dir = '/home/csandeep/work/experiments_deeprl_cell/delayed_reward_test'

total_reward_history_csv = RL_ROOT_DIR + '/example_data/example_reward_dataframe/REWARD_HISTORY.RL_time_variant_agent_DDPG_hard_thpt_limit_0.1_history_5_activity_mult_5_premature_abort_False_alpha_1_beta_1_kappa_10_dynamics_add_to_next_state.csv'

full_reward_history_df = pandas.read_csv(total_reward_history_csv)

DELAY_REWARD_INTERVAL = 15

EVAL_INTERVAL = 1

delayed_reward_col = 'DELAYED_REWARD'

iteration_index_var = 'ITERATION_INDEX'
batch_num_var = 'BATCH_NUM'

delayed_reward_col = 'DELAYED_REWARD'

full_reward_history_df[delayed_reward_col] = 0.0

final_reward_history = base_results_dir + '/rewards.csv'

total_reward_history_df = full_reward_history_df[full_reward_history_df[batch_num_var] <= 20]

MAX_BATCHES = 50

# update the total reward history
for index, row in total_reward_history_df.iterrows():

    batch_num = row[batch_num_var]

    if(batch_num <= MAX_BATCHES):

    # print('batch num var ', batch_num)

        if( (index % EVAL_INTERVAL) == 0):
            point_reward, total_reward_history_df = get_simple_delayed_reward(iteration_index = index, DELAY_REWARD_INTERVAL = DELAY_REWARD_INTERVAL, total_reward_history_df = total_reward_history_df, iteration_index_var = 'ITERATION_INDEX', delayed_reward_col = delayed_reward_col, batch_num_var = 'BATCH_NUM', specific_batch_index = None, reward_var = 'REWARD', print_mode = False)
            print('point reward', point_reward)
            print('batch_num', batch_num, total_reward_history_df[batch_num_var].max())

    else:
        break
total_reward_history_df.to_csv(final_reward_history)

# plot the delayed reward per batch vs actual reward


