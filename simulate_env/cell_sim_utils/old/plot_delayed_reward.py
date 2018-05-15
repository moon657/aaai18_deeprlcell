import numpy as np
from os import path
import pandas
import sys,os

RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

util_dir = RL_ROOT_DIR + '/utils/'
sys.path.append(util_dir)

from double_panel_plot_timeseries import no_datetime_overlay_KPI_plot

####################################
# base_results_dir = '/Users/csandeep/Documents/work/uhana/code/deepRL_cell/deeprl_cell/example_data/example_reward_dataframe/test'
base_results_dir = '/home/csandeep/work/experiments_deeprl_cell/delayed_reward_test'

delayed_reward_col = 'DELAYED_REWARD'
iteration_index_var = 'ITERATION_INDEX'
batch_num_var = 'BATCH_NUM'

final_reward_history = base_results_dir + '/rewards.csv'
total_reward_history_df = pandas.read_csv(final_reward_history)

batch_numbers = list(set(total_reward_history_df[batch_num_var]))

NUM_PLOTS = 10

for batch_num in batch_numbers[0:NUM_PLOTS]:

    # get subselect batch 
    subselect_df = total_reward_history_df[total_reward_history_df[batch_num_var] == batch_num]

    print('batch_num', batch_num)
    print('reward sum', subselect_df['REWARD'].sum())
    #print('reward description', subselect_df['REWARD'].describe())
    
    print('delayed reward sum', subselect_df['DELAYED_REWARD'].sum())
    #print('delayed reward description', subselect_df['DELAYED_REWARD'].describe())
    print(' ')

    # now plot the delayed vs actual reward per batch on a timeseries overlay

    # start the plots
    #####################

    # info for plotting
    #####################
    experiment_params = 'test_delayed_reward'
    KPI_to_plot_file = base_results_dir + '/KPI.txt'

    plotting_info_dict = {}
    plotting_info_dict['KPI_to_plot_file'] = KPI_to_plot_file
    plotting_info_dict['cell_id'] = '136046093'
    plotting_info_dict['time_var'] = 'ITERATION_INDEX'
    plotting_info_dict['base_results_dir'] = base_results_dir
    plotting_info_dict['experiment_params'] = experiment_params

    # plot first batch KPIs
    plotting_info_dict['experiment_params'] = experiment_params + '_early'
    no_datetime_overlay_KPI_plot(subselect_df, plotting_info_dict)


## update the total reward history
#for index, row in total_reward_history_df.iterrows():
#
#    batch_num = row[batch_num_var]
#
#    if(batch_num <= MAX_BATCHES):
#
#    # print('batch num var ', batch_num)
#
#        if( (index % EVAL_INTERVAL) == 0):
#            point_reward, total_reward_history_df = get_simple_delayed_reward(iteration_index = index, DELAY_REWARD_INTERVAL = DELAY_REWARD_INTERVAL, total_reward_history_df = total_reward_history_df, iteration_index_var = 'ITERATION_INDEX', delayed_reward_col = delayed_reward_col, batch_num_var = 'BATCH_NUM', specific_batch_index = None, reward_var = 'REWARD', print_mode = False)
#            print('point reward', point_reward)
#            print('batch_num', batch_num, total_reward_history_df[batch_num_var].max())
#
#    else:
#        break
#total_reward_history_df.to_csv(final_reward_history)
#
## plot the delayed reward per batch vs actual reward


