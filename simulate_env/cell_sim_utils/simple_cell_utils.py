import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import cPickle
import json
import pandas
import sys, os

RL_ROOT_DIR = os.environ['RL_ROOT_DIR']
# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

"""
Generate reward = alpha*Bytes_PPC - beta*Bytes_lost_user - kappa*[hard_thpt_limit - B'] * A * dt
"""
def parse_head_info(train_info_csv = None):

    train_info_df = pandas.read_csv(train_info_csv)
        
    heads_list = list(set(train_info_df['HEAD_ID']))

    heads_list.sort()
    print('heads', heads_list)

    # generate paths to all rfs
    rf_list = []
    hard_thpt_limit_list = []
    cell_ids = []

    
    for head in heads_list:
        head_df = train_info_df[train_info_df['HEAD_ID'] == head].iloc[0]
        
        print('head df')
        print(head_df)

        print(head_df['RF_PATH'])

        rf_path = RL_ROOT_DIR + '/' + str(head_df['RF_PATH']) + '/random_forest_model.' + str(head_df['RF_INFO']) + '.y_var.CELLT_AGG_THP_DL.pkl'
        print(rf_path)

        rf_list.append(rf_path)

        cell = str(head_df['CELL_ID'])
        cell_ids.append(cell)

        # get a list of thpt limits
        hard_thpt_limit_list.append(head_df['HARD_THPT_LIMIT'])

    train_days = train_info_df[train_info_df['TRAIN_TEST_INDICATOR'] == 'TRAIN']['DATE_LOCAL']
    print(train_days)
    
    test_days = train_info_df[train_info_df['TRAIN_TEST_INDICATOR'] == 'TEST']['DATE_LOCAL']
    print(test_days)

    num_heads = len(heads_list)
    return heads_list, rf_list, hard_thpt_limit_list, num_heads, train_days, test_days, cell_ids

def get_simplified_reward(action=None,
                          reward_params_dict=None,
                          print_mode=None,
                          burst_prob_user_selector='same_as_PPC',
                          orig_thpt=None,
                          hard_thpt_limit_flag=True,
                          new_cell_thpt=None,
                          KB_MB_converter=1):

    if (print_mode):
        print('FUNCTION get_simplified_reward')
        print('action', action)

    # map B to B'
    # new_cell_thpt = orig_thpt*(1-action)

    # B' = new_thpt: kb/sec, A = action = burst_prob: [0,1] unitless, T	 = control_interval = 1 min = 60 seconds
    # data_MB = B' * A * T
    burst_prob = action[0]
    control_interval_seconds = reward_params_dict['control_interval_seconds']
    PPC_scheduled_data_MB = float(
        new_cell_thpt * burst_prob *
        control_interval_seconds) / float(KB_MB_converter)

    # modulate various terms in reward computation
    alpha = reward_params_dict['alpha']
    beta = reward_params_dict['beta']
    kappa = reward_params_dict['kappa']

    # do PPC user and regular user have same burst probs?
    if (burst_prob_user_selector == 'same_as_PPC'):
        burst_prob_user = burst_prob
    else:
        burst_prob_user = reward_params_dict['avg_user_burst_prob']

    user_original_data_MB = (orig_thpt * burst_prob_user *
                             control_interval_seconds) / float(KB_MB_converter)
    user_new_data_MB = (new_cell_thpt * burst_prob_user *
                        control_interval_seconds) / float(KB_MB_converter)
    user_lost_data_MB = user_original_data_MB - user_new_data_MB

    # only in flag is set do we add a penalty for thpts that are below a hard limit, eg (K-B')
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

    #reward_scale = 1
    # REWARD COMPUTATION
    reward = alpha * PPC_scheduled_data_MB - beta * user_lost_data_MB - kappa * hard_thpt_limit_MB

    #reward = reward / reward_scale
    if (print_mode):
        print('cell_thpt', orig_thpt)
        print('new_cell_thpt', new_cell_thpt)
        print('burst_prob_PPC', burst_prob)
        print('burst_prob_user', burst_prob_user)
        print('PPC_scheduled_data_MB', PPC_scheduled_data_MB)
        print('data_MB lost user', user_lost_data_MB)
        print('thpt penalty', hard_thpt_limit_MB)
        print('reward', reward)

    return reward, action, PPC_scheduled_data_MB, user_lost_data_MB, hard_thpt_limit_MB


"""
Add instantaneous reward per timestamp to a dataframe
"""
# append reward, state info to reward_history_df for later plotting
def report_rewards(state=None,
                   burst=None,
                   reward=None,
                   reward_history_df=None,
                   iteration_index=None,
                   PPC_data_MB_scheduled=None,
                   user_lost_data_MB=None,
                   print_mode=None,
                   thpt=None,
                   new_thpt=None,
                   thpt_var=None,
                   batch_number=None,
                   hard_thpt_limit_MB=None,
                   experiment_params_dict=None,
                   datetime_str = None,
                   train_test_mode = None,
                   day = None,
                   ind = None):

    # dictionary of useful data 
    df_dict = {}
    df_dict['REWARD'] = reward
    df_dict['ITERATION_INDEX'] = iteration_index
    df_dict['TIME_INDEX'] = iteration_index
    df_dict['ACTION'] = burst
    df_dict['PPC_DATA_SCHEDULED_MB'] = PPC_data_MB_scheduled
    df_dict['USER_LOST_DATA_MB'] = user_lost_data_MB
    df_dict['HARD_THPT_LIMIT_MB'] = hard_thpt_limit_MB
    df_dict['STATE'] = state
    df_dict['BATCH_NUM'] = batch_number
    df_dict[thpt_var] = thpt
    df_dict['new_' + thpt_var] = new_thpt
    df_dict['hard_thpt_limit_flag'] = experiment_params_dict[
        'hard_thpt_limit_flag']
    df_dict['activity_factor_multiplier'] = experiment_params_dict[
        'activity_factor_multiplier']
    df_dict['hard_thpt_limit'] = experiment_params_dict['hard_thpt_limit'][ind]
    df_dict['history_minutes'] = experiment_params_dict['history_minutes']
    df_dict['alpha'] = experiment_params_dict['alpha']
    df_dict['beta'] = experiment_params_dict['beta']
    df_dict['kappa'] = experiment_params_dict['kappa']

    # some versions do not have these quantities
    try:
        df_dict['DATETIME'] = datetime_str
        df_dict['train_test_mode'] = train_test_mode
        df_dict['day'] = day
    except:
        print('failed to write datetime etc')
        print(datetime_str)
        print(train_test_mode)
        print(day)
        pass

    # convert to df
    local_df = pandas.DataFrame(df_dict)
    # append to existing dataframe of previous time results
    reward_history_df = reward_history_df.append(local_df)
    return reward_history_df

"""

Given rewards dataframe of reward per minute, add a new column to 
dataframe called 'DELAYED_REWARD' that normally has reward 0
BUT cumulative reward every DELAY_REWARD_INTERVAL

[CLIFF REWARD FUNCTION]

"""
def get_simple_delayed_reward(iteration_index=None,
                              DELAY_REWARD_INTERVAL=None,
                              total_reward_history_df=None,
                              iteration_index_var='ITERATION_INDEX',
                              delayed_reward_col=None,
                              batch_num_var='BATCH_NUM',
                              specific_batch_index=None,
                              reward_var='REWARD',
                              print_mode=False):

    reward_history_df = total_reward_history_df
    # get data for only one batch
    # reward_history_df = total_reward_history_df[total_reward_history_df[batch_num_var] == specific_batch_index]
    # only add delayed reward every DELAY_REWARD_INTERVAL 
    if ((iteration_index % DELAY_REWARD_INTERVAL == 0) & (iteration_index > 1)):

        # index ranges from 0 to num_rows
        reward_history_df.index = range(reward_history_df.shape[0])
        
        if print_mode:
            print('iteration_index', iteration_index)
            print('dim reward history df', reward_history_df.shape)
            print('original index')
            print('index reward history df', reward_history_df.index)
            print('new index')
            print('index reward history df', reward_history_df.index)

        # get data from [t - DELAY_REWARD_INTERVAL, t]
        min_bound = np.max([0, iteration_index - DELAY_REWARD_INTERVAL])
        max_bound = iteration_index - 1

        # get subset of data for the last DELAY_REWARD_INTERVAL
        # subselect_df = reward_history_df[(reward_history_df[
        #     iteration_index_var] >= min_bound) & (reward_history_df[
        #         iteration_index_var] <= max_bound)]

        subselect_df = reward_history_df[(reward_history_df.index >= min_bound) & (reward_history_df.index <= max_bound)]

        if print_mode:
            print('min_bound', min_bound)
            print('max_bound', max_bound)

        # get cumulative reward in last DELAY_REWARD_INTERVAL
        total_reward = sum(subselect_df[reward_var])



        # suppose data is from [t, t + DELAY_REWARD_INTERVAL]
        # put the DELAYED_REWARD at t+DELAY_REWARD_INTERVAL, 0 at test of points
        # index_of_delayed_reward = subselect_df[subselect_df[iteration_index_var] == max_bound].index[0]

        
        index_of_delayed_reward = subselect_df[subselect_df.index == max_bound].index[0]

        if print_mode:
            print('total_reward', total_reward)
            # location of the 
            print('index subselect_df', subselect_df.index)
            print('index_delayed_reward', index_of_delayed_reward)

        subselect_df[delayed_reward_col] = 0
        subselect_df.loc[index_of_delayed_reward,
                         delayed_reward_col] = total_reward

        reward_history_df.loc[min_bound:max_bound,
                              delayed_reward_col] = 0.0

        reward_history_df.loc[index_of_delayed_reward,
                              delayed_reward_col] = total_reward

        if print_mode:
            print('reward_history_df',
                  reward_history_df.loc[min_bound:max_bound][delayed_reward_col])
            print(subselect_df)
            print(reward_history_df)

        point_reward = total_reward
    else:
        # reward is 0 elsewhere in the epoch
        point_reward = 0.0
    
    if print_mode:
        print('iteration_index', iteration_index)
        print('point_reward', point_reward)
    return point_reward, reward_history_df
