import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import cPickle
import json
import pandas
import sys,os
import random
import ConfigParser

RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + '/utils/'
sys.path.append(util_dir)

# for DDPG algorithm
actor_critic_utils = RL_ROOT_DIR + '/simulate_env/agents/DDPG/'
sys.path.append(actor_critic_utils)

# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

# simulators
simulators_dir = RL_ROOT_DIR + '/simulate_env/simulators/'
sys.path.append(simulators_dir)

# to plot
plot_dir = RL_ROOT_DIR + '/simulate_env/post_process_experiments/'
sys.path.append(plot_dir)

# random forest utils
RF_utils_dir = RL_ROOT_DIR + '/random_forest/'
sys.path.append(RF_utils_dir)

print(sys.path)
print(RL_ROOT_DIR)

from helper_utils_cell_simulator import conf_load_reward_params_dict, load_burst_prob_params_dict, conf_get_single_experiment_setting, resolve_MDDPG_config_paths
from conf_random_forest_simulator import ConfRandomForestTimeVariantSimulatorEnv
from algorithm import * 
import train_util as tu
from textfile_utils import remove_and_create_dir
from simple_cell_utils import parse_head_info
from LOOP_boxplot_experiments import plot_single_batch
from score_random_forest import load_saved_rf
"""
Run any algo on any environment. Save learning curve and rewards history
"""
def conf_simple_evaluate_agent(learning_algo_dict=None, experiment_config = None, file_path_config = None, experiment_settings = None):
    # select the environment
    env_type = experiment_config.get('RL_AGENT_INFO', 'env_type')
    print('env', env_type)
    # run the correct simulator
    train_inds, test_inds, head_filter = tu.read_train_info(experiment_config.get('TRAINING_EPISODE_INFO', 'train_info'))
    experiment_settings['train_inds'] = train_inds
    experiment_settings['test_inds'] = test_inds
    train_info_csv = experiment_config.get('TRAINING_EPISODE_INFO', 'train_info')
    heads_list, rf_models, hard_thpt_limit, num_heads, _, _, _ = parse_head_info(train_info_csv = train_info_csv)
    experiment_settings['hard_thpt_limit'] = hard_thpt_limit
    experiment_settings['rf_models'] = rf_models

    # check if this is for evaluation
    test_flag = experiment_config.getint('RL_AGENT_INFO', 'test_experiment_num')
    if test_flag > 0:
        experiment_settings['train_inds'] = []
        experiment_settings['test_inds'] = test_inds+train_inds
    env = ConfRandomForestTimeVariantSimulatorEnv(experiment_config = experiment_config, file_path_config = file_path_config, experiment_settings = experiment_settings)
    
    NUM_HEAD=experiment_config.getint('RL_AGENT_INFO', 'NUM_HEAD')

    # select the learning agent
    algo_type = experiment_config.get('RL_AGENT_INFO', 'algo')
    print('algo', algo_type)
    if algo_type == 'DDPG':
        algo = DDPG(env.observation_space, env.action_space, WARMUP=learning_algo_dict['WARMUP'], OU_NOISE=learning_algo_dict['OU_NOISE'], clip_action_explore = learning_algo_dict['clip_action_explore'])
        if test_flag > 0:
            file_path = learning_algo_dict['save_neural_net_path'].split('/')
            file_path[-2] = str(test_flag)
            algo.load_model(file_path='/'.join(file_path), env_name='cell')
    elif algo_type == 'AC':
        algo = ActorCritic(env.observation_space, env.action_space, EPSILON=learning_algo_dict['EPSILON'], scale=True)
    elif algo_type == 'MPDDPG':
        algo = MultiHeadPPCDDPG(10, env.observation_space, env.action_space, WARMUP=learning_algo_dict['WARMUP'], NUM_HEAD=experiment_config.getint('RL_AGENT_INFO', 'NUM_HEAD'), OU_NOISE=learning_algo_dict['OU_NOISE'], clip_action_explore=learning_algo_dict['clip_action_explore'])
    elif algo_type == 'MDDPG':
        algo = MultiHeadDDPG(env.observation_space, env.action_space, WARMUP=learning_algo_dict['WARMUP'], NUM_HEAD=experiment_config.getint('RL_AGENT_INFO', 'NUM_HEAD'), OU_NOISE=learning_algo_dict['OU_NOISE'], clip_action_explore=learning_algo_dict['clip_action_explore'])
    elif algo_type == 'H0':
        max_day_values, CELX_dict = get_all_day_celx(experiment_config = experiment_config)
        CELX_df = CELX_dict['ALL']
        # CELX_df.dropna()
        thpt_max = CELX_df['CELLT_AVG_THP_DL'].max()
        thpt_limit = hard_thpt_limit[0]
        rf_path = rf_models[0].split('/')
        rf_path[-3] = rf_path[-3] + '_baseline'
        rf_path = '/'.join(rf_path)
        algo = Heuristic(thpt_limit, load_saved_rf(rf_path), thpt_max)
    else:
        print(algo_type, 'not supported')

    base_results_dir = experiment_settings['experiment_plot_path']
    # train agent on env
    if learning_algo_dict['train_mode'] == 'simple':
        EPISODES = experiment_config.getint('TRAINING_EPISODE_INFO', 'TOTAL_EPISODES')
        TEST_EPISODES = experiment_config.getint('TRAINING_EPISODE_INFO', 'TEST_EPISODES')
        learning_algo_rewards, learning_algo_wt = tu.train(algo, env, MAX_STEP=learning_algo_dict['MAX_STEP'], EPISODES=EPISODES, TEST_EPISODES=TEST_EPISODES, TEST_TERM=learning_algo_dict['TEST_TERM'], SAVE_TERM=learning_algo_dict['SAVE_TERM'], verbose_wt=learning_algo_dict['verbose_wt'], exp_mode=learning_algo_dict['exp_mode'], env_name='cell', save_path=learning_algo_dict['save_neural_net_path'], plot_path = base_results_dir)
    elif learning_algo_dict['train_mode'] == 'ppc':
        TRAIN_DAY_REPEAT = experiment_config.getint('TRAINING_EPISODE_INFO', 'TRAIN_DAY_REPEAT')
        TEST_DAY_REPEAT = experiment_config.getint('TRAINING_EPISODE_INFO', 'TEST_DAY_REPEAT')
        learning_algo_rewards, learning_algo_wt = tu.train_ppc(algo, env, TRAIN_DAY_REPEAT=TRAIN_DAY_REPEAT, TEST_DAY_REPEAT=TEST_DAY_REPEAT, MAX_STEP=learning_algo_dict['MAX_STEP'], TEST_TERM=learning_algo_dict['TEST_TERM'], SAVE_TERM=learning_algo_dict['SAVE_TERM'], verbose_wt=learning_algo_dict['verbose_wt'], env_name='cell', save_path=learning_algo_dict['save_neural_net_path'], plot_path = base_results_dir, head_filter=head_filter)
    else:
        print(learning_algo_dict['train_mode'], 'not supported')
    experiment_params = env_type
    time_var = experiment_config.get('PLOTTING_INFO', 'time_var')

    # plot learning curve
    #rmeans = learning_algo_rewards[:1][0]
    #rstds = learning_algo_rewards[1:][0]
    #train_reward_plot = base_results_dir + '/' + experiment_params + '.batchRewards.png'
    #tu.plot_train_rewards(rmeans, rstds, fig_path=train_reward_plot, xlabel='Training episodes', ylabel='Total rewards', x=None)
   
    # save learning curve data to dataframe
    # learning_algo_rewards_df = pandas.DataFrame({'mean_reward': list(learning_algo_rewards[0]), 'std_reward': list(learning_algo_rewards[1])})
    # experiment_reward_results_csv = base_results_dir + '/' + experiment_params + '.batchRewards.csv'
    # learning_algo_rewards_df.to_csv(experiment_reward_results_csv, index=False)
    
    # write the reward history dataframe for later analysis
    #####################
    reward_history_df = env.reward_history_df
    reward_history_csv = base_results_dir + '/REWARD_HISTORY.' + experiment_params + '.csv'
    reward_history_df.to_csv(reward_history_csv, index=False)

    # get data for unconverged/converged neural net performance
    #####################
    batch_var = 'BATCH_NUM'
    if learning_algo_dict['train_mode'] == 'simple':
        early_batch_index = reward_history_df[batch_var].min()
        late_batch_index = reward_history_df[batch_var].max()
        
        early_batch_df = reward_history_df[reward_history_df[batch_var] == early_batch_index]
        late_batch_df = reward_history_df[reward_history_df[batch_var] == late_batch_index]

        # list of KPIs to overlay and plot
        two_KPI_file = experiment_config.get('PLOTTING_INFO', 'KPI_to_plot_file')

        # plot the converged, unconverged timeseries FOR THE TEST DATA ONLY
        plot_suffix = 'UNCONVERGED_EXP_' + str(experiment_num) 
        plot_single_batch(head_ind=ind, timeseries_df = early_batch_df, two_KPI_file = two_KPI_file, experiment_settings = experiment_settings, datetime_mode = True, time_variable = 'DATETIME', plot_suffix = plot_suffix)

        plot_suffix = 'CONVERGED_EXP_' + str(experiment_num) 
        plot_single_batch(head_ind=ind, timeseries_df = late_batch_df, two_KPI_file = two_KPI_file, experiment_settings = experiment_settings, datetime_mode = True, time_variable = 'DATETIME', plot_suffix = plot_suffix)
    elif learning_algo_dict['train_mode'] == 'ppc':
        cur_index = reward_history_df[batch_var].max()
        two_KPI_file = experiment_config.get('PLOTTING_INFO', 'KPI_to_plot_file')
        for i in xrange(len(test_inds)):
            test_ind = test_inds[-i-1]
            batch_df = reward_history_df[reward_history_df[batch_var] == cur_index]
            plot_suffix = test_ind[0] + ',' + test_ind[1]
            plot_single_batch(head_ind=0, timeseries_df = batch_df, two_KPI_file = two_KPI_file, experiment_settings = experiment_settings, datetime_mode = True, time_variable = 'DATETIME', plot_suffix = plot_suffix)
            # plot_single_batch(head_ind=len(test_inds)-1-i, timeseries_df = batch_df, two_KPI_file = two_KPI_file, experiment_settings = experiment_settings, datetime_mode = True, time_variable = 'DATETIME', plot_suffix = plot_suffix)
            cur_index -= TEST_DAY_REPEAT
        late_batch_df = batch_df

    return reward_history_df, learning_algo_rewards, learning_algo_wt, algo, env, late_batch_df

"""
Run and compare algos on a certain environment, save and plot rewards.
"""
def conf_compare_agents(learning_algo_dict=None, experiment_config = None, file_path_config = None, experiment_settings = None):
    pass

"""
write PARAMS.txt that give settings for a single experiment and whether it worked
"""
def write_experiment_params_success(experiment_results_indicator_fname = None, success_str = None, experiment_settings = None, late_batch_df = None):
    # log all params like alpha, beta, kappa to file
    with open(experiment_results_indicator_fname, 'w') as f:
        success_string = '\t'.join(['SUCCESS',success_str])
        f.write(success_string + '\n')
        for k,v in experiment_settings.iteritems():
            if(k != 'alpha_beta_kappa_dict'):
                out_string = '\t'.join([k,str(v)])
                f.write(out_string + '\n')

        # PRINT THE ACTION STD
        if(success_str != 'FALSE'):
            out_str = 'ACTION'
            f.write(out_str + '\n')
            out_str = late_batch_df['ACTION'].describe().to_string()
            f.write(out_str + '\n')

""" 
    load each timeseries per day and an overall timeseries for the cell
    output a list of points per day to determine episode length for RL
"""

def get_all_day_celx(experiment_config = None):

    # loop over all days to see how long an episode may ever last
    test_days = experiment_config.get('TRAIN_TEST_SPLIT', 'test_days').split(',')

    train_days = experiment_config.get('TRAIN_TEST_SPLIT', 'train_days').split(',')
    
    timeseries_dir = experiment_config.get('TIMESERIES_INPUT_DATA', 'timeseries_dir')

    cell = experiment_config.get('TIMESERIES_INPUT_DATA', 'cell_id')
 
    all_days = train_days + test_days

    CELX_dict = {}
    max_day_values = []
    for day in all_days:
        try:
            CELX_file = '.'.join(['MAST.CELX', cell, day, 'csv'])
            CELX_df = pandas.read_csv(timeseries_dir + '/' + CELX_file)  
            CELX_df = CELX_df.dropna()
            print len(CELX_df)
            CELX_dict[day] =  CELX_df
            # number of rows = length of day
            max_day_values.append(CELX_df.shape[0])

        except:
            pass

    # get all days 
    CELX_file = '.'.join(['MAST.CELX', cell, 'csv'])
    CELX_df = pandas.read_csv(timeseries_dir + '/' + CELX_file)  
    CELX_df = CELX_df.dropna()
    CELX_dict['ALL'] =  CELX_df

    return max_day_values, CELX_dict

""" save parameters on how many episodes, how often to test in a dictionary
for the learning algorithm
"""

def load_learning_algo_dict(experiment_config = None, NN_path = None, experiment_settings = None):

    # list of day lengths by day
    max_day_values, CELX_dict = get_all_day_celx(experiment_config = experiment_config)

    EPISODE_LENGTH = max(max_day_values)

    learning_algo_dict = {}
    learning_algo_dict['MAX_STEP'] = EPISODE_LENGTH
    learning_algo_dict['EPISODE_LENGTH'] = EPISODE_LENGTH
    learning_algo_dict['TEST_TERM'] = experiment_config.getint('TRAINING_EPISODE_INFO', 'TEST_TERM')
    learning_algo_dict['SAVE_TERM'] = experiment_config.getint('TRAINING_EPISODE_INFO', 'SAVE_TERM')
    learning_algo_dict['EXP_EPISODES'] = experiment_config.getint('TRAINING_EPISODE_INFO', 'EXP_EPISODES')
    learning_algo_dict['save_neural_net_path'] = NN_path
    learning_algo_dict['verbose_wt'] = experiment_config.getboolean('TRAINING_EPISODE_INFO', 'verbose_wt')
    learning_algo_dict['exp_mode'] = experiment_config.get('TRAINING_EPISODE_INFO', 'exp_mode')
    learning_algo_dict['train_mode'] = experiment_config.get('TRAINING_EPISODE_INFO', 'train_mode')
    if learning_algo_dict['EXP_EPISODES'] < 0:
        WARMUP = int(.1*learning_algo_dict['EPISODES']*learning_algo_dict['EPISODE_LENGTH'])
    else:
        WARMUP = int(learning_algo_dict['EXP_EPISODES']*learning_algo_dict['EPISODE_LENGTH'])
    learning_algo_dict['WARMUP'] = WARMUP

    # copy to experiment settings so these params get written to text
    for k,v in learning_algo_dict.iteritems():
        experiment_settings[k] = learning_algo_dict[k]

    return learning_algo_dict, experiment_settings

"""
get min, max state vector 
"""
def conf_get_state_bounds(history_minutes = None, experiment_config = None):

    max_day_values, CELX_dict = get_all_day_celx(experiment_config = experiment_config)

    CELX_df = CELX_dict['ALL']
    # CELX_df.dropna()
    print('GETTING BOUNDS FOR ALL DATA')

    KPI_list = experiment_config.get('STATE_SPACE', 'state_features').split(',')

    # get min and max states from data for state space bounds
    ###############################################
    min_state_list = []
    max_state_list = []
    # print CELX_df[KPI_list].describe()
    for lag in range(history_minutes):
        for KPI in KPI_list:
            min_KPI = CELX_df[KPI].quantile(0.15)
            max_KPI = CELX_df[KPI].quantile(0.85)
            print('KPI', KPI)
            print('min_KPI', min_KPI)
            print('max_KPI', max_KPI)
            min_state_list.append(min_KPI)
            max_state_list.append(max_KPI)

    min_state_vector = np.array(min_state_list)
    max_state_vector = np.array(max_state_list)
    print('min_state_vector ', min_state_vector)
    print('max_state_vector ', max_state_vector)

    return min_state_vector, max_state_vector


"""
main wrapper to run a single experiment for a single reward setting, called in PARALLEL on a cluster
"""
def conf_wrapper_run_experiment(experiment_settings = None, experiment_config = None, file_path_config = None):
    ###############################################
    # load the burst probs action space
    burst_discretization = experiment_config.getfloat('ACTION_SPACE', 'burst_discretization')
    min_burst = experiment_config.getfloat('ACTION_SPACE', 'min_burst')
    max_burst = experiment_config.getfloat('ACTION_SPACE', 'max_burst')

    # action space
    burst_prob_params_dict = load_burst_prob_params_dict(
        burst_discretization=burst_discretization,
        max_burst_prob=max_burst,
        min_burst_prob=min_burst)

    # agent will be passed action space
    experiment_settings['burst_prob_params_dict'] = burst_prob_params_dict
    experiment_num = experiment_settings['experiment_num']

    # print('EXPERIMENT SETTINGS')
    # print(experiment_settings)

    print('BURST PROBS')
    print(burst_prob_params_dict)

    ##############################################
    min_state_vector, max_state_vector = conf_get_state_bounds(history_minutes = experiment_settings['history_minutes'], experiment_config = experiment_config)
    # min, max state vector
    experiment_settings['min_state_vector'] = min_state_vector
    experiment_settings['max_state_vector'] = max_state_vector

    # for the reward computation, differs by experiments
    reward_params_dict = conf_load_reward_params_dict(
        alpha= experiment_settings['alpha'],
        beta= experiment_settings['beta'],
        kappa = experiment_settings['kappa'],
        hard_thpt_limit = experiment_settings['hard_thpt_limit']
        )
    
    print('reward_params_dict')
    print(reward_params_dict)
    experiment_settings['reward_params_dict'] = reward_params_dict

    base_results_dir = file_path_config.get('OUTPUT_DIRECTORIES', 'base_results_dir')
    # create a dir for all the experiment metadata and plots
    experiment_params = str(experiment_num)
    print('RUNNING EXPERIMENT ', experiment_params)
    experiment_path = base_results_dir + '/' + experiment_params
    print('experiment path', experiment_path)
    remove_and_create_dir(experiment_path)
   
    # save NN models
    NN_path = experiment_path + '/saved_models'
    remove_and_create_dir(NN_path)
    # list of parameters for experiments
    experiment_results_indicator_fname = experiment_path + '/PARAMS.txt'
    experiment_settings['experiment_plot_path'] = experiment_path

    # do we keep going if a single experiment fails?
    try_catch_mode = experiment_config.getboolean('EXPERIMENT_INFO', 'try_catch_mode')

    # data on how to train agent for, where to save models
    learning_algo_dict, experiment_settings = load_learning_algo_dict(experiment_config = experiment_config, NN_path = NN_path, experiment_settings = experiment_settings)
    clip_action_explore = experiment_settings['clip_action_explore']
    learning_algo_dict['clip_action_explore'] = clip_action_explore
    algo_type = experiment_config.get('RL_AGENT_INFO', 'algo')
    if algo_type in ['DDPG','MPDDPG','MDDPG']:
        OU_NOISE_STR = experiment_settings['OU_NOISE']
        learning_algo_dict['OU_NOISE'] = [float(x) for x in OU_NOISE_STR.split('-')]
    elif algo_type == 'AC':
        epsilon = experiment_config.get('RL_AGENT_INFO', 'EPSILON')
        learning_algo_dict['EPSILON'] = [float(x) for x in epsilon.split(',')]

    print('learning algo dict')
    print(learning_algo_dict)

    # we keep going if a single experiment fails
    if try_catch_mode:

        try:
            # run the learning agent, record results, write plots
            reward_history_df, learning_algo_rewards_df, learning_algo_rewards, learning_algo_wt, algo, env, late_batch_df = conf_simple_evaluate_agent(learning_algo_dict = learning_algo_dict, experiment_config = experiment_config, file_path_config = file_path_config, experiment_settings = experiment_settings)
            success_str = 'TRUE'
        except:
            success_str = 'FALSE'
            late_batch_df = None

        print('TRY_CATCH', try_catch_mode)
        print('SUCCESS', success_str)
        # write the success to a file
        write_experiment_params_success(experiment_results_indicator_fname = experiment_results_indicator_fname, success_str = success_str, experiment_settings = experiment_settings, late_batch_df = late_batch_df)

    # if we want system to stop if even a single experiment fails
    else:
        # run the learning agent, record results, write plots
        reward_history_df, learning_algo_rewards, learning_algo_wt, algo, env, late_batch_df = conf_simple_evaluate_agent(learning_algo_dict = learning_algo_dict, experiment_config = experiment_config, file_path_config = file_path_config, experiment_settings = experiment_settings)
        
        success_str = 'TRUE'
        # write the success to a file
        write_experiment_params_success(experiment_results_indicator_fname = experiment_results_indicator_fname, success_str = success_str, experiment_settings = experiment_settings, late_batch_df = late_batch_df)


"""
have a directory of experiments 0,1,2,3 each having base_MDDPG_params.ini already set
we create joint.ini per experiment in this function and read the experiment settings from a file
"""

def conf_preset_experiment_wrapper(file_path_config = None, experiment_num = None):

    # get the joint config file
    experiment_dir = file_path_config.get('OUTPUT_DIRECTORIES', 'base_results_dir') + '/' + str(experiment_num)
    experiment_config_file = experiment_dir + '/base_MDDPG_params.ini'
    joint_config_file = experiment_dir + '/joint.ini'

    train_info_csv = experiment_dir + '/train_info.csv'

    assert(os.path.exists(experiment_dir))
    print('experiment dir exists')

    # get the paths
    experiment_config = resolve_MDDPG_config_paths(experiment_config_file = experiment_config_file, joint_config_file = joint_config_file, train_info_csv = train_info_csv)

    # get SINGLE EXPERIMENT INFO from config file
    experiment_settings = conf_get_single_experiment_setting(experiment_config = experiment_config)

    ###############################################
    # load the burst probs action space
    burst_discretization = experiment_config.getfloat('ACTION_SPACE', 'burst_discretization')
    min_burst = experiment_config.getfloat('ACTION_SPACE', 'min_burst')
    max_burst = experiment_config.getfloat('ACTION_SPACE', 'max_burst')

    # action space
    burst_prob_params_dict = load_burst_prob_params_dict(
        burst_discretization=burst_discretization,
        max_burst_prob=max_burst,
        min_burst_prob=min_burst)

    # agent will be passed action space
    experiment_settings['burst_prob_params_dict'] = burst_prob_params_dict

    # print('EXPERIMENT SETTINGS')
    # print(experiment_settings)

    print('BURST PROBS')
    print(burst_prob_params_dict)

    ##############################################
    min_state_vector, max_state_vector = conf_get_state_bounds(history_minutes = experiment_settings['history_minutes'], experiment_config = experiment_config)
    # min, max state vector
    experiment_settings['min_state_vector'] = min_state_vector
    experiment_settings['max_state_vector'] = max_state_vector

    # for the reward computation, differs by experiments
    reward_params_dict = conf_load_reward_params_dict(
        alpha= experiment_settings['alpha'],
        beta= experiment_settings['beta'],
        kappa = experiment_settings['kappa'],
        hard_thpt_limit = experiment_settings['hard_thpt_limit']
        )
    
    print('reward_params_dict')
    print(reward_params_dict)
    experiment_settings['reward_params_dict'] = reward_params_dict

    base_results_dir = file_path_config.get('OUTPUT_DIRECTORIES', 'base_results_dir')
    # create a dir for all the experiment metadata and plots
    experiment_params = str(experiment_num)
    print('RUNNING EXPERIMENT ', experiment_params)
    experiment_path = base_results_dir + '/' + experiment_params + '/results'
    print('experiment path', experiment_path)
    remove_and_create_dir(experiment_path)

    # save NN models
    NN_path = experiment_path + '/saved_models'
    remove_and_create_dir(NN_path)
    # list of parameters for experiments
    experiment_results_indicator_fname = experiment_path + '/PARAMS.txt'
    experiment_settings['experiment_plot_path'] = experiment_path

    # do we keep going if a single experiment fails?
    try_catch_mode = experiment_config.getboolean('EXPERIMENT_INFO', 'try_catch_mode')

    # data on how to train agent for, where to save models
    learning_algo_dict, experiment_settings = load_learning_algo_dict(experiment_config = experiment_config, NN_path = NN_path, experiment_settings = experiment_settings)
    clip_action_explore = experiment_settings['clip_action_explore']
    learning_algo_dict['clip_action_explore'] = clip_action_explore
    algo_type = experiment_config.get('RL_AGENT_INFO', 'algo')
    if algo_type in ['DDPG','MPDDPG','MDDPG']:
        OU_NOISE_STR = experiment_settings['OU_NOISE']
        learning_algo_dict['OU_NOISE'] = [float(x) for x in OU_NOISE_STR.split('-')]
    elif algo_type == 'AC':
        epsilon = experiment_config.get('RL_AGENT_INFO', 'EPSILON')
        learning_algo_dict['EPSILON'] = [float(x) for x in epsilon.split(',')]

    print('learning algo dict')
    print(learning_algo_dict)

    # we keep going if a single experiment fails
    if try_catch_mode:

        try:
            # run the learning agent, record results, write plots
            reward_history_df, learning_algo_rewards_df, learning_algo_rewards, learning_algo_wt, algo, env, late_batch_df = conf_simple_evaluate_agent(learning_algo_dict = learning_algo_dict, experiment_config = experiment_config, file_path_config = file_path_config, experiment_settings = experiment_settings)
            success_str = 'TRUE'
        except:
            success_str = 'FALSE'
            late_batch_df = None

        print('TRY_CATCH', try_catch_mode)
        print('SUCCESS', success_str)
        # write the success to a file
        write_experiment_params_success(experiment_results_indicator_fname = experiment_results_indicator_fname, success_str = success_str, experiment_settings = experiment_settings, late_batch_df = late_batch_df)

    # if we want system to stop if even a single experiment fails
    else:
        # run the learning agent, record results, write plots
        reward_history_df, learning_algo_rewards, learning_algo_wt, algo, env, late_batch_df = conf_simple_evaluate_agent(learning_algo_dict = learning_algo_dict, experiment_config = experiment_config, file_path_config = file_path_config, experiment_settings = experiment_settings)
        
        success_str = 'TRUE'
        # write the success to a file
        write_experiment_params_success(experiment_results_indicator_fname = experiment_results_indicator_fname, success_str = success_str, experiment_settings = experiment_settings, late_batch_df = late_batch_df)

