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
from dateutil import parser
import ConfigParser

# load utils from the transition matrix
RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + '/utils'
sys.path.append(util_dir)
from textfile_utils import list_from_textfile
from simple_cell_utils import parse_head_info

# read a config file for MDDPG
# parse train_info.csv to get CELL_IDS, NUM_HEADS, train_days, test_days, hard_thpt_limits etc
def resolve_MDDPG_config_paths(experiment_config_file = None,  joint_config_file = None, train_info_csv = None):
    # parse train info_file
    heads_list, rf_list, hard_thpt_limit_list, num_heads, train_days, test_days, cell_ids = parse_head_info(train_info_csv = train_info_csv)

    single_cell_id = cell_ids[0]
    cell_ids_str = ','.join(cell_ids)
    train_days_str = ','.join(train_days)
    test_days_str = ','.join(test_days)
    thpt_limit_str = ','.join([str(x) for x in hard_thpt_limit_list])

    # RL relevant params
    experiment_config = ConfigParser.ConfigParser()
    experiment_config.read(experiment_config_file)    
    print experiment_config.sections()

    # now write the joint config file
    # lets create that config file for next time...
    cfgfile = open(joint_config_file,'w')
    Config = experiment_config

    random_forest_feature_list =  RL_ROOT_DIR + experiment_config.get('RANDOM_FOREST_PARAMS', 'random_forest_feature_list') 
    timeseries_dir =  RL_ROOT_DIR + experiment_config.get('TIMESERIES_INPUT_DATA', 'timeseries_dir') 
    KPI_to_plot_file = RL_ROOT_DIR + experiment_config.get('PLOTTING_INFO', 'KPI_to_plot_file') 

    Config.add_section('TRAIN_TEST_SPLIT')

    # add the settings to the structure of the file, and lets write it out...
    Config.set('RANDOM_FOREST_PARAMS','random_forest_feature_list', random_forest_feature_list)
    Config.set('TRAINING_EPISODE_INFO','train_info', train_info_csv)
    Config.set('TIMESERIES_INPUT_DATA','timeseries_dir', timeseries_dir)
    Config.set('TIMESERIES_INPUT_DATA','CELL_ID', single_cell_id)
    Config.set('TIMESERIES_INPUT_DATA','ALL_CELL_IDS', cell_ids_str)
    Config.set('PLOTTING_INFO','KPI_to_plot_file', KPI_to_plot_file)
    Config.set('RL_AGENT_INFO','NUM_HEAD', str(num_heads))
    Config.set('TRAIN_TEST_SPLIT','train_days', train_days_str)
    Config.set('TRAIN_TEST_SPLIT','test_days', test_days_str)
    Config.set('SINGLE_EXPERIMENT_INFO','hard_thpt_limit', thpt_limit_str)

    try:
        predefined_experiments_file =  RL_ROOT_DIR + experiment_config.get('EXPERIMENT_INFO', 'predefined_experiments_file') 
        Config.set('EXPERIMENT_INFO','predefined_experiments_file', predefined_experiments_file)
    except:
        pass

    Config.write(cfgfile)
    cfgfile.close()

    return Config

#######################################################
def conf_get_single_experiment_setting(experiment_config=None):
    experiment_settings = {}
    experiment_settings['alpha'] = experiment_config.getint('SINGLE_EXPERIMENT_INFO', 'alpha')
    experiment_settings['beta'] = experiment_config.getint('SINGLE_EXPERIMENT_INFO', 'beta')
    experiment_settings['kappa'] = experiment_config.getint('SINGLE_EXPERIMENT_INFO', 'kappa')
    experiment_settings['activity_factor_multiplier'] = experiment_config.getint('SINGLE_EXPERIMENT_INFO', 'activity_factor_multiplier')
    experiment_settings['hard_thpt_limit_flag'] = experiment_config.getboolean('SINGLE_EXPERIMENT_INFO', 'hard_thpt_limit_flag')
    experiment_settings['history_minutes'] = experiment_config.getint('SINGLE_EXPERIMENT_INFO', 'history_minutes')
    experiment_settings['OU_NOISE'] = experiment_config.get('SINGLE_EXPERIMENT_INFO', 'OU_NOISE')
    experiment_settings['delayed_reward_mode'] = experiment_config.getboolean('SINGLE_EXPERIMENT_INFO', 'delayed_reward_mode')
    experiment_settings['DELAY_REWARD_INTERVAL'] = experiment_config.getint('SINGLE_EXPERIMENT_INFO', 'DELAY_REWARD_INTERVAL')
    experiment_settings['KB_TO_MB'] = experiment_config.getfloat('RANDOM_FOREST_PARAMS', 'KB_TO_MB')

    alpha_beta_kappa_dict = {'alpha': experiment_settings['alpha'], 'beta': experiment_settings['beta'], 'kappa': experiment_settings['kappa']}
    experiment_settings['alpha_beta_kappa_dict'] = alpha_beta_kappa_dict
    experiment_settings['experiment_prefix'] = experiment_config.get('EXPERIMENT_INFO', 'experiment_prefix')
    experiment_settings['TEST_TERM'] = experiment_config.getint('TRAINING_EPISODE_INFO', 'TEST_TERM')
    experiment_settings['TEST_EPISODES'] = experiment_config.getint('TRAINING_EPISODE_INFO', 'TEST_EPISODES')
    experiment_settings['timeseries_dir'] = experiment_config.get('TIMESERIES_INPUT_DATA', 'timeseries_dir')
    experiment_settings['clip_action_explore'] = 'TRUE'
    experiment_settings['simulator_mode'] = experiment_config.get('RL_AGENT_INFO', 'env_type')

    # these are optional settings based on DDPG or MDDPG
    # ideally train/test days are not needed to be specified if in test mode

    try:
        experiment_settings['TOTAL_EPISODES'] = experiment_config.getint('TRAINING_EPISODE_INFO', 'TOTAL_EPISODES')
    except:
        pass

    try:
        experiment_settings['train_days'] = experiment_config.get('TRAIN_TEST_SPLIT', 'train_days')
    except:
        pass

    try:
        experiment_settings['test_days'] = experiment_config.get('TRAIN_TEST_SPLIT', 'test_days')
    except:
        pass

    try:
        experiment_settings['experiment_num'] = experiment_config.getint('SINGLE_EXPERIMENT_INFO', 'experiment_num')
    except:
        pass

    try:
        experiment_settings['cell_id'] = experiment_config.get('TIMESERIES_INPUT_DATA', 'cell_id')
    except:
        pass

    try:
        limits = experiment_config.get('SINGLE_EXPERIMENT_INFO', 'hard_thpt_limit').split(',')
        experiment_settings['hard_thpt_limit'] = [float(limit) for limit in limits]
    except:
        pass

    print('SINGLE EXPERIMENT SETTINGS', experiment_settings)
    # list of settings
    return experiment_settings

# utils for the simple cell simulator
def get_random_train_day(day_list = None, master_cell_records_dir = None, cell = None, file_prefix = 'MAST.CELX.', date_var = 'DATE_LOCAL', datetime_var = 'DATETIME_LOCAL'):

    day = random.choice(day_list)
    specific_day_csv = master_cell_records_dir + str(file_prefix) + str(cell) + '.' + str(day) + '.csv'

    CELX_df = adjust_timestamp_CELX(MAST_CELX_fname = specific_day_csv, datetime_var = datetime_var)

    return CELX_df, day

# USED
def random_train_day(day_list = None, master_cell_records_dir = None, cell = None, file_prefix = 'MAST.CELX.', datetime_var = 'DATETIME_LOCAL'):

    day = random.choice(day_list)

    specific_day_csv = master_cell_records_dir + '/' + str(file_prefix) + str(cell) + '.' + str(day) + '.csv'

    CELX_df = adjust_timestamp_CELX(MAST_CELX_fname = specific_day_csv, datetime_var = datetime_var)

    return CELX_df, day

# USED
def adjust_timestamp_CELX(MAST_CELX_fname = None, datetime_var = 'DATETIME_LOCAL'):
    ###############################################
    fmt = '%Y-%m-%d %H:%M:%S'
    cell_name = MAST_CELX_fname.split('.')[2]
    if cell_name[0] == '1':
    	datetime_var = 'DATETIME_Melbourne'
    try:
        dateparse = lambda dates: [pandas.datetime.strptime(d, fmt) for d in dates]
        CELX_df = pandas.read_csv(
            MAST_CELX_fname, parse_dates=[datetime_var],
            date_parser=dateparse).dropna()
        # CELX_df = pandas.read_csv(
        #     MAST_CELX_fname, parse_dates=[datetime_var],
        #     date_parser=dateparse)
        CELX_df.set_index(datetime_var, inplace=True)
    except:
        dateparse = lambda dates: [parser.parse(d) for d in dates]
        CELX_df = pandas.read_csv(
            MAST_CELX_fname, parse_dates=[datetime_var],
            date_parser=dateparse).dropna()
        # CELX_df = pandas.read_csv(
        #     MAST_CELX_fname, parse_dates=[datetime_var],
        #     date_parser=dateparse)
        CELX_df.set_index(datetime_var, inplace=True)
    return CELX_df


########################################################
def get_burstProb_actions(burst_prob_params_dict):
	min_burst_prob = burst_prob_params_dict['min_burst_prob']
	max_burst_prob = burst_prob_params_dict['max_burst_prob']
	burst_discretization =  burst_prob_params_dict['burst_discretization']
	optimal_burst_prob = burst_prob_params_dict['optimal_burst_prob'] 

	# burst_prob_vector = np.arange(min_burst_prob, max_burst_prob, burst_discretization)
	# num_actions = len(burst_prob_vector)
	num_actions = int((max_burst_prob-min_burst_prob)/burst_discretization) + 1
	burst_prob_vector = np.linspace(min_burst_prob, max_burst_prob, num_actions)
	action_space = spaces.Discrete(num_actions)

	burst_prob_to_action = {}
	action_to_burst_prob = {}

	for action_num, burst_prob in enumerate(burst_prob_vector):
		burst_prob_to_action[burst_prob] = action_num
		action_to_burst_prob[action_num] = burst_prob

	num_actions = action_space.n

	# optimal discretized burst
	diff = np.abs(burst_prob_vector - optimal_burst_prob)
	optimal_burst_index = diff.argmin()
	optimal_discretized_burst = burst_prob_vector[optimal_burst_index]
	print('optimal discretized burst', optimal_discretized_burst)
	print('num_actions', num_actions)
	print('burst_prob_vector', burst_prob_vector)
	print('optimal burst index', optimal_burst_index)

	burst_prob_params_dict['action_space'] = action_space 
	burst_prob_params_dict['num_actions'] = num_actions 
	burst_prob_params_dict['burst_prob_to_action'] = burst_prob_to_action 
	burst_prob_params_dict['action_to_burst_prob'] = action_to_burst_prob
	burst_prob_params_dict['burst_prob_vector'] = burst_prob_vector
	burst_prob_params_dict['optimal_discretized_burst'] = optimal_discretized_burst
	return burst_prob_params_dict




def load_burst_prob_params_dict(burst_discretization=None, max_burst_prob=None, min_burst_prob = None, optimal_burst_prob=0.25):

	burst_discretization = burst_discretization
	burst_prob_params_dict = {}
	burst_prob_params_dict['min_burst_prob'] = min_burst_prob
	burst_prob_params_dict['max_burst_prob'] = max_burst_prob
	burst_prob_params_dict['burst_discretization'] = burst_discretization
	burst_prob_params_dict['optimal_burst_prob'] = optimal_burst_prob
	burst_prob_params_dict = get_burstProb_actions(burst_prob_params_dict)

	return burst_prob_params_dict


def conf_load_reward_params_dict(alpha=None, beta=None, kappa = None, hard_thpt_limit = None, B_0 = 1.0, k=1, control_interval_seconds=60, avg_user_burst_prob=.05):
    reward_params_dict = {}
    reward_params_dict['alpha'] = alpha
    reward_params_dict['beta'] = beta
    # modulate hard thpt penalty: -activity*kappa*(K-B')
    reward_params_dict['kappa'] = kappa
    # value for K in above eqn
    reward_params_dict['hard_thpt_limit'] = hard_thpt_limit
    
    # for simple simulator B = B_0/C
    reward_params_dict['B_0'] = B_0
    reward_params_dict['k'] = k
    reward_params_dict['control_interval_seconds'] = control_interval_seconds
    reward_params_dict['avg_user_burst_prob'] = avg_user_burst_prob
    return reward_params_dict

def load_reward_params_dict(alpha=None, beta=None, k=None, control_interval_seconds=None, avg_user_burst_prob=None):

	reward_params_dict = {}
	reward_params_dict['alpha'] = alpha
	reward_params_dict['beta'] = beta
	reward_params_dict['k'] = k
	reward_params_dict['control_interval_seconds'] = control_interval_seconds
	reward_params_dict['avg_user_burst_prob'] = avg_user_burst_prob

	return reward_params_dict


