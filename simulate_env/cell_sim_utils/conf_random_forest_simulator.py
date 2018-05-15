""" 
    same as openAI time-variant cell simulator
    but B = RANDOM_FOREST([C,N,E]) vector
    
    Requires hard_thpt_limit to be chosen correctly
    and KB_MB_converter to be set correctly

    Reads most params from a conf file

"""
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pandas
import sys, os
import ConfigParser

# load utils from the transition matrix
RL_ROOT_DIR = os.environ['RL_ROOT_DIR']
# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

# random forest utils
RF_utils_dir = RL_ROOT_DIR + '/random_forest/'
sys.path.append(RF_utils_dir)

# utils
utils_dir = RL_ROOT_DIR + '/utils/'
sys.path.append(utils_dir)

# how to control a timeseries
impulse_response_utils_dir = RL_ROOT_DIR + '/simulate_env/impulse_response_utils/'
sys.path.append(impulse_response_utils_dir)

from simple_cell_utils import report_rewards, get_simplified_reward
from MAST_CELX_with_control import get_state_with_history_from_CELX, get_controlled_state_transition_per_scheme, get_dateTime_iterator_from_CELX
from score_random_forest import score_RF_single_input, load_saved_rf
from helper_utils_cell_simulator import random_train_day, get_burstProb_actions 
from textfile_utils import list_from_textfile

class ConfRandomForestTimeVariantSimulatorEnv(gym.Env):

    """ openAI cell simulator that takes a MAST.CELX timeseries
    and a state vector s[t] = [c(t), c(t-1), ..., c(t-k)]. 
    
    The unperturbed system is s_u[t]. The dynamics are:

    s[t+1] = s_u[t+1] + A[t] + noise

    s_u[t+1] is the unperturbed dynamics for the next state (baseline 
    time-variant cell dynamics). The action, burst prob, A[t] adds
    to the baseline congestion level.

    Args:
        - experiment_settings: dictionary of env parameters

    Returns:
        - None

    """
    # SC progress: Done

    def __init__(self, experiment_config = None, file_path_config = None, experiment_settings = None):

        print('CONF COMBINED SIMULATOR')
        ##############################################
        self.print_mode = experiment_config.getboolean('EXPERIMENT_INFO', 'print_mode')
        self.env_type = experiment_config.get('RL_AGENT_INFO', 'env_type')
        print('SIMULATING', self.env_type)

        # params for random forest
        #################################################
        rf_models = experiment_settings['rf_models']
        self.hard_thpt_limit = experiment_settings['hard_thpt_limit'] 
        if(self.env_type == 'random_forest'):
            
            # self.rf_list, self.rf_models = generate_random_forest_head_list(train_info_csv = train_info_csv)
            self.rf_list = []
            for rf_model in rf_models:
                print('using model', rf_model)
                self.rf_list.append(load_saved_rf(rf_model))

            random_forest_feature_list = experiment_config.get('RANDOM_FOREST_PARAMS', 'random_forest_feature_list')
            self.RF_feature_list = list_from_textfile(random_forest_feature_list)
       
        # if simple_thpt model, B = 1/C

        self.thpt_var = experiment_config.get('RANDOM_FOREST_PARAMS', 'thpt_var')
        self.KB_TO_MB = experiment_config.get('RANDOM_FOREST_PARAMS', 'KB_TO_MB')
        #################################################

        # parameters for reward computation and actions
        #################################################
        self.burst_prob_params_dict = experiment_settings['burst_prob_params_dict']
        self.reward_params_dict = experiment_settings['reward_params_dict']
        # for discrete actions, map a numeric action from 1 to N to a burst_prob from 0 to 1
        self.continuous_action_mode = experiment_config.getboolean('ACTION_SPACE', 'continuous_action_mode')
        
        # min and max action spaces
        # [C, A, N, E] = [0, 0, 0, 0]
        self.min_state_vector = experiment_settings['min_state_vector']
        # [C, A, N, E] = [1, 1, 1, 1]
        self.max_state_vector = experiment_settings['max_state_vector']

        # does reward have [K-B'] penalty for throughput before hard limit of K?
        self.hard_thpt_limit_flag = experiment_settings['hard_thpt_limit_flag']
        
        # a dataframe of state, action, reward history for logging
        self.reward_history_df = pandas.DataFrame()
        self.iteration_index = 0
        # batch = EPISODE
        self.batch_number = 0
        
        ##############################################
        # what range are the burst probabilities?
        # map discrete/continuous actions to bursts
        if (self.continuous_action_mode):
            self.action_space = spaces.Box(
                low=self.burst_prob_params_dict['min_burst_prob'],
                high=self.burst_prob_params_dict['max_burst_prob'],
                shape=(1,))
        else:
            # discrete actions
            self.burst_prob_params_dict = get_burstProb_actions(
                self.burst_prob_params_dict)
            self.action_space = self.burst_prob_params_dict['action_space']

        # state space
        ##############################################
        self.observation_space = spaces.Box(low=self.min_state_vector,
                                            high=self.max_state_vector)

        # params to train/test on separate days
        #################################################
        self.cell_id = experiment_config.get('TIMESERIES_INPUT_DATA', 'cell_id')
        self.timeseries_dir = experiment_config.get('TIMESERIES_INPUT_DATA', 'timeseries_dir')
        self.train_days = experiment_config.get('TRAIN_TEST_SPLIT', 'train_days').split(',')
        self.test_days = experiment_config.get('TRAIN_TEST_SPLIT', 'test_days').split(',')
        # self.TRAIN_MODE = True

        # parameters for experiments - these change per run so not from a file
        ##############################################
        self.history_minutes = experiment_settings['history_minutes']
        self.activity_factor_multiplier = experiment_settings['activity_factor_multiplier']
        # how do we sample next state?
        self.state_propagation_method = experiment_config.get('RL_AGENT_INFO', 'state_propagation_method')
        self.noise_mean_std = [float(x) for x in experiment_config.get('RL_AGENT_INFO', 'noise_mean_std').split(',')]
        self.KPI_list = experiment_config.get('STATE_SPACE', 'state_features').split(',') 
        self.experiment_settings = experiment_settings
        self.train_inds = experiment_settings['train_inds']
        self.test_inds = experiment_settings['test_inds']
        self.TRAIN_MODE = True
        self.ind = None
        self.rf_ind = None

        if (self.print_mode):
            print('FUNCTION init')
            print('action space', self.action_space)
            print('observation space', self.observation_space)
        self._seed()

    """ 
        Dynamics: given s,a return s', reward, done,
        Log the rewards over time to visualize later
    """

    # SC progress: Done
    def _step(self, action):
        # convert action to a burst btwn [0,1]
        if (self.continuous_action_mode):
            burst = action
        else:
            action_to_burst_prob_dict = self.burst_prob_params_dict[
                'action_to_burst_prob']
            burst = np.asarray([action_to_burst_prob_dict[action]])

        # get the current timestamp in simulation
        try:
            datetime_obj = self.minute_iterator_obj.next()
            datetime_str = datetime_obj.__str__()
            # print(' time ', datetime_obj.__str__(), 'input action ', burst)
        # if end of day, episode is done
        except StopIteration:
            print('END DAY batch', self.batch_number)
            next_state = self.state
            reward = 0
            done_flag = True
            return next_state, reward, done_flag, {}

        # self.state exists, now do the state update based on action
        float_action = float(burst[0])

        # state update dynamics
        next_controlled_state_dict, next_state_vec, next_controlled_state_vec = get_controlled_state_transition_per_scheme(datetime_obj = datetime_obj, df = self.CELX_df, history_minutes = self.history_minutes, KPI_list = self.KPI_list, activity_factor_multiplier = self.activity_factor_multiplier, state_propagation_method = self.state_propagation_method, state_dict = self.state_dict, action = float_action, KPI = 'CELLT_AGG_COLL_PER_TTI_DL', print_mode = False, noise_mean_std = self.noise_mean_std)

        # compute throughputs based on random forest
        ###############################################################
        KPI = 'CELLT_AGG_COLL_PER_TTI_DL'
    
        if(self.env_type == 'random_forest'):
            current_coll_metric = self.state_dict[KPI + '_LAG_0']
            next_coll_metric = next_controlled_state_dict[KPI + '_LAG_0']

            RF_state_dict = {}
            RF_next_controlled_state_dict = {}

            for KPI in self.RF_feature_list:
                RF_state_dict[KPI] = self.state_dict[KPI + '_LAG_0']
                RF_next_controlled_state_dict[KPI] = next_controlled_state_dict[KPI + '_LAG_0']

            orig_thpt, orig_state_RF_input = score_RF_single_input(ordered_feature_list = self.RF_feature_list, rf = self.rf_list[self.rf_ind], state_df = RF_state_dict)

            new_cell_thpt, next_state_RF_input = score_RF_single_input(ordered_feature_list = self.RF_feature_list, rf = self.rf_list[self.rf_ind], state_df = RF_next_controlled_state_dict)

        ###############################################################
        # original cell thpt is inversely related to current state C
        elif(self.env_type == 'simple_thpt_model'):
            current_coll_metric = self.state_dict[KPI + '_LAG_0']
            next_coll_metric = next_controlled_state_dict[KPI + '_LAG_0']

            orig_thpt = float(self.reward_params_dict['B_0']) / float(current_coll_metric)
            new_cell_thpt = float(self.reward_params_dict['B_0']) / float(next_coll_metric)
        else:
            print('not supported', self.env_type)

        ###############################################################
        if (self.print_mode):
            print('state', self.state)
            print('action', float_action)
            print('burst', burst)
            print('next_state', next_controlled_state_vec)
            print('current_coll_metric', current_coll_metric)
            print('orig_thpt', orig_thpt)
            print('next_coll_metric', next_coll_metric)
            print('new_cell_thpt', new_cell_thpt)
        # print 'orig_thpt:%r, new_cell_thpt:%r' % (orig_thpt, new_cell_thpt)
        self.reward_params_dict['hard_thpt_limit'] = self.hard_thpt_limit[self.rf_ind]

        # compute the reward based on the action, current state
        reward, burst, PPC_data_MB_scheduled, user_lost_data_MB, hard_thpt_limit_MB = get_simplified_reward(
            action=burst,
            reward_params_dict=self.reward_params_dict,
            print_mode=self.print_mode,
            burst_prob_user_selector='same_as_PPC',
            orig_thpt=orig_thpt,
            hard_thpt_limit_flag=self.hard_thpt_limit_flag,
            new_cell_thpt = new_cell_thpt,
            KB_MB_converter = self.KB_TO_MB)

        # logging step
        # append the state information and reward to reward_history_df, which is
        # used for later plotting
        self.reward_history_df = report_rewards(
            state=current_coll_metric,
            burst=burst,
            reward=reward,
            reward_history_df=self.reward_history_df,
            iteration_index=self.iteration_index,
            PPC_data_MB_scheduled=PPC_data_MB_scheduled,
            user_lost_data_MB=user_lost_data_MB,
            print_mode=self.print_mode,
            thpt=orig_thpt,
            new_thpt=new_cell_thpt,
            thpt_var=self.thpt_var,
            batch_number=self.batch_number,
            hard_thpt_limit_MB = hard_thpt_limit_MB, 
            experiment_params_dict = self.experiment_settings,
            datetime_str = datetime_str, 
            train_test_mode = self.TRAIN_MODE,
            day = self.day,
            ind = self.rf_ind)
        ###############################################################
        # update the state
        self.state_dict = next_controlled_state_dict
        self.state = next_controlled_state_vec

        # update the state and simulation params
        done_flag = False
        self.iteration_index += 1

        if (self.print_mode):
            print('FUNCTION step')
            print('curr_state', self.state)
            print('next_state', next_controlled_state_vec)
            print('reward', reward)
            print('done_flag', done_flag)

        return next_controlled_state_vec, reward, done_flag, {}

    """ 
        Reset timeseries to start of day for a new episode
    """

    # SC progress: Done
    def _reset(self):
        # get a list of random training and test days
        if self.ind is None:
            if self.TRAIN_MODE == True:
                self.CELX_df, self.day = random_train_day(day_list = self.train_days, master_cell_records_dir = self.timeseries_dir, cell = self.cell_id)
                print('RF TRAIN DAY ', self.day)
            else:
                self.CELX_df, self.day = random_train_day(day_list = self.test_days, master_cell_records_dir = self.timeseries_dir, cell = self.cell_id)
                print('RF TEST DAY ', self.day)
        else:
            cell_id, day = self.ind[0], self.ind[1]
            self.CELX_df, self.day = random_train_day(day_list = [day], master_cell_records_dir = self.timeseries_dir, cell = cell_id)
            if self.TRAIN_MODE == True:
                print('RF TRAIN DAY ', self.day)
            else:
                print('RF TEST DAY ', self.day)

        self.minute_iterator_obj, start_date, end_date = get_dateTime_iterator_from_CELX(CELX_df = self.CELX_df, buffer_minutes = 2, history_minutes = self.history_minutes)

        # get the initial state dict and state vec
        self.state_dict, self.state, subset_df = get_state_with_history_from_CELX(df = self.CELX_df, end_datetime_obj = start_date, history_minutes = self.history_minutes, KPI_list = self.KPI_list)

        self.batch_number += 1
        self.iteration_index = 0

        if (self.print_mode):
            print('FUNCTION reset')
            print('state', self.state)
            print('state_dict', self.state_dict)
        return self.state
