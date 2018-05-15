""" 
    Generic time-variant cell simulator
        1. handles Random Forest (RF_mode)
            - B = RANDOM_FOREST([C,N,E]) vector
        2. handles B = 1/C [basic_thpt_model_mode]
        
        3. delayed_reward_mode
    
    Requires hard_thpt_limit to be chosen correctly
    and KB_MB_converter to be set correctly

    KB_MB_converter = 1 for simple thpt calculation

"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import pandas
import sys, os

# load utils from the transition matrix
RL_ROOT_DIR = os.environ['RL_ROOT_DIR']
# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

# random forest utils
RF_utils_dir = RL_ROOT_DIR + '/random_forest/'
sys.path.append(RF_utils_dir)

# how to control a timeseries
impulse_response_utils_dir = RL_ROOT_DIR + '/simulate_env/impulse_response_utils/'
sys.path.append(impulse_response_utils_dir)

from simple_cell_utils import report_rewards, get_simplified_reward, get_simple_delayed_reward
from reward_computation_utils import *
from MAST_CELX_with_control import *
from score_random_forest import score_RF_single_input, load_saved_rf
from helper_utils_cell_simulator import random_train_day


class CombinedTimeVariantSimulatorEnv(gym.Env):
    """ openAI cell simulator that takes a MAST.CELX timeseries
    and a state vector s[t] = [c(t), c(t-1), ..., c(t-k)]. 
    
    The unperturbed system is s_u[t]. The dynamics are:

    s[t+1] = s_u[t+1] + A[t] + noise

    s_u[t+1] is the unperturbed dynamics for the next state (baseline 
    time-variant cell dynamics). The action, burst prob, A[t] adds
    to the baseline congestion level.

    Args:
        - env_params_dict: dictionary of env parameters

    Returns:
        - see env.reward_history_df for a pandas dataframe of all actions/rewards

    """

    def __init__(self, env_params_dict=None):

        print('COMBINED SIMULATOR')
        # env_params_dict: place all arguments for the simulator here
        ##############################################
        self.print_mode = env_params_dict['print_mode']

        self.continuous_action_mode = env_params_dict['continuous_action_mode']
        # parameters for reward computation
        self.reward_params_dict = env_params_dict['reward_params_dict']

        # for discrete actions, map a numeric action from 1 to N to a burst_prob from 0 to 1
        self.burst_prob_params_dict = env_params_dict['burst_prob_params_dict']
        # [C, A, N, E] = [0, 0, 0, 0]
        self.min_state_vector = env_params_dict['min_state_vector']
        # [C, A, N, E] = [1, 1, 1, 1]
        self.max_state_vector = env_params_dict['max_state_vector']
        self.thpt_var = env_params_dict['thpt_var']

        # use RF for B or B = 1/c [RF_mode, simple_thpt_mode]
        self.simulator_mode = env_params_dict['simulator_mode']

        # params for random forest
        #################################################
        if (self.simulator_mode == 'RF_mode'):
            # pkl file that maps state to B
            self.rf = load_saved_rf(env_params_dict['random_forest_model'])
            self.RF_feature_list = env_params_dict['RF_feature_list']
            self.KB_TO_MB = env_params_dict['KB_TO_MB']

        # params for delayed reward mode
        #################################################
        self.delayed_reward_mode = env_params_dict['delayed_reward_mode']
        self.DELAY_REWARD_INTERVAL = env_params_dict['DELAY_REWARD_INTERVAL']
        self.delayed_reward_col = 'DELAYED_REWARD'

        #################################################
        # PARAMETERS TO PREMATURELY ABORT THE simulation if throughput is too low
        # if we have a very low thpt for num_last_entries after 
        # min_iterations_before_done steps of the simulation is over, send a done and reset
        #################################################
        self.min_iterations_before_done = env_params_dict[
            'min_iterations_before_done']
        self.num_last_entries = env_params_dict['num_last_entries']
        self.bad_thpt_threshold = env_params_dict['bad_thpt_threshold']
        # does reward have [K-B'] penalty for throughput before hard limit of K?
        self.hard_thpt_limit_flag = env_params_dict['hard_thpt_limit_flag']
        self.premature_abort_flag = env_params_dict['premature_abort_flag']
        # if we randomly reset to middle of day, 
        # make sure we can at least have these many steps
        self.min_episode_steps = 30
        self.max_episode_steps = 50

        # a dataframe of state, action, reward history for logging
        self.reward_history_df = pandas.DataFrame()
        self.reward_history_df[self.delayed_reward_col] = 0.0
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
        self.cell_id = env_params_dict['cell_id']
        self.timeseries_dir = env_params_dict['timeseries_dir']
        self.train_days = env_params_dict['train_days']
        self.test_days = env_params_dict['test_days']
        self.TRAIN_MODE = True

        # parameters for reading off a timeseries file
        ##############################################
        self.history_minutes = env_params_dict['history_minutes']
        self.activity_factor_multiplier = env_params_dict[
            'activity_factor_multiplier']
        # how do we sample next state?
        self.state_propagation_method = env_params_dict[
            'state_propagation_method']
        self.noise_mean_std = env_params_dict['noise_mean_std']
        self.KPI_list = env_params_dict['KPI_list']
        # in case of a early stop, large negative reward
        self.LARGE_NEGATIVE_REWARD = -1000
        # for reward calculation

        # get an iterator that loops thru dates in CELX_df timeseries
        self.buffer_minutes = 2
        # if timeseries goes from [start, end] then only look at data until end - buffer_minutes
        self.experiment_params_dict = env_params_dict['experiment_params_dict']

        if (self.print_mode):
            print('FUNCTION init')
            print('action space', self.action_space)
            print('observation space', self.observation_space)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    """ 
        Dynamics: given s,a return s', reward, done
    """

    def _step(self, action):

        # convert action to a burst btwn [0,1]
        if (self.continuous_action_mode):
            burst = action
        else:
            action_to_burst_prob_dict = self.burst_prob_params_dict[
                'action_to_burst_prob']
            burst = action_to_burst_prob_dict[action]

        # get the current timestamp in simulation
        try:
            datetime_obj = self.minute_iterator_obj.next()
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
        next_controlled_state_dict, next_state_vec, next_controlled_state_vec = get_controlled_state_transition_per_scheme(
            datetime_obj=datetime_obj,
            df=self.CELX_df,
            history_minutes=self.history_minutes,
            KPI_list=self.KPI_list,
            activity_factor_multiplier=self.activity_factor_multiplier,
            state_propagation_method=self.state_propagation_method,
            state_dict=self.state_dict,
            action=float_action,
            KPI='CELLT_AGG_COLL_PER_TTI_DL',
            print_mode=False,
            noise_mean_std=self.noise_mean_std)

        # compute throughputs based on random forest [DIFFERENT FROM TIME-VARIANT] 
        ###############################################################
        # original cell thpt is inversely related to current state C
        if (self.simulator_mode == 'RF_mode'):

            KPI = 'CELLT_AGG_COLL_PER_TTI_DL'
            current_coll_metric = self.state_dict[KPI + '_LAG_0']
            next_coll_metric = next_controlled_state_dict[KPI + '_LAG_0']

            RF_state_dict = {}
            RF_next_controlled_state_dict = {}

            for KPI in self.RF_feature_list:
                RF_state_dict[KPI] = self.state_dict[KPI + '_LAG_0']
                RF_next_controlled_state_dict[KPI] = next_controlled_state_dict[
                    KPI + '_LAG_0']

            orig_thpt, orig_state_RF_input = score_RF_single_input(
                ordered_feature_list=self.RF_feature_list,
                rf=self.rf,
                state_df=RF_state_dict)

            new_cell_thpt, next_state_RF_input = score_RF_single_input(
                ordered_feature_list=self.RF_feature_list,
                rf=self.rf,
                state_df=RF_next_controlled_state_dict)
        elif (self.simulator_mode == 'simple_thpt_mode'):
            pass
        else:
            pass

        # predicted_y,X = score_RF_single_input(ordered_feature_list = None, rf = None, state_df = None)
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

        # compute the reward based on the action, current state
        reward, burst, PPC_data_MB_scheduled, user_lost_data_MB, hard_thpt_limit_MB = get_simplified_reward(
            action=burst,
            reward_params_dict=self.reward_params_dict,
            print_mode=self.print_mode,
            burst_prob_user_selector='same_as_PPC',
            orig_thpt=orig_thpt,
            hard_thpt_limit_flag=self.hard_thpt_limit_flag,
            new_cell_thpt=new_cell_thpt,
            KB_MB_converter=self.KB_TO_MB)

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
            hard_thpt_limit_MB=hard_thpt_limit_MB,
            experiment_params_dict=self.experiment_params_dict)
        ###############################################################

        # self.reward_history_df has the per-timestep reward, if the timestamp is a DELAY_REWARD_INTERVAL boundary, get the cumulative reward over the last N mins and write to 'delayed reward' column
        reward_history_df_index = self.reward_history_df.shape[0]

        delayed_reward, self.reward_history_df = get_simple_delayed_reward(
            iteration_index=reward_history_df_index,
            DELAY_REWARD_INTERVAL=self.DELAY_REWARD_INTERVAL,
            total_reward_history_df=self.reward_history_df,
            delayed_reward_col=self.delayed_reward_col)

        # 0 for most times, a spike of cumulative reward every DELAY_REWARD_INTERVAL
        if (self.delayed_reward_mode):
            reward = delayed_reward

            if(self.print_mode):
                if(delayed_reward > 0):
                    print('delayed_reward: ', delayed_reward)
                    print('index: ', reward_history_df_index)

        # just for debugging
        self.reward = reward
        self.action_taken = burst

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

        # PREMATURE ABORT STEP IF TOO BAD A THROUGHPUT
        # if we reached enough timestamps and the last N thpts are below 
        # a threshold, abort
        if (self.premature_abort_flag):
            if (self.iteration_index >= self.min_iterations_before_done):
                # if last N points had a thpt below threshold, abort the simulation
                num_entries_low_thpt = (self.reward_history_df[self.thpt_var][
                    -self.num_last_entries:] <= self.bad_thpt_threshold).sum()
                fraction_bad_thpt_entries = float(num_entries_low_thpt) / float(
                    self.num_last_entries)
                if (fraction_bad_thpt_entries >= .9):
                    done_flag = True
                    print('PREMATURE ABORT DUE TO LOW THRUPUT')
                    print('num_entries_low_thpt', num_entries_low_thpt)
                    print('bad_thpt_threshold', self.bad_thpt_threshold)
                    print('last', self.reward_history_df[self.thpt_var][
                        -self.num_last_entries:])
                    reward = self.LARGE_NEGATIVE_REWARD
                    print('premature reward', reward)
        return next_controlled_state_vec, reward, done_flag, {}

    """ 
        Reset timeseries to start of day for a new episode
        OR random time in day
    """

    def _reset(self):
        # self.cell_id = env_params_dict['cell_id']
        # self.timeseries_dir = env_params_dict['timeseries_dir']
        # self.train_days = env_params_dict['train_days']
        # self.test_days = env_params_dict['test_days']

        # get a list of random training and test days
        if (self.TRAIN_MODE):
            self.CELX_df, day = random_train_day(
                day_list=self.train_days,
                master_cell_records_dir=self.timeseries_dir,
                cell=self.cell_id)
            print('RF TRAIN DAY ', day)
        else:
            self.CELX_df, day = random_train_day(
                day_list=self.test_days,
                master_cell_records_dir=self.timeseries_dir,
                cell=self.cell_id)
            print('RF TEST DAY ', day)

        if self.random_reset_mode:
            self.minute_iterator_obj, start_date, end_date = get_random_dateTime_iterator(
                buffer_minutes=self.buffer_minutes,
                history_minutes=self.history_minutes,
                CELX_df=self.CELX_df,
                MIN_STEP_MINUTES=self.min_episode_steps,
                MAX_STEP_MINUTES=self.max_episode_steps)

        else:
            self.minute_iterator_obj, start_date, end_date = get_dateTime_iterator_from_CELX(
                CELX_df=self.CELX_df,
                buffer_minutes=self.buffer_minutes,
                history_minutes=self.history_minutes)
        # print('reset episode mode ', self.random_reset_mode, start_date.__str__(), end_date.__str__())

        # get the initial state dict and state vec
        self.state_dict, self.state, subset_df = get_state_with_history_from_CELX(
            df=self.CELX_df,
            end_datetime_obj=start_date,
            history_minutes=self.history_minutes,
            KPI_list=self.KPI_list)

        # if we dont randomly reset, we are testing, only increment batch number then
        if not self.random_reset_mode:
            self.batch_number += 1

        self.iteration_index = 0

        if (self.print_mode):
            print('FUNCTION reset')
            print('state', self.state)
            print('state_dict', self.state_dict)
        return self.state
