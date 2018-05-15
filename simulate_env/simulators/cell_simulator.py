import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import cPickle
import json
import pandas
import sys,os
from reward_computation_utils import *
from sample_transition_matrix_utils import *

# load utils from the transition matrix
RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + '/state_space_generation'
sys.path.append(util_dir)

class CellSimulatorEnv(gym.Env):
    def __init__(self, CELX_df=None, transition_matrix_fname=None, burst_prob_params_dict = None, cell_id='136046093', print_mode=True, reset_state_vector=None, deterministic_reset_mode=None, RF_model_pkl_file=None, sampling_mode=None, reward_params_dict=None, KPI_list=None, min_state_vector=None, max_state_vector=None, time_var=None, thpt_var=None, delayed_reward_mode=None, delay_reward_interval=None, continuous_action_mode=False, date_var=None, state_value_map_df=None, max_thpt_value=None):

        # MAST.CELX file
        # transition matrix
        # init state, cell_id, num_ppc users, burst prob vector
        self.transition_matrix_fname = transition_matrix_fname
        self.print_mode = print_mode
        self.deterministic_reset_mode = deterministic_reset_mode

	self.time_var = time_var
	self.date_var = date_var
	self.CELX_df = None
	self.row_number = None
	self.RF_model=RF_model_pkl_file
	#self.RF_model = load_RF_pkl(RF_model_pkl_file)
	self.trans_matrix = None
	# init the reward history df
	self.reward_history_df = pandas.DataFrame()
	self.state_tuple_map = None
	self.batch_number = None
	self.thpt_var = thpt_var
	self.continuous_action_mode = continuous_action_mode

	if(sampling_mode == 'deterministic_CELX_file_mode'): 
		self.CELX_df = CELX_df
		self.row_number = 1
		self.timestamp = None
		self.batch_number = 0
	elif(sampling_mode == 'sample_trans_matrix_mode'):
		with open(self.transition_matrix_fname) as fp:
			self.json_object = json.load(fp)

		self.state_value_map_df = state_value_map_df
		self.state_index_var = 'STATE_INDEX'
		self.state_value_map_df.set_index(self.state_index_var, inplace = True)
		self.state_value_map_dict = self.state_value_map_df.to_dict()	

		self.iteration_index = 0
		self.batch_number = 0

		# load these programatically later
		year = 2016
		month = 01
		day = 01
		hour = 11
		minute = 30
		second = 0
		rest = 0
		minute_duration = 90
		self.matrix_bins_minutes = 15
		self.simulation_date_str = '_'.join([str(year), str(month), str(day)])

		self.start_date = datetime.datetime(year, month, day, hour, minute, second, rest)
		self.end_date = self.start_date + datetime.timedelta(minutes=minute_duration)
		self.sampling_procedure = 'prob'
		self.minute_iterator_obj = datetimeIterator(from_date = self.start_date, to_date = self.end_date)

	elif(sampling_mode == 'RF_update_mode'):
		pass
	else:
		pass

	# how to move between states? value can be:
	    #    1. read off CELX file: deterministic_CELX_file_mode
	    #    2. start with init state and sample matrix A with no action: sample_trans_matrix_mode    
	    #    3.  s, a, s' according to update model: RF_update_mode
        self.sampling_mode = sampling_mode
	# has fields: alpha and beta
        self.reward_params_dict = reward_params_dict

        # RF feature information
	# [C, A, N, E]
        self.KPI_list = KPI_list

        # [0, .10, .2, .5, 1.0]
        self.cell_id = cell_id

	self.burst_prob_params_dict = burst_prob_params_dict

        # [C, A, N, E] = [0, 0, 0, 0]
        self.min_state_vector = min_state_vector
        # [C, A, N, E] = [1, 1, 1, 1]
        self.max_state_vector = max_state_vector
	# value to reset to, such as min_state_vector
        self.reset_state_vector = reset_state_vector

	self.max_thpt_value = max_thpt_value

        # state is a continuous value of C,A,N,E = 4
        self.num_state_features = len(np.shape(self.min_state_vector))

        # what range are the burst probabilities?

	if(self.continuous_action_mode):
		self.action_space = spaces.Box(low=self.burst_prob_params_dict['min_burst_prob'], high=self.burst_prob_params_dict['max_burst_prob'], shape=(1,))
	else:
		# discrete actions
		self.burst_prob_params_dict = get_burstProb_actions(self.burst_prob_params_dict)
		self.action_space = self.burst_prob_params_dict['action_space']

        self.observation_space = spaces.Box(low=self.min_state_vector, high=self.max_state_vector)
	self.iteration_index = 1
	self.state_update_mode = 'linear'
	self.delayed_reward_mode = delayed_reward_mode
	self.delay_reward_interval = delay_reward_interval

	# params for the delayed reward
	self.index_var_name = 'ITERATION_INDEX'
	self.delayed_reward_col = 'DELAYED_REWARD'

	self.reward_history_df[self.delayed_reward_col] = 0.0

        if(self.print_mode):
            print('FUNCTION init')
            print('cell', self.cell_id)
            print('continuous action mode', self.continuous_action_mode)
            print('action space', self.action_space)
            print('observation space', self.observation_space)
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # fill in this here
    # MODES: 
    #    - read off CELX file: deterministic_CELX_file_mode
    #    - start with init state and sample matrix A with no action: sample_trans_matrix_mode    
    #    - s, a, s' according to update model: RF_update_mode
    #        - PPC update
    #        - reward update
    # key step: environment discretization

    def _step(self,action):
        if(self.sampling_mode == 'deterministic_CELX_file_mode'):
            # sample from CELX file
            next_state, self.timestamp, self.batch_number, row_celx_df, date = get_CELX_line(self.CELX_df, self.row_number, self.KPI_list, self.print_mode, self.time_var, self.date_var)
            
            self.row_number = self.row_number + 1
	    self.iteration_index = self.row_number

	    # reward update
            # reward, state, action, next_state = get_reward(self.state, action, next_state, self.RF_model, self.reward_params_dict, self.print_mode)           

	    if(self.continuous_action_mode):
		burst = action
	    else:
		action_to_burst_prob_dict = self.burst_prob_params_dict['action_to_burst_prob']
		burst = action_to_burst_prob_dict[action] 
	    if(self.print_mode):
		    print('action', action)
		    print('burst', burst)

            instantaneous_reward, state, burst, PPC_data_MB_scheduled, user_lost_data_MB = get_data_MB_reward(self.state, burst, self.RF_model, self.reward_params_dict, self.KPI_list, self.thpt_var, self.state_update_mode, self.print_mode)            

	    self.reward_history_df = create_state_reward_dict(state, burst, instantaneous_reward, self.KPI_list, self.reward_history_df, self.timestamp, self.iteration_index, self.batch_number, PPC_data_MB_scheduled, user_lost_data_MB, self.print_mode, date, self.time_var, self.date_var)

	    # self.reward_history_df has the per-timestep reward, if the timestamp is a 15 minute boundary, get the cumulative reward over the last N mins and write to 'delayed reward' column
	    delayed_reward, self.reward_history_df = get_delayed_reward(self.iteration_index, self.delay_reward_interval, self.reward_history_df, self.index_var_name, self.delayed_reward_col)

	    if(self.delayed_reward_mode):
		    reward = delayed_reward
	    else:
		    reward = instantaneous_reward

            done_flag = False

        elif(self.sampling_mode == 'sample_trans_matrix_mode'):
	    
	    try:
		    self.datetime_obj = self.minute_iterator_obj.next()
		    #print('sampling ', self.sampling_mode, ' time ', self.datetime_obj.__str__(), 'input action ', action)
	    except StopIteration:
		    #print('END DAY sampling ', self.sampling_mode)
		    next_state = self.state
		    reward = 0
		    done_flag = True
		    return next_state, reward, done_flag, {}
	    
	    # load time variant transition
	    #print('loaded trans matrix')
	    T, non_NA_rows = load_transition_matrices(datetime_obj = self.datetime_obj, matrix_bins_minutes = self.matrix_bins_minutes, json_object = self.json_object)
	    
	    time_str = datetime_to_timeStr(self.datetime_obj)

	    self.timestamp = self.datetime_obj.__str__()
	    self.iteration_index = self.iteration_index + 1

	    # which 15 min matrix to start from?
	    binned_datetime_obj = round_datetime_to_binned_minute(self.datetime_obj, self.matrix_bins_minutes)
	    
	    # get a state according to various sampling procedures
	    #print('got specific state')
	    state_num, trans_matrix_info_dict = get_specific_state(T=T, non_NA_rows=non_NA_rows, sampling_procedure='prob')

	    # add the action to the collision metric
	    #print('get controlled action')
	    next_state_num, next_state, orig_thpt = get_controlled_new_state(state_num=state_num, action=action, trans_matrix_info_dict=trans_matrix_info_dict, state_value_map_dict=self.state_value_map_dict, KPI_list=self.KPI_list, thpt_var = self.thpt_var, max_thpt_value=self.max_thpt_value)

	    # reward update
	    #print('get reward')
            reward, state, action, PPC_data_MB_scheduled, user_lost_data_MB = get_data_MB_reward(next_state, action, self.RF_model, self.reward_params_dict, self.KPI_list, self.thpt_var, self.state_update_mode, self.print_mode, orig_thpt=orig_thpt)     
	    #print('reward ', reward) 
	    #print('state ', state)
	    #print('action ', action)
	    #print('KPI ', self.KPI_list)
	    #print('cts action ', self.continuous_action_mode)
	    self.state_dim = self.observation_space.shape[0]
	    self.action_dim = self.action_space.shape[0]

	    #print('state dim ', self.state_dim)
	    #print('action dim ', self.action_dim)
	    #print('report reward')
	    self.reward_history_df = create_state_reward_dict(state, action, reward, self.KPI_list, self.reward_history_df, self.timestamp, self.iteration_index, self.batch_number, PPC_data_MB_scheduled, user_lost_data_MB, self.print_mode, self.simulation_date_str, self.time_var, self.date_var) 

	    # check if cumulative reward in last N timestamps was too low and action was greater than zero, then abort
            done_flag = False

        elif(self.sampling_mode == 'RF_update_mode'):
            pass
        else:
            pass

        if(self.print_mode):
            print('FUNCTION step')
            print('det reset mode', self.deterministic_reset_mode)
            print('curr_state', self.state)
            print('next_state', next_state)
            print('reward', reward)
            print('done_flag', done_flag)

	# update the state!
	self.state = next_state
        return next_state, reward, done_flag, {}

    def _reset(self):
        if(self.deterministic_reset_mode):
            self.state = self.reset_state_vector
        else:
            self.state = self.np_random.uniform(low=self.min_state_vector, high=self.max_state_vector)

        if(self.sampling_mode == 'deterministic_CELX_file_mode'):
            # sample from CELX file
	    self.row_number = 1
            self.state, self.timestamp, self.batch_number, row_celx_df, date = get_CELX_line(self.CELX_df, self.row_number, self.KPI_list, self.print_mode, self.time_var, self.date_var)
	    print('reset episode deterministic ', 'state ', self.state,  'row_number ', self.row_number)

	elif(self.sampling_mode == 'sample_trans_matrix_mode'):
	    # reset the transition matrix vector
	    print('reset episode sample_trans_matrix_mode ', 'state ', self.state)
	    self.minute_iterator_obj = datetimeIterator(from_date = self.start_date, to_date = self.end_date)
	    self.batch_number += 1
	    self.state = self.min_state_vector
	    self.reward = 0

        if(self.print_mode):
            print('FUNCTION reset')
            print('det reset mode', self.deterministic_reset_mode)
            print('state', self.state)
        return self.state

    def _render(self, mode='human', close=False):
        if(self.print_mode):
            print('FUNCTION render')
            print('render func not implemented yet')
        return

