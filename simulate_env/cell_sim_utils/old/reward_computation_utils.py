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
util_dir = RL_ROOT_DIR + '/state_space_generation'
sys.path.append(util_dir)

########################################################

def get_delayed_reward(df_index = None, DELAY_REWARD_INTERVAL = None, reward_history_df = None, index_var = None, delayed_reward_col = None):

    # only add delayed reward every DELAY_REWARD_INTERVAL 
	if( (df_index % DELAY_REWARD_INTERVAL == 0) & (df_index > 1) ):

		#print('df_index', df_index)
		#print('dim reward history df', reward_history_df.shape)
		#print('original index')
		#print('index reward history df', reward_history_df.index)
		#print('new index')
		reward_history_df.index = range(reward_history_df.shape[0])
		#print('index reward history df', reward_history_df.index)
		
		min_bound = np.max([0, df_index - DELAY_REWARD_INTERVAL])
		max_bound = df_index

		subselect_df = reward_history_df[(reward_history_df[index_var] >= min_bound) & (reward_history_df[index_var] <= max_bound)]
		#print('min_bound', min_bound)
		#print('max_bound', max_bound)

		total_reward = sum(subselect_df['REWARD'])
		#print('total_reward', total_reward)

		# location of the 
		#print('index subselect_df', subselect_df.index)
		index_of_delayed_reward = subselect_df[subselect_df[index_var] == max_bound].index[0]
		#print('index_delayed_reward', index_of_delayed_reward)

		subselect_df[delayed_reward_col] = 0

		subselect_df.loc[index_of_delayed_reward, delayed_reward_col] = total_reward
		reward_history_df.loc[index_of_delayed_reward, delayed_reward_col] = total_reward
		#print('reward_history_df', reward_history_df.loc[min_bound:max_bound][delayed_reward_col])

		#print(subselect_df)
		#print(reward_history_df)

		point_reward = total_reward
	else:
		point_reward = 0.0

	#print('df_index', df_index)
	#print('point_reward', point_reward)

	return point_reward, reward_history_df



def extract_state_dict(state_vector, KPI_list):
	state_dict = {}

	for state_index, KPI in enumerate(KPI_list):
		state_dict[KPI] = state_vector[state_index]

	return state_dict

# from the transition utils we need
def compute_new_thpt(orig_thpt, action, reward_params_dict, state_update_mode):   
	if(state_update_mode == 'linear'):
		# ranges [1/k*orig_thpt, orig_thpt]
		k = reward_params_dict['k']
		new_thpt = orig_thpt*(1 - action/k)
	elif(state_update_mode == 'exponential'):
		new_thpt = None 
	else:
		new_thpt = None
	# this should eventually have the PPC update rule

	return new_thpt

def create_state_reward_dict(state, action, reward, KPI_list, reward_history_df, timestamp, iteration_index, batch_number, PPC_data_MB_scheduled, user_lost_data_MB, print_mode, date, time_var, date_var):

	# some examples here
	# state = np.array([4.35550145482188, 1.32357679201152, 50.507866656780202,
	# 	       198.38478223786299], dtype=object)

	# next_state = np.array([2.35550145482188, 2.32357679201152, 55.507866656780202,
	# 	       150.38478223786299], dtype=object)

	# reward = np.array([ 10606.90105631])

	# KPI_list = ['CELLT_AGG_COLL_PER_TTI_DL', 'CELLT_AGG_SPECF_DL', 'CELLT_AVG_NUM_SESS', 'CELLT_SUM_THPTIME_DL']
	df_dict = {}
	#df_dict['REWARD'] = [reward[0]]
	df_dict['REWARD'] = reward
	df_dict[time_var] = timestamp
	df_dict['ITERATION_INDEX'] = iteration_index
	df_dict['ACTION'] = action
	df_dict['BATCH_NUMBER'] = batch_number
	df_dict['DELAYED_REWARD'] = 0.0
	df_dict['PPC_DATA_SCHEDULED_MB'] = PPC_data_MB_scheduled
	df_dict['USER_LOST_DATA_MB'] = user_lost_data_MB
	df_dict[date_var] = date

	for i,KPI in enumerate(KPI_list):
		curr_state_str = KPI
		df_dict[curr_state_str] = [state[i]]
		
	local_df = pandas.DataFrame(df_dict)

	if(print_mode):
        	print('FUNCTION create_state_reward_dict')
		print('df dict', df_dict)
		print('local df', local_df)

	reward_history_df = reward_history_df.append(local_df)
	return reward_history_df


def load_RF_pkl(save_random_forest_pkl):
    # load the rf module
    with open(save_random_forest_pkl, 'rb') as f:
        rf = cPickle.load(f)
        return rf

def get_CELX_line(CELX_df, row_number, KPI_list, print_mode, time_var, date_var):

    total_rows = CELX_df.shape[0]

    effective_row_number = row_number % total_rows
    effective_batch_number = np.floor(row_number/total_rows)

    if(print_mode):
        print('FUNCTION get_CELX_line')
        print('total rows', total_rows)
        print('effective_row_number', effective_row_number)
        print('effective_batch_number', effective_batch_number)

    #celx_line = np.array(CELX_df.ix[effective_row_number][KPI_list])
    #timestamp = CELX_df.ix[effective_row_number][time_var]

    celx_line = np.array(CELX_df.iloc[effective_row_number][KPI_list])
    timestamp = CELX_df.iloc[effective_row_number][time_var]
    date = CELX_df.iloc[effective_row_number][date_var]

    celx_row_df = CELX_df.iloc[effective_row_number]

    if(print_mode):
        print('FUNCTION get_CELX_line')
        print('total rows', total_rows)
        print('effective_row_number', effective_row_number)
        print('effective_row_number', effective_batch_number)
        print('celx line', celx_line)
        print('feature list', KPI_list)
    return celx_line, timestamp, effective_batch_number, celx_row_df, date

def get_data_MB_reward(state, action, rf, reward_params_dict, KPI_list, thpt_var, state_update_mode, print_mode, burst_prob_user_selector='same_as_PPC', orig_thpt=None):

    # assume state has [C,A,N,E,B]
    if(print_mode):
        print('FUNCTION get_data_MB_reward')
	print('state', state)
	print('action', action)

    # extract thpt from state vector
    # state_dict = extract_state_dict(state, KPI_list)
    # orig_thpt = state_dict[thpt_var]

    # map B to B'
    new_cell_thpt = compute_new_thpt(orig_thpt, action, reward_params_dict, state_update_mode)

    KB_MB_converter = 1000

    # B' = new_thpt: kb/sec, A = action = burst_prob: [0,1] unitless, T	 = control_interval = 1 min = 60 seconds
    # data_MB = B' * A * T
    burst_prob = action
    control_interval_seconds = reward_params_dict['control_interval_seconds']
    PPC_scheduled_data_MB = float(new_cell_thpt * burst_prob * control_interval_seconds)/float(KB_MB_converter)

    # TODO: add noise to these based off the RF quantiles - add in risk criteria
    alpha = reward_params_dict['alpha']
    beta = reward_params_dict['beta']

    if(burst_prob_user_selector == 'same_as_PPC'):
	    burst_prob_user = burst_prob
    else:
	    burst_prob_user = reward_params_dict['avg_user_burst_prob']

    user_original_data_MB = (orig_thpt * burst_prob_user * control_interval_seconds)/float(KB_MB_converter)
    user_new_data_MB = (new_cell_thpt * burst_prob_user * control_interval_seconds)/float(KB_MB_converter)
    user_lost_data_MB = user_original_data_MB - user_new_data_MB

    reward_scale = 1
    reward = alpha * PPC_scheduled_data_MB - beta * user_lost_data_MB

    reward = reward/reward_scale
    if(print_mode):
	print('cell_thpt', orig_thpt)
	print('new_cell_thpt', new_cell_thpt)
	print('burst_prob_PPC', burst_prob)
	print('burst_prob_user', burst_prob_user)
	print('PPC_scheduled_data_MB', PPC_scheduled_data_MB)
	print('data_MB lost user', user_lost_data_MB)
	print('reward', reward)

    return reward[0], state, action, PPC_scheduled_data_MB, user_lost_data_MB


# calculate thpt reward - map the state to a predicted thpt using an RF
def get_reward(state, action, next_state, rf, reward_params_dict, print_mode):
    if(print_mode):
        print('FUNCTION get_reward')
	print('state', state)
	print('action', action)
	print('next_state', next_state)
	#print('cell_thpt', cell_thpt)
	#print('new_cell_thpt', new_cell_thpt)
	#print('reward', reward)

    # current B
    cell_thpt = rf.predict(state)

    # new B = B'
    new_cell_thpt = rf.predict(next_state)

    # TODO: add noise to these based off the RF quantiles - add in risk criteria
    alpha = reward_params_dict['alpha']
    beta = reward_params_dict['beta']

    reward = alpha*new_cell_thpt - beta * (cell_thpt - new_cell_thpt)
    # or a hugely negative reward if B - B' is negative
#    if(new_cell_thpt >= cell_thpt):
#	    reward = alpha*new_cell_thpt 
#    else:
#	    reward = large_negative_value

    # state, action, next_state, reward: add these to your output dataframe
    return reward, state, action, next_state

def load_transition_matrix(trans_matrix_fname):
    trans_matrix_name = 'transition_matrix'
    state_tuple_name = 'states'

    with open(trans_matrix_fname) as fp:
	trans_matrix_json = json.load(fp)
	transition_matrix = np.matrix(trans_matrix_json[trans_matrix_name])
	state_tuple_map = trans_matrix_json[state_tuple_name]

    return transition_matrix, state_tuple_map

def simulate_transition(transition_matrix, init_state, KPI_list, state_tuple_map, print_mode):
    # T[i,]

    state_number, state_tuple_string, state_min_boundary, state_max_boundary = state_to_number(init_state, KPI_list, state_tuple_map, print_mode)
    print('check state', state_number)

    specific_row = transition_matrix[state_number,]
    # randomly sample from this transition matrix
    transition_vector = np.nan_to_num(specific_row)

    # states go from 0 to N-1
    num_states = transition_matrix.shape[0]
    state_vectors = range(num_states)
    
#    state_vectors = []
#    for i, p in enumerate(transition_vector):
#	    print i,p
#	    state_vectors.append(i)

    p_vec = np.array(transition_vector)[0]
    if(print_mode):
        print('FUNCTION simulate_transition')
	print('init_state', init_state)
	print('state_number', state_number)
	print('transition vector', transition_vector[0][0])
	print('p_vector', p_vec) 
	print('transition_sum should be 1', np.nansum(transition_vector))
	print('state_vectors', state_vectors) 
	print('min state', np.min(state_vectors)) 
	print('max state', np.max(state_vectors)) 
	print('num states', len(state_vectors)) 

    # have a dictionary that maps from state number to state_tuple
    # ex: 5 = (0,1,3,2)
    
    final_state_number = np.random.choice(state_vectors, 1, p=p_vec)[0]	

    min_final_state, max_final_state, min_state_df, max_state_df = number_to_state(final_state_number, state_tuple_map, KPI_list, print_mode) 

    if(print_mode):
	print('final_state_number', final_state_number)
	print('min_final_state', min_final_state)
	print('max_final_state', max_final_state)

    return min_final_state

