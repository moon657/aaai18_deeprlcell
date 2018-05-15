# UHANA CONFIDENTIAL
# __________________
#
# Copyright (C) Uhana, Incorporated
# [2015] - [2016] Uhana Incorporated
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Uhana Incorporated and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Uhana Incorporated
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or Reproduction of this material
# or Unauthorized copying of this file, via any medium is strictly
# prohibited and strictly forbidden.

import json
import numpy as np
import argparse
import sys,os
import pandas
from datetime import timedelta, date

# load utils from the transition matrix
RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + '/utils/'
sys.path.append(util_dir)
print sys.path

from time_conversion_utils import * 
from textfile_utils import list_from_file,list_from_textfile
from panel_plot_timeseries import *

def check_trans_matrix_validity(T):
	num_states = T.shape[0]
	state_vectors = range(num_states)

	# look thru all the rows and find one that is not all NAs
	non_NA_rows = []
	matrix_valid = True

	tol = .01
	one_value = 1.0 - tol

	for idx in state_vectors:
		# all NA rows are now all 0
		# if sum is zero, then disregard
		num_NA_rows = np.sum(T[idx])

		# a row that is not all NA
		if(num_NA_rows != 0.0):
			non_NA_rows.append(idx)

			# get the sum for transitions
			specific_row = np.array(T[idx,:])
			row_sum = np.nansum(np.array(T[idx,:]))
			if(row_sum <= one_value):
				print('row_sum', row_sum)
				print('idx', idx)
				matrix_valid = False

	return matrix_valid, non_NA_rows 

# converts '11:30' to a datetime object with specified year, month, day
def datetime_to_timeStr(orig_datetime_obj):
        year = orig_datetime_obj.year
        month = orig_datetime_obj.month
        day = orig_datetime_obj.day
        hour = orig_datetime_obj.hour
        minute = orig_datetime_obj.minute

	time_str = str(hour) + ':' + str(minute)
	return time_str

def timeStr_to_datetime(year, month, day, time_str, second, rest):
	hour = int(time_str.split(':')[0])
	minute = int(time_str.split(':')[1])

	datetime_obj = datetime.datetime(year, month, day, hour, minute, second, rest)
	return datetime_obj

def load_transition_matrices(datetime_obj=None, matrix_bins_minutes=None, json_object=None):
	#print('')
	#print('current_time', datetime_obj.__str__())
	
	time_str = datetime_to_timeStr(datetime_obj)	
	#print('time_str', time_str)

	# which 15 min matrix to start from?
	binned_datetime_obj = round_datetime_to_binned_minute(datetime_obj, matrix_bins_minutes)
	#print('rounded_time', binned_datetime_obj.__str__())

	# next 15 min matrix
	next_binned_datetime_obj = binned_datetime_obj + datetime.timedelta(minutes=matrix_bins_minutes) 
	#print('next_rounded_time', next_binned_datetime_obj.__str__())

	# 10:30
	rounded_time_str = datetime_to_timeStr(binned_datetime_obj)	
	#print('SAMPLING_FROM: rounded_time_str', rounded_time_str)
	
	# 10:45
	next_rounded_time_str = datetime_to_timeStr(next_binned_datetime_obj)	
	#print('SAMPLING_FROM: next_rounded_time_str', next_rounded_time_str)

	# name for transition_matrix: 10:30 - 10:45 
	transition_str = rounded_time_str + '-' + next_rounded_time_str
	#print('TRANSITION_STR: transition_str', transition_str)

	# how the transitions look
	# ('SAMPLING_FROM: rounded_time_str', '12:45')
	# ('SAMPLING_FROM: next_rounded_time_str', '13:0')
	# ('TRANSITION_STR: transition_str', '12:45-13:0')

	# transition matrices
	T_curr = np.matrix(json_object[rounded_time_str])
	T_next = np.matrix(json_object[transition_str])
	T = np.nan_to_num(T_curr) + np.nan_to_num(T_next)

	# see if rows sum to 1
	# check validity of T_curr, T_next, T_sum
	matrix_valid, non_NA_rows = check_trans_matrix_validity(T)
	#print('sum matrix validity', matrix_valid, non_NA_rows)
	assert(matrix_valid == True)
	return T, non_NA_rows

def state_num_to_KPI_dict(state_num=None, KPI_list=None, state_value_map_dict=None):
	# add the state to the matrix
	KPI_dict = {}
	continuous_state = []
	for KPI in KPI_list:
		state_value = state_value_map_dict[KPI][state_num]
		KPI_dict[KPI] = state_value
		continuous_state.append(state_value)

	return KPI_dict, np.array(continuous_state)


def get_controlled_new_state(KPI_dict = None, action=None, KPI_list=None, activity_multiplier=10, thpt_var=None, max_thpt_value=None):

	orig_coll_metric = KPI_dict['CELLT_AGG_COLL_PER_TTI_DL'] 
	new_coll_metric = (orig_coll_metric + activity_multiplier*action)

	# print('action ', action, 'orig_coll_metric ', orig_coll_metric, 'new_coll_metric ', new_coll_metric)

	# now update the 
	KPI_dict['CELLT_AGG_COLL_PER_TTI_DL'] = new_coll_metric

	KPI_dict[thpt_var] = float(max_thpt_value)/float(new_coll_metric)

	# print(thpt_var, KPI_dict[thpt_var])

	thpt = KPI_dict[thpt_var]
	continuous_state = []
	for KPI in KPI_list:
		continuous_state.append(KPI_dict[KPI])

	final_state = np.array(continuous_state)
	#print('continuous state', continuous_state)
	# print('KPI dict', KPI_dict)
	return final_state, thpt, KPI_dict

def impulse_response_transition(state=None, action=None, noise_params=None, KPI_list=None, state_value_map_dict=None, datetime_obj=None, state_history_df=None, activity_factor_multiplier=5):
	# convert state to KPI_df

	state=state[0]

	KPI_dict = state_num_to_KPI_dict(state_num=state, KPI_list=KPI_list, state_value_map_dict=state_value_map_dict)

	orig_coll_metric = KPI_dict['CELLT_AGG_COLL_PER_TTI_DL'][0] 
	new_coll_metric = orig_coll_metric + activity_factor_multiplier*action
	print('action ', action, 'orig_coll_metric ', orig_coll_metric, 'new_coll_metric ', new_coll_metric)

	curr_time_str = datetime_obj.__str__()
	# now update the 
	KPI_dict['CELLT_AGG_COLL_PER_TTI_DL'] = new_coll_metric
	KPI_dict['DATETIME'] = [curr_time_str]
	KPI_dict['ACTION'] = [action]

	KPI_df = pandas.DataFrame(KPI_dict)
	state_history_df = state_history_df.append(KPI_df)
	return KPI_dict, KPI_df, state_history_df

def get_new_state(state=None, action=None, trans_matrix_info_dict=None):
	T = trans_matrix_info_dict['T']
	state_vectors = trans_matrix_info_dict['state_vectors']

	## get the sum for transitions
	specific_row = np.array(T[state,:])
	row_sum = np.nansum(np.array(T[state,:]))
	# sample a transition from the row of interest
	transition_vector = np.nan_to_num(specific_row)[0]

	#print('transition_vector', transition_vector)
	# list non_empty_states
	non_empty_states = list(np.where(transition_vector > 0)[0])
	#print('non empty states', non_empty_states)
	# rank non-empty states

	p_non_empty = transition_vector[non_empty_states]
	#print('p non_empty', p_non_empty)

	# assume they are ranked already

	# select new state based on the 'a' value according to 2 sample modes
	n = len(non_empty_states)
	probs = np.linspace(0,1, n)
	closest_activity_index = np.argmin(abs(probs - action))
	final_state = non_empty_states[closest_activity_index]

	#print('probs ', probs)
	#print('a ', action, ' closest ', probs[closest_activity_index], ' indx ', closest_activity_index)
	#print('states ', non_empty_states, 'selected ', final_state)
	#print(' ')

	# change based on action
	# final_state = np.random.choice(state_vectors, 1, p=transition_vector)
	return final_state, transition_vector


def get_specific_state(T=None, non_NA_rows=None, sampling_procedure=None, bin_resolution=10, KPI_list = None, state_value_map_dict = None):
	trans_matrix_info_dict = {}
	# sample from this
	num_states = T.shape[0]
	state_vectors = range(num_states)
	
	trans_matrix_info_dict['num_states'] = num_states
	trans_matrix_info_dict['state_vectors'] = state_vectors
	trans_matrix_info_dict['T'] = T

	# bin all states into groups of 10
	# [0, 10, 20, 30, 40, 50, 60, 70]
	bins = [bin_resolution*x for x in range((num_states/bin_resolution)+2) ]
	
	# how hist looks, should sum to 1 but does not
	# array([ 0.        ,  0.00333333,  0.01333333,  0.02      ,  0.02333333,
	#	        0.02666667,  0.01333333])

	unnorm_hist,bin_edges = np.histogram(non_NA_rows, bins=bins)
	# bin_edges: array([ 0, 10, 20, 30, 40, 50, 60, 70])

	hist = np.array([float(x)/len(non_NA_rows) for x in unnorm_hist])

	tol = .001
	one_min = 1-tol
	one_max = 1+tol
	assert( (sum(hist) >= one_min) and (sum(hist) <= one_max))

	# in what state bounds do most of the non-NA rows lie??
	max_value = bin_edges[hist.argmax() + 1]
	min_value = bin_edges[hist.argmax()]
	biased_rows = [x for x in non_NA_rows if ( (x >= min_value) and (x<= max_value))]
	# a subset of non_NA_rows: [50, 52, 53, 54, 55, 56, 57, 58, 60]
	# these reflect the congestion state better
	#print(biased_rows)

	# select a representative row for the matrix
	if(sampling_procedure == 'random'):
		row_index = np.random.choice(non_NA_rows, 1)
		print('row_index', row_index)
	elif(sampling_procedure == 'prob'):
		row_index = np.random.choice(biased_rows, 1)
		#print('row_index', row_index)
	elif(sampling_procedure == 'least_congested'):
		least_congested_state = sorted(biased_rows)[0]
		row_index = least_congested_state
		#print('row_index', row_index)
	else:
		row_index = np.random.choice(non_NA_rows, 1)
		#print('row_index', row_index)

	# get a specific row to begin in
	state_num = int(row_index)

	# map state_num -> state
	state_KPI_dict, state = state_num_to_KPI_dict(state_num = state_num, KPI_list= KPI_list, state_value_map_dict = state_value_map_dict)

	return state_num, trans_matrix_info_dict, state_KPI_dict, state

def write_results(datetime_obj=None, state_history_df=None, action=None, KPI_dict=None):

	# add the state to the matrix
	curr_time_str = datetime_obj.__str__()
	KPI_dict['DATETIME'] = [curr_time_str]
	KPI_dict['ACTION'] = [action]

	KPI_df = pandas.DataFrame(KPI_dict)
	state_history_df = state_history_df.append(KPI_df)
	return state_history_df, KPI_df


def write_history_df(datetime_obj=None, state_value_map_dict=None, final_state=None, state_history_df=None, KPI_list=None, action=None):

	# add the state to the matrix
	KPI_dict = {}
	curr_time_str = datetime_obj.__str__()
	KPI_dict['DATETIME'] = [curr_time_str]
	KPI_dict['ACTION'] = [action]

	for KPI in KPI_list:
		KPI_dict[KPI] = [state_value_map_dict[KPI][final_state]]

	#print(KPI_dict)

	KPI_df = pandas.DataFrame(KPI_dict)
	state_history_df = state_history_df.append(KPI_df)

	return state_history_df, KPI_df
