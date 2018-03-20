from __future__ import print_function
from __future__ import division
import sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import itertools
import argparse
import ConfigParser
import pandas
import collections
from collections import OrderedDict
import operator
import h5py
import joblib
import time

# plotting utils
AAAI_ROOT_DIR = os.environ['AAAI_ROOT_DIR']
util_dir = AAAI_ROOT_DIR + '/utils/'
sys.path.append(util_dir)
from textfile_utils import list_from_textfile, remove_and_create_dir

# run experiments in parallel
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

def continuous_state_to_discrete(cont_state = None, cont_state_to_discrete_dict = None):
    pass

"""
maps a real such as 7.6 into closest_bin_index = 0, closest_y  = 7.5
"""
def get_discrete_state_from_cont_query(query_y = None, discretized_y = None, print_mode = False):
    closest_bin_index = np.argmin(np.abs(query_y - discretized_y))
    closest_y = discretized_y[closest_bin_index]

    if print_mode:
        print('query ', query_y, ' closest ', closest_y, ' closest state ', closest_bin_index)
        print(' ')
    return closest_bin_index, closest_y

"""
dictionary to convert between cont y and discrete states
"""
def discrete_state_cont_y_conversion_dicts(discretized_y = None, print_mode = False):
    state_to_y = collections.OrderedDict()
    y_to_state = collections.OrderedDict()

    for bin_index, discrete_y in enumerate(discretized_y):
        y_to_state[discrete_y] = bin_index
        state_to_y[bin_index] = discrete_y

    if print_mode:
        print('state_to_y : ', state_to_y)
        print(' ')
        print('y_to_state : ', y_to_state)
        print(' ')

    return state_to_y, y_to_state
