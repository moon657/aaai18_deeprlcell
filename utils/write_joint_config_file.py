

""" Run several experiments in parallel for various parameters 
    of the random forest cell simulator, reward = RF(CANE)
"""
import sys, os
import numpy as np
from os import path
import pandas
import ConfigParser
import argparse

# run experiments in parallel
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

# plotting utils
AAAI_ROOT_DIR = os.environ['AAAI_ROOT_DIR']
util_dir = AAAI_ROOT_DIR + '/utils/'
sys.path.append(util_dir)

from textfile_utils import remove_and_create_dir

def parse_args():
    parser = argparse.ArgumentParser(description='forecasting pipe params')

    # where input/out dirs are - differs per user
    parser.add_argument(
        '--file_path_config_file',
        type=str,
        required=False,
        default = AAAI_ROOT_DIR + '/forecast/file_paths.ini' 
    )

    parser.add_argument(
        '--forecasting_config_file',
        type=str,
        required=False,
        default = AAAI_ROOT_DIR + '/forecast/overall_forecast_pipe_params.ini' 
    )

    parser.add_argument(
        '--joint_config_file',
        type=str,
        required=False,
        default = AAAI_ROOT_DIR + '/forecast/joint_forecast_pipe_params.ini' 
    )

    return parser.parse_args()

def resolve_full_paths(input_fpath_vars = None, forecasting_config = None, input_files_dir = None):

    for input_fpath_var in input_fpath_vars:

        local_path = forecasting_config.get('PATHS', input_fpath_var)

        print('local_path', local_path)

        full_path = input_files_dir + local_path
        print('full_path', full_path)

        forecasting_config.set('PATHS', input_fpath_var, full_path)

    return forecasting_config

if __name__ == "__main__":
    args = parse_args()

    joint_config_file = args.joint_config_file

    # file paths for input output dirs
    file_path_config = ConfigParser.ConfigParser()    
    file_path_config.read(args.file_path_config_file)
    print file_path_config.sections()

    # forecasting config
    forecasting_config = ConfigParser.ConfigParser()    
    forecasting_config.read(args.forecasting_config_file)
    print forecasting_config.sections()

    input_files_dir = file_path_config.get('FILE_PATHS', 'input_files_dir') 
    input_fpath_vars = file_path_config.get('FILE_PATHS', 'input_fpath_vars').split(',')

    output_results_dir = file_path_config.get('FILE_PATHS', 'output_results_dir') 
    output_fpath_vars = file_path_config.get('FILE_PATHS', 'output_fpath_vars').split(',')

    # add input files dir to these fields in forecasting config
    forecasting_config = resolve_full_paths(input_fpath_vars = input_fpath_vars, forecasting_config = forecasting_config, input_files_dir = input_files_dir)

    forecasting_config = resolve_full_paths(input_fpath_vars = output_fpath_vars, forecasting_config = forecasting_config, input_files_dir = output_results_dir)

    cfgfile = open(joint_config_file,'w')
    forecasting_config.write(cfgfile)
    cfgfile.close()

