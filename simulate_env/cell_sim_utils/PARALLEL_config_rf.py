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
RL_ROOT_DIR = os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + '/utils/'
sys.path.append(util_dir)

# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

# simulators
simulators_dir = RL_ROOT_DIR + '/simulate_env/simulators/'
sys.path.append(simulators_dir)

# generate list of parameters
post_process_dir = RL_ROOT_DIR + '/post_process_experiments/'
sys.path.append(post_process_dir)

from generate_experiment_params import conf_get_best_experiments_to_replicate, conf_get_single_experiment_setting
from conf_evaluate_agent import conf_preset_experiment_wrapper
from load_configs import resolve_config_paths
from textfile_utils import remove_and_create_dir

def parse_args():
    parser = argparse.ArgumentParser(description='call time variant cell simulator')
    conf_dir = RL_ROOT_DIR + '/IJCAI_exp/MDDPG/example_parallel_MDDPG/'

    # where input/out dirs are - differs per user
    parser.add_argument(
        '--file_path_config_file',
        type=str,
        required=False,
        default = conf_dir + 'csandeep_paths.ini'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
  
    # file paths for input output dirs
    # user specific paths
    file_path_config = ConfigParser.ConfigParser()    
    file_path_config.read(args.file_path_config_file)
    print file_path_config.sections()

    # SINGLE or PARALLEL
    experiment_run_mode = file_path_config.get('EXPERIMENT_INFO', 'experiment_run_mode')
    # WHERE ARE BASE RESULTS
    experiment_list = range(file_path_config.getint('EXPERIMENT_INFO', 'max_experiment') + 1)

    ## run a single experiment for a test
    if experiment_run_mode == 'SINGLE':
        print('RUN TEST MODE')

        experiment_num = experiment_list[0]
        conf_preset_experiment_wrapper(file_path_config = file_path_config, experiment_num = experiment_num)

    ## run all experiments across cores
    elif experiment_run_mode == 'PARALLEL':
        d = Parallel(n_jobs=num_cores)(delayed(conf_preset_experiment_wrapper)(file_path_config = file_path_config, experiment_num = p) for p in experiment_list)
    else:
        pass

