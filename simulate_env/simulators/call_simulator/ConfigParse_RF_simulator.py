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
from conf_evaluate_agent import conf_wrapper_run_experiment
from load_configs import resolve_config_paths
from textfile_utils import remove_and_create_dir

def parse_args():
    parser = argparse.ArgumentParser(description='call time variant cell simulator')

    conf_dir = RL_ROOT_DIR + '/IJCAI_exp/MDDPG/conf/'

    # where input/out dirs are - differs per user
    parser.add_argument(
        '--file_path_config_file',
        type=str,
        required=False,
        default = conf_dir + 'csandeep_paths.ini'
    )

    # can use a default that sets NUM_EPISODES, params for RL agent
    parser.add_argument(
        '--experiment_config_file',
        type=str,
        required=False,
        default = conf_dir + 'base_MDDPG_params.ini'
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
  
    # file paths for input output dirs
    # user specific paths
    file_path_config = ConfigParser.ConfigParser()    
    file_path_config.read(args.file_path_config_file)
    print file_path_config.sections()

    # RL relevant params
    # NO RELATIVE PATHS
    base_experiment_config = ConfigParser.ConfigParser()
    base_experiment_config.read(args.experiment_config_file)    
    print base_experiment_config.sections()

    # programatically remove and create the results dir
    base_results_dir = file_path_config.get('OUTPUT_DIRECTORIES', 'base_results_dir')
    #remove_and_create_dir(base_results_dir)

    # join the two conf files
    joint_config_file = base_results_dir + '/joint.ini'

    experiment_config, file_path_config = resolve_config_paths(experiment_config_file = args.experiment_config_file, file_path_config_file = args.file_path_config_file,  joint_config_file = joint_config_file)

    # update experiment_config
    experiment_config = ConfigParser.ConfigParser()
    experiment_config.read(joint_config_file)    
    
    # do we re-run a set of experiments that run well before?
    # predefined_experiments_mode = experiment_config.getboolean('EXPERIMENT_INFO', 'predefined_experiments_mode') 
    predefined_experiments_file = experiment_config.get('EXPERIMENT_INFO', 'predefined_experiments_file')

    # a list of experiment settings to run in PARALLEL
    experiment_params_list = conf_get_best_experiments_to_replicate(experiment_config = experiment_config)

    # SINGLE or PARALLEL
    experiment_run_mode = experiment_config.get('EXPERIMENT_INFO', 'experiment_run_mode')

    ## run a single experiment for a test
    if experiment_run_mode == 'SINGLE':
        # settings for a unit-test that works well
        experiment_settings = conf_get_single_experiment_setting(experiment_config = experiment_config)
        print('RUN TEST MODE')
        conf_wrapper_run_experiment(experiment_settings = experiment_settings, experiment_config = experiment_config, file_path_config = file_path_config)

    ## run all experiments across cores
    elif experiment_run_mode == 'PARALLEL':
        d = Parallel(n_jobs=num_cores)(delayed(conf_wrapper_run_experiment)(experiment_settings = p, experiment_config = experiment_config, file_path_config = file_path_config) for p in experiment_params_list)
    else:
        pass

