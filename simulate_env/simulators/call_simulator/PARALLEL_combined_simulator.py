""" Run several experiments in parallel for various parameters 
    of the combined cell simulator

    This simulator has a delayed reward mode
    
"""
import sys, os
import numpy as np
from os import path
import pandas
import argparse

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

from evaluate_DDPG_utils import load_DDPG_dict, wrapper_run_DDPG_experiment
from textfile_utils import list_from_textfile, remove_and_create_dir
from helper_utils_cell_simulator import adjust_timestamp_CELX


def parse_args():
    parser = argparse.ArgumentParser(description='call time variant cell simulator')
    # two columns, signals to plot at end of run
    parser.add_argument(
        '--KPI_file',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161227/RL_time_variant/KPI.txt'
    )

    # where to place results
    parser.add_argument(
        '--base_results_dir',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161227/RL_time_variant/'
    )


    # pkl file mapping state to throughput B
    parser.add_argument(
        '--random_forest_model',
        type=str,
        required=True
    )


    # pkl file mapping state to throughput B
    parser.add_argument(
        '--random_forest_feature_list',
        type=str,
        required=True
    )

    # how many days?
    parser.add_argument(
        '--TOTAL_EPISODES',
        type=int,
        required=False,
        default=20
    )

    # how often to test
    parser.add_argument(
        '--TEST_TERM',
        type=int,
        required=False,
        default=1
    )


    # how often to save neural net
    parser.add_argument(
        '--SAVE_TERM',
        type=int,
        required=False,
        default=1
    )

    # num days to average over when calculating std of reward
    parser.add_argument(
        '--TEST_EPISODES',
        type=int,
        required=False,
        default=5
    )


    # num days to average over when calculating std of reward
    parser.add_argument(
        '--experiment_run_mode',
        type=str,
        required=False,
        default='SINGLE'
    )

    # do we stop if a single experiment fails
    parser.add_argument(
        '--try_catch_mode',
        type=str,
        required=False,
        default='TRUE'
    )

    # cell timeseries file
    parser.add_argument(
        '--timeseries_dir',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161128/RL_unit_test/output/master_celx_rec/'
    )


    # cell timeseries file
    parser.add_argument(
        '--cell_id',
        type=str,
        required=False,
        default='136046093'
    )


    # cell timeseries file
    parser.add_argument(
        '--MAST_CELX_fname',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161128/RL_unit_test/output/master_celx_rec/MAST.CELX.136046093.160928.csv'
    )


    parser.add_argument(
        '--train_test_day_file',
        type=str
    )

    parser.add_argument(
        '--date_var',
        type=str
    )


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # where plots go
    base_results_dir = args.base_results_dir
    # CELX file: timeseries
    MAST_CELX_fname = args.MAST_CELX_fname
    # how to identify experiment
    experiment_prefix = 'TELSTRA'
    print_mode = False
    # KPI (feature) of interest
    KPI_to_plot_file = args.KPI_file
    # RF model to map CANE to B
    random_forest_model = args.random_forest_model
    # features for RF model [HAVE TO BE IN ORDER THAT WE DID TRAINING!]
    random_forest_feature_list = args.random_forest_feature_list
    # RF features
    random_forest_features = list_from_textfile(random_forest_feature_list)

    # how many days
    TOTAL_EPISODES = args.TOTAL_EPISODES
    # how often do we test
    TEST_TERM = args.TEST_TERM
    # how long to test for
    TEST_EPISODES = args.TEST_EPISODES
    # try catch mode
    TRY_CATCH_MODE = args.try_catch_mode
    # how often to save net
    SAVE_TERM = args.SAVE_TERM
    # how often to save net
    timeseries_dir = args.timeseries_dir
    # cell id
    cell_id = args.cell_id
    # experiment run
    experiment_run_mode = args.experiment_run_mode

    # for training and testing in different modes
    train_test_day_file = args.train_test_day_file
    date_var = args.date_var

    # only use some days from train/some for test
    train_df = pandas.read_csv(train_test_day_file)
    train_days = list(set(train_df[train_df['TRAIN_TEST_INDICATOR'] == 'TRAIN'][date_var]))
    test_days = list(set(train_df[train_df['TRAIN_TEST_INDICATOR'] == 'TEST'][date_var]))

    print('train ', train_days) 
    print('test ', test_days) 

    random_forest_data = {'random_forest_model': random_forest_model, 'RF_feature_list': random_forest_features}

    # state features = collision metric for now
    # KPI_list = ['CELLT_AGG_COLL_PER_TTI_DL']

    KPI_list = random_forest_features

    thpt_var = 'CELLT_AGG_THP_DL'

    # make DATETIME parseable
    CELX_df = adjust_timestamp_CELX(MAST_CELX_fname = MAST_CELX_fname)

    # get a quantile of thpt for the hard_thpt_limit
    median_cell_thpt = CELX_df[thpt_var].quantile(.50)

    high_cell_thpt = CELX_df[thpt_var].quantile(.98)
    
    print('median thpt ', median_cell_thpt)
    print('high thpt ', high_cell_thpt)

    # sweep through all experiment parameters
    ##########################################
    hard_thpt_limit_flag_space = [True, False]

    premature_abort_space = [False]

    history_space = [15]

    #kappa_space = [1, 2, 5, 10]
    kappa_space = [1]

    #alpha_space = [1, 2, 5, 10]
    alpha_space = [1]

    #beta_space = [1, 2, 5, 10]
    beta_space = [1]

    #activity_factor_multiplier_space = [1,2,3,5]
    activity_factor_multiplier_space = [5,7]

    #hard_thpt_limit_space = [.8, 1.0, 1.5, 2.0, 2.5, 5]
    hard_thpt_limit_space = [.8, 1.0]

    clip_action_explore_list = ['TRUE', 'FALSE']
    OU_NOISE_VEC = ['0.15-0.20', '0.05-0.05', '0.01-0.01']
    
    clip_action_explore = 'TRUE'
    OU_NOISE_STR = '0.15-0.20'
    
    delayed_reward_mode = True
    delay_reward_interval = 15

    delay_list = [1,2,5,15]

    # modulation factors for the reward, see writeup
    alpha_beta_kappa_combos = []
    for alpha in alpha_space:
        min_beta = min(beta_space)
        min_kappa = min(kappa_space)
        params_dict = {'alpha': alpha, 'beta': min_beta, 'kappa': min_kappa}
        alpha_beta_kappa_combos.append(params_dict)

    for beta in beta_space:
        min_alpha = min(alpha_space)
        min_kappa = min(kappa_space)
        params_dict = {'alpha': min_alpha, 'beta': beta, 'kappa': min_kappa}
        alpha_beta_kappa_combos.append(params_dict)

    for kappa in kappa_space:
        min_alpha = min(alpha_space)
        min_beta = min(beta_space)
        params_dict = {'alpha': min_alpha, 'beta': min_beta, 'kappa': kappa}
        alpha_beta_kappa_combos.append(params_dict)


    # just for test
    params_dict = {'alpha': 1, 'beta': 1, 'kappa': 1}
    alpha_beta_kappa_combos = [params_dict]

    # loop over cell_id, delayed reward intervals also
    # list of dictionaries with parameters per experiment
    experiment_num = 0
    experiment_params_list = []
    for hard_thpt_limit_flag in hard_thpt_limit_flag_space:
            for activity_factor_multiplier in activity_factor_multiplier_space:
                for hard_thpt_limit in hard_thpt_limit_space:
                    for alpha_beta_kappa_dict in alpha_beta_kappa_combos:
                        for delay_reward_interval in delay_list:

                            experiment_settings = {}
                            experiment_settings['hard_thpt_limit_flag'] = hard_thpt_limit_flag
                            experiment_settings['activity_factor_multiplier'] = activity_factor_multiplier
                            experiment_settings['hard_thpt_limit'] = hard_thpt_limit * median_cell_thpt
                            experiment_settings['alpha_beta_kappa_dict'] = alpha_beta_kappa_dict
                            experiment_settings['history_minutes'] = history_space[0]
                            experiment_settings['experiment_prefix'] = experiment_prefix
                            experiment_settings['TOTAL_EPISODES'] = TOTAL_EPISODES
                            experiment_settings['TEST_TERM'] = TEST_TERM
                            experiment_settings['TEST_EPISODES'] = TEST_EPISODES
                            experiment_settings['OU_NOISE'] = OU_NOISE_STR
                            experiment_settings['clip_action_explore'] = clip_action_explore
                            experiment_settings['timeseries_dir'] = timeseries_dir
                            experiment_settings['train_days'] = train_days
                            experiment_settings['test_days'] = test_days
                            experiment_settings['cell_id'] = cell_id
                            experiment_settings['experiment_num'] = experiment_num

                            # differentiate what type of simulation to do
                            experiment_settings['simulator_mode'] = 'RF_mode'
                            experiment_settings['delayed_reward_mode'] = delayed_reward_mode
                            experiment_settings['DELAY_REWARD_INTERVAL'] = delay_reward_interval
                            experiment_settings['KB_TO_MB'] = 1000

                            experiment_num += 1
                            experiment_params_list.append(experiment_settings)

    # DDPG learning algorithm info
    #####################
    # each episode is a day

    # number of rows = length of day
    EPISODE_LENGTH= CELX_df.shape[0]
    WARMUP = int(.1*TOTAL_EPISODES*EPISODE_LENGTH)

    DDPG_dict = load_DDPG_dict(
        MAX_STEP=EPISODE_LENGTH,
        EPISODES=TOTAL_EPISODES,
        TEST_EPISODES=TEST_EPISODES,
        TEST_TERM=TEST_TERM,
        PRINT_LEN=40,
        SAVE_TERM=SAVE_TERM)
    DDPG_dict['WARMUP'] = WARMUP
    random_reset_during_train = False
    DDPG_dict['random_reset_during_train'] = random_reset_during_train
    DDPG_dict['OU_NOISE'] = [.15, .20] 

    # info for plotting
    #####################
    plotting_info_dict = {}
    plotting_info_dict['KPI_to_plot_file'] = KPI_to_plot_file
    plotting_info_dict['cell_id'] = cell_id
    plotting_info_dict['time_var'] = 'ITERATION_INDEX'

    env_type = 'combined'

    # run a single experiment for a test
    if experiment_run_mode == 'SINGLE':
        # settings for a unit-test that works well
        experiment_settings = {}
        experiment_settings['hard_thpt_limit_flag'] = True
        experiment_settings['activity_factor_multiplier'] = 5
        experiment_settings['hard_thpt_limit'] = 1.0*median_cell_thpt
        alpha = 1
        beta = 2
        kappa = 1
        experiment_settings['alpha_beta_kappa_dict'] = {'alpha': alpha, 'beta': beta, 'kappa': kappa}
        experiment_settings['history_minutes'] = 7
        experiment_settings['experiment_prefix'] = experiment_prefix
        experiment_settings['TOTAL_EPISODES'] = TOTAL_EPISODES
        experiment_settings['TEST_TERM'] = TEST_TERM
        experiment_settings['OU_NOISE'] = OU_NOISE_STR
        experiment_settings['clip_action_explore'] = clip_action_explore
        experiment_settings['timeseries_dir'] = timeseries_dir
        experiment_settings['train_days'] = train_days
        experiment_settings['test_days'] = test_days
        experiment_settings['cell_id'] = cell_id
        experiment_settings['experiment_num'] = 1

        # differentiate what type of simulation to do
        experiment_settings['simulator_mode'] = 'RF_mode'
        experiment_settings['delayed_reward_mode'] = delayed_reward_mode
        experiment_settings['DELAY_REWARD_INTERVAL'] = 15
        experiment_settings['KB_TO_MB'] = 1000

        print('RUN TEST MODE')
        wrapper_run_DDPG_experiment(experiment_settings = experiment_settings, DDPG_dict = DDPG_dict, plotting_info_dict = plotting_info_dict, base_results_dir = base_results_dir, MAST_CELX_fname = MAST_CELX_fname, KPI_list = KPI_list, CELX_df = CELX_df, print_mode = print_mode, try_catch_mode = TRY_CATCH_MODE, env_type = env_type, random_forest_data = random_forest_data)

    # run all experiments across cores
    elif experiment_run_mode == 'PARALLEL':
        d = Parallel(n_jobs=num_cores)(delayed(wrapper_run_DDPG_experiment)(experiment_settings = p, DDPG_dict = DDPG_dict, plotting_info_dict = plotting_info_dict, base_results_dir = base_results_dir, MAST_CELX_fname = MAST_CELX_fname, KPI_list = KPI_list, CELX_df = CELX_df, print_mode = print_mode, try_catch_mode = TRY_CATCH_MODE, env_type = env_type, random_forest_data = random_forest_data) for p in experiment_params_list)

    elif experiment_run_mode == 'SEQUENTIAL':
        for p in experiment_params_list:
            wrapper_run_DDPG_experiment(experiment_settings = p, DDPG_dict = DDPG_dict, plotting_info_dict = plotting_info_dict, base_results_dir = base_results_dir, MAST_CELX_fname = MAST_CELX_fname, KPI_list = KPI_list, CELX_df = CELX_df, print_mode = print_mode, try_catch_mode = TRY_CATCH_MODE, env_type = env_type, random_forest_data = random_forest_data)


