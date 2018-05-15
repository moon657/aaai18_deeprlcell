import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import cPickle
import json
import pandas
import sys,os
import random

RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + '/utils/'
sys.path.append(util_dir)

# for DDPG algorithm
actor_critic_utils = RL_ROOT_DIR + '/simulate_env/agents/DDPG/'
sys.path.append(actor_critic_utils)

# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

# cell simulator helpers
simulators_dir = RL_ROOT_DIR + '/simulate_env/simulators/'
sys.path.append(simulators_dir)

print(sys.path)
print(RL_ROOT_DIR)

from helper_utils_cell_simulator import load_reward_params_dict, load_burst_prob_params_dict
from time_variant_simulator_simplified import TimeVariantSimulatorEnv
from random_forest_simulator import RandomForestTimeVariantSimulatorEnv
from combined_simulator import CombinedTimeVariantSimulatorEnv
from algorithm import DDPG
import train_util as tu
from double_panel_plot_timeseries import no_datetime_overlay_KPI_plot
from textfile_utils import remove_and_create_dir

def simple_evaluate_DDPG(env = None, DDPG_dict=None, plotting_info_dict=None, agent_type=None):
    algo = DDPG(env.observation_space, env.action_space, WARMUP=DDPG_dict['WARMUP'], OU_NOISE=DDPG_dict['OU_NOISE'], clip_action_explore = DDPG_dict['clip_action_explore'])
    
    # get the action mode correct
    DDPG_rewards, DDPG_wt = tu.train(algo, env, MAX_STEP=DDPG_dict['MAX_STEP'], EPISODES=DDPG_dict['EPISODES'], TEST_EPISODES=DDPG_dict['TEST_EPISODES'], TEST_TERM=DDPG_dict['TEST_TERM'], SAVE_TERM=DDPG_dict['SAVE_TERM'], exp_mode='ou_noise', env_name='cell', random_reset_during_train = DDPG_dict['random_reset_during_train'], save_path=DDPG_dict['save_neural_net_path'])
    
    experiment_params = plotting_info_dict['experiment_params']
    KPI_to_plot_file = plotting_info_dict['KPI_to_plot_file']
    base_results_dir = plotting_info_dict['base_results_dir']
    cell_id = plotting_info_dict['cell_id'] 
    time_var = plotting_info_dict['time_var']
    
    rmeans = DDPG_rewards[:1][0]
    rstds = DDPG_rewards[1:][0]
    train_reward_plot = base_results_dir + '/' + experiment_params + '.batchRewards.png'
    tu.plot_train_rewards(rmeans, rstds, fig_path=train_reward_plot, xlabel='Training episodes', ylabel='Total rewards', x=None)
    
    DDPG_rewards_df = pandas.DataFrame({'mean_reward': list(DDPG_rewards[0]), 'std_reward': list(DDPG_rewards[1])})
    experiment_reward_results_csv = base_results_dir + '/' + experiment_params + '.batchRewards.csv'
    DDPG_rewards_df.to_csv(experiment_reward_results_csv, index=False)
    
    batch_var = 'BATCH_NUM'
    reward_history_df = env.reward_history_df
    reward_history_df.set_index(time_var, inplace=True)
   
    # write the reward history dataframe for later analysis
    #####################
    reward_history_csv = base_results_dir + '/REWARD_HISTORY.' + experiment_params + '.csv'
    reward_history_df.to_csv(reward_history_csv, index=False)

    # get data for unconverged/converged neural net performance
    #####################
    early_batch_index = reward_history_df[batch_var].min()
    late_batch_index = reward_history_df[batch_var].max()
    
    early_batch_df = reward_history_df[reward_history_df[batch_var] == early_batch_index]
    late_batch_df = reward_history_df[reward_history_df[batch_var] == late_batch_index]

    # start the plots
    #####################

    # plot first batch KPIs
    plotting_info_dict['experiment_params'] = experiment_params + '_early'
    no_datetime_overlay_KPI_plot(early_batch_df, plotting_info_dict)

    # plot last batch converged KPIs
    plotting_info_dict['experiment_params'] = experiment_params + '_late'
    no_datetime_overlay_KPI_plot(late_batch_df, plotting_info_dict)
    
    return reward_history_df, DDPG_rewards_df, env, DDPG_rewards, DDPG_wt, algo, late_batch_df, early_batch_df

def load_DDPG_dict(MAX_STEP=None, EPISODES=None,TEST_EPISODES=None, TEST_TERM=None, PRINT_LEN = None, SAVE_TERM=None):

	DDPG_dict = {}
	DDPG_dict['MAX_STEP'] = MAX_STEP
	DDPG_dict['EPISODES'] = EPISODES
	DDPG_dict['TEST_EPISODES'] = TEST_EPISODES
	DDPG_dict['TEST_TERM'] = TEST_TERM
	DDPG_dict['PRINT_LEN'] = PRINT_LEN
	DDPG_dict['SAVE_TERM'] = SAVE_TERM

	return DDPG_dict

def get_state_bounds(history_minutes = None, CELX_df = None, KPI_list = None):

    # get min and max states from data for state space bounds
    ###############################################
    min_state_list = []
    max_state_list = []
    for lag in range(history_minutes):
        for KPI in KPI_list:
            min_KPI = 0.9*CELX_df[KPI].min()
            max_KPI = 1.1*CELX_df[KPI].max()
            print('KPI', KPI)
            print('min_KPI', min_KPI)
            print('max_KPI', max_KPI)
            min_state_list.append(min_KPI)
            max_state_list.append(max_KPI)

    min_state_vector = np.array(min_state_list)
    max_state_vector = np.array(max_state_list)
    print('min_state_vector ', min_state_vector)
    print('max_state_vector ', max_state_vector)

    return min_state_vector, max_state_vector

def wrapper_run_DDPG_experiment(experiment_settings = None, DDPG_dict = None, plotting_info_dict = None, base_results_dir = None, MAST_CELX_fname = None, KPI_list = None, CELX_df = None, print_mode = None, random_forest_data = None, env_type = 'time-variant', thpt_var = 'THPT', try_catch_mode='TRUE'):

    ###############################################
    # load the burst probs action space
    burst_discretization = .025
    burst_prob_params_dict = load_burst_prob_params_dict(
        burst_discretization=burst_discretization,
        max_burst_prob=1.0,
        min_burst_prob=0.00,
        optimal_burst_prob=.25)
    
    print(experiment_settings)

    # EXPERIMENT PARAMETERS
    ###############################################
    # if (K-B') added to reward
    hard_thpt_limit_flag = experiment_settings['hard_thpt_limit_flag']
    # if we stop simulation early if low thpt
    premature_abort_flag = False
    # past history to include in state
    history_minutes = experiment_settings['history_minutes']
    # hard thpt limit
    hard_thpt_limit = experiment_settings['hard_thpt_limit']
    # activity_factor multiplier
    activity_factor_multiplier = experiment_settings['activity_factor_multiplier']

    # for the reward computation
    alpha_beta_kappa_dict = experiment_settings['alpha_beta_kappa_dict']
    
    experiment_prefix = experiment_settings['experiment_prefix']

    alpha = alpha_beta_kappa_dict['alpha']
    beta = alpha_beta_kappa_dict['beta']
    kappa = alpha_beta_kappa_dict['kappa']

    experiment_settings['alpha'] = alpha
    experiment_settings['beta'] = beta
    experiment_settings['kappa'] = kappa

    OU_NOISE_STR = experiment_settings['OU_NOISE']
    clip_action_explore = experiment_settings['clip_action_explore']
    DDPG_dict['OU_NOISE'] = [float(x) for x in OU_NOISE_STR.split('-')]
    DDPG_dict['clip_action_explore'] = clip_action_explore

    #############################################
    min_state_vector, max_state_vector = get_state_bounds(history_minutes = history_minutes, CELX_df = CELX_df, KPI_list = KPI_list)

    # for the reward computation
    reward_params_dict = load_reward_params_dict(
        alpha= alpha,
        beta= beta,
        k= 1,
        control_interval_seconds=60,
        avg_user_burst_prob=.05)
    # B = B_0/C
    reward_params_dict['B_0'] = 1.0
    # modulate hard thpt penalty: -activity*kappa*(K-B')
    reward_params_dict['kappa'] = kappa
    # value for K in above eqn
    reward_params_dict['hard_thpt_limit'] = hard_thpt_limit

    # to initialize the cell simulator
    env_params_dict = {}
    env_params_dict['print_mode'] = print_mode
    env_params_dict['continuous_action_mode'] = True
    env_params_dict['reward_params_dict'] = reward_params_dict
    env_params_dict['burst_prob_params_dict'] = burst_prob_params_dict
    env_params_dict['min_state_vector'] = min_state_vector
    env_params_dict['max_state_vector'] = max_state_vector
    env_params_dict['thpt_var'] = thpt_var

    # episode is at least 15 steps before termination
    env_params_dict['min_iterations_before_done'] = 50
    # if THPT below bad_thpt_threshold for num_last_entries AND we have surpassed min_iterations, env can abort prematurely
    env_params_dict['num_last_entries'] = 10
    env_params_dict['bad_thpt_threshold'] = .75 * reward_params_dict[
        'hard_thpt_limit']
    env_params_dict['hard_thpt_limit_flag'] = hard_thpt_limit_flag
    env_params_dict['CELX_df'] = CELX_df
    state_propagation_method = 'add_to_next_state'
    env_params_dict['state_propagation_method'] = state_propagation_method
    env_params_dict['noise_mean_std'] = [0, .15]
    env_params_dict['KPI_list'] = KPI_list
    env_params_dict['premature_abort_flag'] = premature_abort_flag
    env_params_dict['history_minutes'] = history_minutes
    env_params_dict['activity_factor_multiplier'] = activity_factor_multiplier
    env_params_dict['experiment_params_dict'] = experiment_settings

    # for splitting training and testing days
    env_params_dict['timeseries_dir'] = experiment_settings['timeseries_dir']
    env_params_dict['train_days'] = experiment_settings['train_days']
    env_params_dict['test_days'] = experiment_settings['test_days']
    env_params_dict['cell_id'] = experiment_settings['cell_id']

    # differentiate what type of simulation to do
    env_params_dict['simulator_mode'] = experiment_settings['simulator_mode']
    env_params_dict['delayed_reward_mode'] = experiment_settings['delayed_reward_mode']
    env_params_dict['DELAY_REWARD_INTERVAL'] = experiment_settings['DELAY_REWARD_INTERVAL']
    env_params_dict['KB_TO_MB'] = experiment_settings['KB_TO_MB']

    # how many episodes for DDPG algorithm
    agent_type = 'DDPG'
    
    # string that defines parameters for experiment to name output directories
    experiment_params = '_'.join([
        experiment_prefix, 'thptLimit',
        str(hard_thpt_limit), 'H', str(history_minutes),
        'M', str(activity_factor_multiplier),
        'a', str(alpha),
        'b', str(beta),
        'k', str(kappa),
        'env', env_type,
        'ou', OU_NOISE_STR,
        'clip', clip_action_explore
    ])

    experiment_params = str(experiment_settings['experiment_num'])

    print('RUNNING EXPERIMENT ', experiment_params)
    experiment_path = base_results_dir + '/' + experiment_params
    print('experiment path', experiment_path)
    remove_and_create_dir(experiment_path)

    NN_path = experiment_path + '/saved_models'
    remove_and_create_dir(NN_path)
    DDPG_dict['save_neural_net_path'] = NN_path

    experiment_results_indicator_fname = experiment_path + '/PARAMS.txt'

    # DDPG function
    ######################################################################
    # we keep going if a single experiment fails
    if try_catch_mode == 'TRUE':
        try:

            # info for plotting
            #####################
            plotting_info_dict['experiment_params'] = env_type
            plotting_info_dict['base_results_dir'] = experiment_path
            
            # run the correct simulator
            if(env_type == 'random_forest'):
                env_params_dict['random_forest_model'] = random_forest_data['random_forest_model']
                env_params_dict['RF_feature_list'] = random_forest_data['RF_feature_list']
                env = RandomForestTimeVariantSimulatorEnv(env_params_dict)
            elif(env_type == 'combined'):
                env_params_dict['random_forest_model'] = random_forest_data['random_forest_model']
                env_params_dict['RF_feature_list'] = random_forest_data['RF_feature_list']
                env = CombinedTimeVariantSimulatorEnv(env_params_dict)
            else:
                env = TimeVariantSimulatorEnv(env_params_dict)
 
            # run the DDPG agent, record results, write plots
            reward_history_df, DDPG_rewards_df, env, DDPG_rewards, DDPG_wt, algo, late_batch_df, early_batch_df = simple_evaluate_DDPG(env = env, DDPG_dict = DDPG_dict, plotting_info_dict= plotting_info_dict, agent_type = agent_type)

            # debug the learned actions
            print('summary of early actions')
            print(early_batch_df['ACTION'].describe())

            print('summary of latest actions')
            print(late_batch_df['ACTION'].describe())

            with open(experiment_results_indicator_fname, 'w') as f:
                success_string = '\t'.join(['SUCCESS','TRUE'])
                f.write(success_string + '\n')
                for k,v in experiment_settings.iteritems():
                    if(k != 'alpha_beta_kappa_dict'):
                        out_string = '\t'.join([k,str(v)])
                        f.write(out_string + '\n')

                # PRINT THE ACTION STD
                out_str = 'ACTION'
                f.write(out_str + '\n')
                out_str = late_batch_df['ACTION'].describe().to_string()
                f.write(out_str + '\n')


        except:
            try:
                with open(experiment_results_indicator_fname, 'w') as f:
                    for k,v in experiment_settings.iteritems():
                        if(k != 'alpha_beta_kappa_dict'):
                            out_string = '\t'.join([k,str(v)])
                            f.write(out_string + '\n')
                    success_string = '\t'.join(['SUCCESS','FALSE'])
                    f.write(success_string + '\n')

            except:
                print('experiment failed, COULD NOT WRITE FILE')
                print('FAILED PARAMS ', experiment_params)
                

    # if we want system to stop if even a single experiment fails
    else:

        # info for plotting
        #####################
        plotting_info_dict['experiment_params'] = env_type
        plotting_info_dict['base_results_dir'] = experiment_path
       
        # run the correct simulator
        if(env_type == 'random_forest'):
            env_params_dict['random_forest_model'] = random_forest_data['random_forest_model']
            env_params_dict['RF_feature_list'] = random_forest_data['RF_feature_list']
            env = RandomForestTimeVariantSimulatorEnv(env_params_dict)
        elif(env_type == 'combined'):
            env_params_dict['random_forest_model'] = random_forest_data['random_forest_model']
            env_params_dict['RF_feature_list'] = random_forest_data['RF_feature_list']
            env = CombinedTimeVariantSimulatorEnv(env_params_dict)
        else:
            env = TimeVariantSimulatorEnv(env_params_dict)

        # run the DDPG agent, record results, write plots
        reward_history_df, DDPG_rewards_df, env, DDPG_rewards, DDPG_wt, algo, late_batch_df, early_batch_df = simple_evaluate_DDPG(env = env, DDPG_dict = DDPG_dict, plotting_info_dict= plotting_info_dict, agent_type = agent_type)

        # debug the learned actions
        print('summary of early actions')
        print(early_batch_df['ACTION'].describe())

        print('summary of latest actions')
        print(late_batch_df['ACTION'].describe())

        with open(experiment_results_indicator_fname, 'w') as f:
            success_string = '\t'.join(['SUCCESS','TRUE'])
            f.write(success_string + '\n')
            for k,v in experiment_settings.iteritems():
                if(k != 'alpha_beta_kappa_dict'):
                    out_string = '\t'.join([k,str(v)])
                    f.write(out_string + '\n')

            # PRINT THE ACTION STD
            out_str = 'ACTION'
            f.write(out_str + '\n')
            out_str = late_batch_df['ACTION'].describe().to_string()
            f.write(out_str + '\n')

