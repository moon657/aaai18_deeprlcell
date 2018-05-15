# make sure RL_ROOT_DIR is defined to be top of repo
RL_WORKING_DIR=~/Dev/uhana/cleanup_RL_code/results

REWARD_HISTORY_FILE=/tmp/for_noah/data/REWARD_HISTORY.RL_time_variant_agent_DDPG_hard_thpt_limit_0.1_history_5_activity_mult_1_premature_abort_False_alpha_1_beta_1_kappa_1_dynamics_add_to_next_state_env_time-variant.csv

# location of experiments results 
EXPERIMENTS_DIR=/tmp/subset_RF_cell_sim

# default, compare_experiments_by_params, compare_experiments_by_number
EXPERIMENT_MODE=compare_experiments_by_params

RESULTS_DIR=$RL_WORKING_DIR/plotting_results
rm -rf $RESULT_DIR
mkdir -p $RESULTS_DIR

EXPERIMENT_ONE_NUM=235
EXPERIMENT_TWO_NUM=232

# params = alpha,beta,kappa,M,K
EXPERIMENT_ONE_PARAMS=5,1,1,3,8608.66278092
EXPERIMENT_TWO_PARAMS=1,1,1,3,6456.49708569

# seaborn or default
DISPLAY_MODE=seaborn
FEATURES_LIST=$RL_ROOT_DIR/conf/features_list.txt

python $RL_ROOT_DIR/simulate_env/cell_sim_utils/plot_reward_history.py --input_csv $REWARD_HISTORY_FILE --results_dir $RESULTS_DIR --experiments_dir $EXPERIMENTS_DIR --mode $EXPERIMENT_MODE --experiment_one_number $EXPERIMENT_ONE_NUM --experiment_two_number $EXPERIMENT_TWO_NUM --experiment_one_params $EXPERIMENT_ONE_PARAMS --experiment_two_params $EXPERIMENT_TWO_PARAMS --features_list $FEATURES_LIST --display_mode $DISPLAY_MODE
