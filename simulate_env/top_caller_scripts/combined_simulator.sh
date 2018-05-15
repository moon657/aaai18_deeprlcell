# make sure RL_ROOT_DIR is defined to be top of repo

# path in repo to example data
RL_EXAMPLE_DATA_DIR=${RL_ROOT_DIR}/example_data

# where results will go on your machine!
DATE=20170112
RL_WORKING_DIR=/home/csandeep/work/experiments_deeprl_cell/alpha_RF_delayed/${DATE}
rm -rf $RL_WORKING_DIR
mkdir -p ${RL_WORKING_DIR}

DAY=160928
CELL=136046093

# where plots go
base_results_dir=$RL_WORKING_DIR/RF_cell_sim
rm -rf $base_results_dir
mkdir -p $base_results_dir

CODE_DIR=${RL_ROOT_DIR}/simulate_env/simulators/call_simulator

MAST_CELX_fname=$RL_EXAMPLE_DATA_DIR/timeseries/MAST.CELX.$CELL.$DAY.csv
timeseries_dir=$RL_EXAMPLE_DATA_DIR/timeseries/

KPI_file=$RL_EXAMPLE_DATA_DIR/conf/KPI.txt

var_to_predict=CELLT_AGG_THP_DL

random_forest_model=$RL_EXAMPLE_DATA_DIR/example_RF_results/random_forest_model.${CELL}.y_var.${var_to_predict}.pkl

random_forest_feature_list=$RL_EXAMPLE_DATA_DIR/example_RF_results/unranked.final.features.cell.${CELL}.y_var.${var_to_predict}.txt

train_test_day_file=${RL_EXAMPLE_DATA_DIR}/conf/train_test_day.csv
# join original data on this field with train_test_day_file
date_var='DATE_Melbourne'

TOTAL_EPISODES=150
TEST_EPISODES=3
EXPERIMENT_RUN_MODE='PARALLEL'
TRY_CATCH_MODE='FALSE'
TEST_TERM=10
SAVE_TERM=$TEST_TERM

python -i $CODE_DIR/PARALLEL_combined_simulator.py --KPI_file $KPI_file --base_results_dir $base_results_dir --MAST_CELX_fname $MAST_CELX_fname --random_forest_model $random_forest_model --random_forest_feature_list $random_forest_feature_list --TOTAL_EPISODES $TOTAL_EPISODES --TEST_EPISODES $TEST_EPISODES --experiment_run_mode $EXPERIMENT_RUN_MODE --try_catch_mode $TRY_CATCH_MODE --TEST_TERM $TEST_TERM --SAVE_TERM $SAVE_TERM --train_test_day_file $train_test_day_file --date_var $date_var --timeseries_dir $timeseries_dir

