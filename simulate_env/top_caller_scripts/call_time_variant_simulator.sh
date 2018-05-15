# make sure RL_ROOT_DIR is defined to be top of repo

# path in repo to example data
RL_EXAMPLE_DATA_DIR=${RL_ROOT_DIR}/example_data

# where results will go on your machine!
#RL_WORKING_DIR=/Users/csandeep/Documents/work/uhana/work/20161227/cleanup_RL_code

DATE=20170107
RL_WORKING_DIR=/home/csandeep/work/BETA_experiments_deepRL_cell/$DATE
mkdir -p ${RL_WORKING_DIR}

DAY=160928
CELL=136046093

# where plots go
base_results_dir=$RL_WORKING_DIR/time_variant_cell_sim
rm -rf $base_results_dir
mkdir -p $base_results_dir

MAST_CELX_fname=$RL_EXAMPLE_DATA_DIR/timeseries/MAST.CELX.$CELL.$DAY.csv
timeseries_dir=$RL_EXAMPLE_DATA_DIR/timeseries/

train_test_day_file=${RL_EXAMPLE_DATA_DIR}/conf/train_test_day.csv
# join original data on this field with train_test_day_file
date_var='DATE_Melbourne'

KPI_file=$RL_EXAMPLE_DATA_DIR/conf/KPI.txt

EXPERIMENT_RUN_MODE='PARALLEL'
TRY_CATCH_MODE='FALSE'

TOTAL_EPISODES=40
TEST_EPISODES=3
TEST_TERM=2
SAVE_TERM=$TEST_TERM

python -i ${RL_ROOT_DIR}/simulate_env/simulators/call_simulator/PARALLEL_time_variant_simulator.py --KPI_file $KPI_file --base_results_dir $base_results_dir --MAST_CELX_fname $MAST_CELX_fname --experiment_run_mode $EXPERIMENT_RUN_MODE --try_catch_mode $TRY_CATCH_MODE --TOTAL_EPISODES $TOTAL_EPISODES --TEST_EPISODES $TEST_EPISODES --SAVE_TERM $SAVE_TERM --train_test_day_file $train_test_day_file --date_var $date_var --timeseries_dir $timeseries_dir --TEST_TERM $TEST_TERM
