# make sure RL_ROOT_DIR is defined to be top of repo

# path in repo to example data
RL_EXAMPLE_DATA_DIR=${RL_ROOT_DIR}/example_data

# where to place results
RL_WORKING_DIR=/home/noah/Dev/uhana/cleanup_RL_code
#RL_WORKING_DIR=/Users/csandeep/Documents/work/uhana/work/20161227/cleanup_RL_code
DAY=160428
CELL=136046093

# where plots go
base_results_dir=$RL_WORKING_DIR/test_controlled_impulse_response
rm -rf $base_results_dir
mkdir -p $base_results_dir

CODE_DIR=${RL_ROOT_DIR}/simulate_env/impulse_response_utils

MAST_CELX_fname=$RL_EXAMPLE_DATA_DIR/timeseries/MAST.CELX.$CELL.$DAY.csv

KPI_file=$RL_EXAMPLE_DATA_DIR/conf/KPI.txt

python $CODE_DIR/MAST_CELX_with_control.py --KPI_file $KPI_file --base_results_dir $base_results_dir --MAST_CELX_fname $MAST_CELX_fname
