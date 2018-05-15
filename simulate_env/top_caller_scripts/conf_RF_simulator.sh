# where conf files are for ATT simulation
CONF_DIR=$RL_ROOT_DIR/conf/RF/ATT_yol/MDDPG_conf
# change these config files for your local settings

# params for experiment
RL_params_config=$CONF_DIR/base_MDDPG_params.ini
# where data is on your local machine
file_paths_config=$CONF_DIR/csandeep_paths.ini

CODE_DIR=$RL_ROOT_DIR/simulate_env/simulators/call_simulator

# run several RL simulations in parallel
python -i $CODE_DIR/ConfigParse_RF_simulator.py --experiment_config_file $RL_params_config --file_path_config_file $file_paths_config
