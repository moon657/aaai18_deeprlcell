import sys, os

RL_ROOT_DIR = os.environ['RL_ROOT_DIR']
# cell simulator helpers
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

from simple_cell_utils import *
from helper_utils_cell_simulator import *

train_info_csv = RL_ROOT_DIR + '/IJCAI_exp/MDDPG/example_parallel_MDDPG/0/train_info.csv'

joint_config_file = RL_ROOT_DIR + '/IJCAI_exp/MDDPG/example_parallel_MDDPG/0/joint.ini'

experiment_config_file = RL_ROOT_DIR + '/IJCAI_exp/MDDPG/example_parallel_MDDPG/0/base_MDDPG_params.ini'

heads_list, rf_list, hard_thpt_limit_list, num_heads, train_days, test_days, cell_ids = parse_head_info(train_info_csv)

print(heads_list)
print(rf_list)
print(hard_thpt_limit_list)
print(num_heads)
print(train_days)
print(test_days)
print(num_heads)
print(cell_ids)

resolve_MDDPG_config_paths(experiment_config_file = experiment_config_file,  joint_config_file = joint_config_file, train_info_csv = train_info_csv)
