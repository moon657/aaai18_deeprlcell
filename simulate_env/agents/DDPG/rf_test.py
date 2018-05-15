import numpy as np
import pandas as pd
import sys, os
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes()

RL_ROOT_DIR = os.environ['RL_ROOT_DIR']
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

# random forest utils
RF_utils_dir = RL_ROOT_DIR + '/random_forest/'
sys.path.append(RF_utils_dir)

# utils
utils_dir = RL_ROOT_DIR + '/utils/'
sys.path.append(utils_dir)

# how to control a timeseries
impulse_response_utils_dir = RL_ROOT_DIR + '/simulate_env/impulse_response_utils/'
sys.path.append(impulse_response_utils_dir)


from score_random_forest import score_RF_single_input, load_saved_rf
from helper_utils_cell_simulator import random_train_day 
from textfile_utils import list_from_textfile

rf_dir = '/Users/tchu/Documents/Uhana/remote/deeprl_cell/IJCAI_exp/rf_models.txt'
rf_list = []
rf_models = list_from_textfile(rf_dir)
for rf_model in rf_models:
    rf_list.append(load_saved_rf(rf_model))
cell_id = '136046093'
timeseries_dir = '/Users/tchu/Documents/Uhana/PPC_test/example_data/timeseries/'
fig_dir = '/Users/tchu/Documents/Uhana/PPC_test/'
day = '2016_9_28'
df, _ = random_train_day(day_list = [day], master_cell_records_dir = timeseries_dir, cell = cell_id)
kpi_list = ['CELLT_AGG_SPECF_DL','CELLT_AVG_NUM_SESS','CELLT_AGG_COLL_PER_TTI_DL']
bs = np.zeros((len(rf_list), len(df)))
for i, rf in enumerate(rf_list):
	bs[i] = rf.predict(df[kpi_list].values)

def plot_multi_train_rewards(rmeans,
                       fig_path='./figs/train_reward',
                       xlabel='Training episodes',
                       ylabel='Total rewards',
                       x=None,
                       legend=None):
    if x is None:
        x = range(rmeans.shape[1])
    if legend is None:
        legend = [str(i) for i in xrange(rmeans.shape[0])]
    fig = plt.figure()
    colors = 'brgmyk'
    for i in xrange(rmeans.shape[0]):
        plt.plot(x, rmeans[i], linewidth=2, color=colors[i], label=legend[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    fig.savefig(fig_path)
    plt.close()

plot_multi_train_rewards(bs, fig_path=fig_dir+cell_id+'_rf_comp.png', legend=['rf_head0','rf_head1'])

