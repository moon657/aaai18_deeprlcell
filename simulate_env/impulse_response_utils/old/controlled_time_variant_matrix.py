import json
import argparse
import sys
import os
import pandas
import numpy as np
from datetime import timedelta, date

# load utils from the transition matrix
PIRAN_ROOT = os.environ['PIRAN_ROOT']
util_dir = PIRAN_ROOT + '/learning_suite/python/utils/'
sys.path.append(util_dir)

forecaster_utils = PIRAN_ROOT + '/forecaster/src/python/'
sys.path.append(forecaster_utils)
cell_sim_utils = PIRAN_ROOT + '/learning_suite/python/control_theory_code/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils)

from time_conversion_utils import *
from utils import list_from_file, list_from_textfile
from panel_plot_timeseries import *
from sample_transition_matrix_utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Transition Matrix')
    parser.add_argument(
        '--trans_matrix_json_name',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161128/results_transition_matrix/transition_matrix.json',
        help='name of json with transition matrix')
    parser.add_argument(
        '--state_value_map',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161128/results_transition_matrix/states_counts.csv',
        help='map from state to boundaries')

    parser.add_argument(
        '--base_results_dir',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161227/time_variant_impulses/'
    )
    parser.add_argument(
        '--KPI_file',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161128/impulse_data/KPI.txt'
    )

    return parser.parse_args()

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def remove_and_create_dir(path):
    dir = os.path.dirname(path)
    print('attempting to delete ', dir, ' path ', path)
    if os.path.exists(dir):
        os.system("rm -rf " + dir)
    os.system("mkdir -p " + dir)

if __name__ == '__main__':
    args = parse_args()

    # load in all args
    ###########################
    trans_matrix_json_name = args.trans_matrix_json_name
    state_value_map_csv = args.state_value_map
    base_results_dir = args.base_results_dir
    KPI_file = args.KPI_file

    thpt_var = 'THPT'
    max_thpt_value = 100000
    print_mode = False

    state_value_map_df = pandas.read_csv(state_value_map_csv)
    state_index_var = 'STATE_INDEX'
    state_value_map_df.set_index(state_index_var, inplace=True)
    state_value_map_dict = state_value_map_df.to_dict()
    # state_value_map_dict[KPI][state]

    KPI_list = list_from_textfile(KPI_file)

    with open(trans_matrix_json_name) as fp:
        json_object = json.load(fp)

    # overall transition matrix
    T = np.matrix(json_object['transition_matrix'])
    states = json_object['states']
    print(json_object.keys())

    # how example keys look:
    # [u'11:30', u'states', u'dates', u'13:0-13:15', u'12:15-12:30', u'12:30-12:45', u'12:45-13:0', u'11:30-11:45', u'13:15', u'transition_matrix', u'12:45', u'11:45', u'ranking_key', u'11:45-12:0', u'12:0-12:15', u'13:0', u'data', u'12:30', u'12:15', u'12:0']

    # states: 
    # [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]]
    # [[time_bin_index, state_number]] : in our example [0:8, 0:63]

    # >>> np.matrix(json_object['11:30']).shape
    # (64, 64)

    # loop thru time by minutes based on loop time decide which matrix to load, then sample randomly from that matrix
    year = 2016
    month = 01
    day = 01
    hour = 11
    minute = 30
    second = 0
    rest = 0

    start_date = datetime.datetime(year, month, day, hour, minute, second, rest)
    minute_duration = 90
    end_date = start_date + datetime.timedelta(minutes=minute_duration)
    matrix_bins_minutes = 15
    impulse_interval_minutes = 5

    print('looping by minutes')
    control_list = [0.0, 0.20, 1.0]

    sampling_procedure = 'prob'
    state_propagation_mode_list = ['additive', 'feed-forward', 'random_unperturbed', 'add_to_next_state']

    for state_propagation_method in state_propagation_mode_list:
        experiment_path = base_results_dir + state_propagation_method + '/' 
        remove_and_create_dir(experiment_path)

        for control in control_list:
            state_history_df = pandas.DataFrame()
            minute_iterator_obj = datetimeIterator(
                from_date=start_date, to_date=end_date)

            # per minute calculation
            for datetime_obj in minute_iterator_obj:

                # load time variant transition
                T, non_NA_rows = load_transition_matrices(
                    datetime_obj=datetime_obj,
                    matrix_bins_minutes=matrix_bins_minutes,
                    json_object=json_object)

                time_str = datetime_to_timeStr(datetime_obj)

                # which 15 min matrix to start from?
                binned_datetime_obj = round_datetime_to_binned_minute(
                    datetime_obj, matrix_bins_minutes)
                impulse_datetime_obj = round_datetime_to_binned_minute(
                    datetime_obj, impulse_interval_minutes)

                if (datetime_obj == impulse_datetime_obj):
                    if print_mode:
                        print('impulse interval')
                        print('IMPULSE')
                        print('current_time', datetime_obj.__str__())
                        print('impulse time', impulse_datetime_obj.__str__())
                        print('')

                    action = control
                else:
                    action = 0.0

                # at beginning, get an unperturbed state
                if (datetime_obj == binned_datetime_obj):
                    if print_mode:
                        print('')
                        print('start of minute')
                        print('current_time', datetime_obj.__str__())
                        print('time_str', time_str)
                        print('rounded_time', binned_datetime_obj.__str__())

                    # T: transition_matrix for the 15 min list
                    # non_NA_rows: list of states populated for current-time bin
                    # ex: [12, 21, 22, 25, 26]

                    # say states are ranked from 0 to 10 from least congested to most
                    # at any time bin, all 10 states exist but congested times mostly have states 9-10
                    # figure out where most of the states' mass is and samply only from there e.g 9-10

                    # get a state according to various sampling procedures
                    # state c_u: collision, unperturbed 
                    state_num, trans_matrix_info_dict, KPI_dict_state, state = get_specific_state(
                        T=T,
                        non_NA_rows=non_NA_rows,
                        sampling_procedure=sampling_procedure,
                        KPI_list=KPI_list,
                        state_value_map_dict=state_value_map_dict)
                    action = 0.0

                else:
                    # state is already the previous state
                    pass
        
                if print_mode:
                    print('KPI_dict_state', KPI_dict_state)

                # c_u': next unperturbed state 
                next_unperturbed_state_num, trans_matrix_info_dict, KPI_dict_next_unperturbed_state, next_unperturbed_state = get_specific_state(
                    T=T,
                    non_NA_rows=non_NA_rows,
                    sampling_procedure='prob',
                    KPI_list=KPI_list,
                    state_value_map_dict=state_value_map_dict)

                if print_mode:
                    print('KPI_dict_next_unperturbed_state',
                          KPI_dict_next_unperturbed_state)

                # state is loaded, c_u = state, u = unperturbed
                # c_a' = c_u + action, c_a' = next_state_with_action
                # update dictionary of cts state to new cts state with action

                next_state_with_action, thpt, KPI_dict_next_state_with_action = get_controlled_new_state(
                    KPI_dict=KPI_dict_state,
                    action=action,
                    KPI_list=KPI_list,
                    activity_multiplier=10,
                    thpt_var=thpt_var,
                    max_thpt_value=max_thpt_value)
                
                if print_mode:
                    print('KPI_dict_next_state_with_action',
                          KPI_dict_next_state_with_action)

                # map c_u to next state c' in SEVERAL competing ways

                # sum the collision metrics only
                # several ways to compute KPI_dict_next_state: additive, feedfwd, random sample
                if state_propagation_method == 'additive':
                    # c' = c_a' + c_u'  where c_a' = c_u + A
                    # not the best solution
                    KPI_dict_next_state = KPI_dict_next_unperturbed_state
                    KPI_dict_next_state[
                        'CELLT_AGG_COLL_PER_TTI_DL'] = KPI_dict_next_unperturbed_state[
                            'CELLT_AGG_COLL_PER_TTI_DL'] + KPI_dict_next_state_with_action[
                                'CELLT_AGG_COLL_PER_TTI_DL']
                elif state_propagation_method == 'feed-forward':
                    # c' = c_a'
                    KPI_dict_next_state = KPI_dict_next_state_with_action
                elif state_propagation_method == 'random_unperturbed':
                    # c' = c_u'
                    # no effect of action, totally random transition
                    KPI_dict_next_state = KPI_dict_next_unperturbed_state

                elif state_propagation_method == 'add_to_next_state':
                    # c' = c_u' + a
                    # no effect of action, totally random transition
                    next_state, thpt, KPI_dict_next_state = get_controlled_new_state(
                        KPI_dict=KPI_dict_next_unperturbed_state,
                        action=action,
                        KPI_list=KPI_list,
                        activity_multiplier=10,
                        thpt_var=thpt_var,
                        max_thpt_value=max_thpt_value)

                else:
                    pass
                # feed fwd the state
                KPI_dict_state = KPI_dict_next_state

                state_history_df, KPI_df = write_results(
                    datetime_obj=datetime_obj,
                    state_history_df=state_history_df,
                    action=action,
                    KPI_dict=KPI_dict_state)

            # mean of collision metrics
            cleaned_state_df = state_history_df.replace([np.inf, -np.inf],
                                                        np.nan).dropna()
            sum_collision = cleaned_state_df.sum()['CELLT_AGG_COLL_PER_TTI_DL']

            cell_id = '136046093'
            experiment_prefix = '_'.join(
                ['action', str(control), 'prop', state_propagation_method])

            state_history_csv = experiment_path + '/state_history.' + experiment_prefix + '.csv'
            state_history_df.set_index('DATETIME')
            state_history_df.to_csv(state_history_csv, inplace=True)

            datetime_mode = True
            time_variable = 'DATETIME'
            fmt = '%Y-%m-%d %H:%M:%S'
            marker_plot_mode = True
            start_time = 'EMPTY'
            end_time = 'EMPTY'
            title_prefix = experiment_prefix + '_'.join(
                ['_sum_collision', str(sum_collision)])

            
            plot_loop_KPI(KPI_list, state_history_csv, datetime_mode,
                          time_variable, fmt, experiment_path, cell_id,
                          title_prefix, marker_plot_mode, start_time, end_time,
                          experiment_prefix)
