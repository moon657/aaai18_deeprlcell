import argparse
import os 
import sys
import pandas
import numpy as np
import datetime
from datetime import timedelta, date

# load utils from the transition matrix
RL_ROOT_DIR = os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + 'utils/'
cell_sim_utils = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'

sys.path.append(util_dir)
sys.path.append(cell_sim_utils)

from textfile_utils import list_from_textfile, remove_and_create_dir
from panel_plot_timeseries import plot_loop_KPI
from time_conversion_utils import datetimeIterator, round_datetime_to_binned_minute

def parse_args():
    """ Parse default arguments """

    parser = argparse.ArgumentParser(description='Control of Timeseries')
    parser.add_argument(
        '--base_results_dir',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161227/controlled_CELX/'
    )

    parser.add_argument(
        '--MAST_CELX_fname',
        type=str,
        required=False,
        default = '/Users/csandeep/Documents/work/uhana/work/20161128/RL_unit_test/output/master_celx_rec/MAST.CELX.136046093.160428.csv'
    )

    parser.add_argument(
        '--KPI_file',
        type=str,
        required=False,
        default='/Users/csandeep/Documents/work/uhana/work/20161227/controlled_CELX/KPI.txt'
    )

    return parser.parse_args()

def write_results(datetime_obj=None, state_history_df=None, action=None, KPI_dict=None):

    # add the state to the matrix
    curr_time_str = datetime_obj.__str__()
    KPI_dict['DATETIME'] = [curr_time_str]
    KPI_dict['ACTION'] = [action]

    KPI_df = pandas.DataFrame(KPI_dict)
    state_history_df = state_history_df.append(KPI_df)
    return state_history_df, KPI_df


def get_random_dateTime_iterator(buffer_minutes = 2, history_minutes = 10, CELX_df = None, MIN_STEP_MINUTES = 20, MAX_STEP_MINUTES = 50):

    #################################################################
    # get an iterator that loops thru dates in CELX_df timeseries
    minute_iterator_obj, start_date, end_date = get_dateTime_iterator_from_CELX(CELX_df = CELX_df, buffer_minutes = buffer_minutes, history_minutes = history_minutes)

    #print('start date', start_date)
    #print('end date', end_date)

    duration =  end_date - start_date
    SECOND_TO_MINUTE = 60

    duration_minutes = float(duration.seconds)/float(SECOND_TO_MINUTE)
    #print('duration minutes', duration_minutes)

    allowable_duration = int(duration_minutes) - int(MAX_STEP_MINUTES)
    random_offset_minutes = np.random.choice(allowable_duration)

    # get the random datetime iterator
    #print('random offset minutes', random_offset_minutes)

    random_start_date = start_date + datetime.timedelta(minutes = random_offset_minutes)

    #print('random start date', random_start_date)
    # get a datetime iterator from random_start to the end

    random_end_date = random_start_date + datetime.timedelta(minutes = MAX_STEP_MINUTES)

    # time_conversion_utils.py:def datetimeIterator
    local_minute_iterator_obj = datetimeIterator(from_date = random_start_date, to_date = random_end_date, delta=datetime.timedelta(minutes=1))

    #print(random_start_date.__str__())
    #print(end_date.__str__())

    return local_minute_iterator_obj, random_start_date, random_end_date






def get_controlled_state_transition_per_scheme(datetime_obj = None, df = None, history_minutes = None, KPI_list = None, activity_factor_multiplier = None, state_propagation_method = None, state_dict = None, action = None, KPI = 'CELLT_AGG_COLL_PER_TTI_DL', print_mode = False, noise_mean_std = None, action_feedback=True):


    # s_u': next unperturbed state
    ###############################
    next_datetime_obj = datetime_obj + datetime.timedelta(minutes = 1)

    next_state_dict, next_state_vec, next_subset_df = get_state_with_history_from_CELX(df = df, end_datetime_obj = next_datetime_obj, history_minutes = history_minutes, KPI_list = KPI_list)

    if print_mode:
        print('NEXT_KPI_DICT_LAG_0: ', next_state_dict[KPI + '_LAG_0'])
        print('NEXT_KPI_DICT_LAG_1: ', next_state_dict[KPI + '_LAG_1'])
    ###############################

    if state_propagation_method == 'feed-forward':
        # s' = s_u + A
        next_controlled_state_dict, next_controlled_state_vec, next_controlled_datetime_vec = get_next_controlled_state_with_history(next_state_dict = state_dict, activity_factor_multiplier = activity_factor_multiplier, action = action, KPI_list = KPI_list, history_minutes = history_minutes, curr_state_dict = state_dict, noise_mean_std = noise_mean_std)

    elif state_propagation_method == 'random_unperturbed':
        # s' = s_u'
        # no effect of action, totally random transition
        next_controlled_state_dict, next_controlled_state_vec, next_controlled_datetime_vec = get_next_controlled_state_with_history(next_state_dict = next_state_dict, activity_factor_multiplier = activity_factor_multiplier, action = 0, KPI_list = KPI_list, history_minutes = history_minutes, curr_state_dict = state_dict, noise_mean_std = noise_mean_std)

    elif state_propagation_method == 'add_to_next_state':
        # s' = s_u' + a
        # no effect of action, totally random transition
        next_controlled_state_dict, next_controlled_state_vec, next_controlled_datetime_vec = get_next_controlled_state_with_history(next_state_dict = next_state_dict, activity_factor_multiplier = activity_factor_multiplier, action = action, KPI_list = KPI_list, history_minutes = history_minutes, curr_state_dict = state_dict, noise_mean_std = noise_mean_std)

    else:
        pass
    # feed fwd the state

    # convert the state dict to a vector
    return next_controlled_state_dict, next_state_vec, next_controlled_state_vec

# USED
def get_state_with_history_from_CELX(df = None, end_datetime_obj = None, history_minutes = None, KPI_list = None):

    """ Given a dataframe WITH datetime index and query time t, get the state
        s_bar = s[t, t-history] and dictionary of states with keys 'KPI_LAG_0' 
        for the value at time t
    """

    start_datetime_obj = end_datetime_obj - datetime.timedelta(minutes = history_minutes)

    subset_df = df[start_datetime_obj: end_datetime_obj][KPI_list]

    state_dict = {}
    state_vec = []

    for lag in range(history_minutes):
        TIME_LAG = 'DATETIME_LAG_' + str(lag)

        specific_time_obj = end_datetime_obj - datetime.timedelta(minutes = lag) 
        specific_time_df = subset_df[specific_time_obj:specific_time_obj]
        if len(specific_time_df) == 0:
            print 'bad at %r' % specific_time_obj
            print len(df) 

        state_dict[TIME_LAG] = specific_time_obj.__str__()
        for KPI in KPI_list:
            # a lag of 0 corresponds to latest row
            KPI_lag = KPI + '_LAG_' + str(lag)

            state_at_lag = specific_time_df[KPI][0]

            state_dict[KPI_lag] = state_at_lag

            state_vec.append(state_at_lag)

    return state_dict, state_vec, subset_df

def state_vec_from_state_dict(state_dict = None, KPI_list = None, history_minutes = None):

    state_vec = []
    datetime_vec = []
    for lag in range(history_minutes):
        TIME_LAG = 'DATETIME_LAG_' + str(lag)

        specific_time = state_dict[TIME_LAG]
        datetime_vec.append(specific_time)

        for KPI in KPI_list:
            # a lag of 0 corresponds to latest row
            KPI_lag = KPI + '_LAG_' + str(lag)

            state_at_lag = state_dict[KPI_lag]

            state_vec.append(state_at_lag)

    return state_vec, datetime_vec

def get_dateTime_iterator_from_CELX(CELX_df = None, buffer_minutes = None, history_minutes = None):
    """ Get min, max datetimes from CELX and return an iterator to 
    sequentially cycle through date ranges
    """

    # min and max datetimes in the dataset
    min_datetime = CELX_df.index[0].to_datetime()
    max_datetime = CELX_df.index[-1].to_datetime()
    min_datetime = min_datetime.replace(hour=max(8, min_datetime.hour))
    max_datetime = max_datetime.replace(hour=min(20, max_datetime.hour))
    # print min_datetime, max_datetime

    # need to let at least 'history' timepoints pass before we make state
    start_date = min_datetime + datetime.timedelta(minutes = history_minutes)
    end_date = max_datetime -  datetime.timedelta(minutes = buffer_minutes)

    minute_iterator_obj = datetimeIterator(
        from_date=start_date, to_date=end_date)

    return minute_iterator_obj, start_date, end_date

def get_next_controlled_state_with_history(next_state_dict = None, control_KPI = 'CELLT_AGG_COLL_PER_TTI_DL', activity_factor_multiplier = 10, action = None, KPI_list = None, history_minutes = None, curr_state_dict= None, noise_mean_std = None, action_feedback=True):

    noise_mean = noise_mean_std[0]
    noise_std = noise_mean_std[1]

    num_samples = 1
    noise = np.random.normal(noise_mean, noise_std, num_samples)

    # s' = s_u' + a
    next_controlled_state_dict = next_state_dict
    control_KPI_key = control_KPI + '_LAG_0'
    delta_C = next_state_dict[control_KPI_key] - next_state_dict[control_KPI+'_LAG_1']
    #next_controlled_state_dict[control_KPI_key] = next_state_dict[control_KPI_key] + activity_factor_multiplier*action + noise[0]
    if action > 0:
        next_controlled_state_dict[control_KPI_key] = delta_C + curr_state_dict[control_KPI_key] + activity_factor_multiplier*action + noise[0]
    else:
        next_controlled_state_dict[control_KPI_key] = next_state_dict[control_KPI_key] + noise[0]
    # for feedback in other lags

    if(action_feedback):
        for lag in range(history_minutes):
            if(lag > 0):
                lag_key = control_KPI + '_LAG_' + str(lag)
                curr_key = control_KPI + '_LAG_' + str(lag-1)
                next_controlled_state_dict[lag_key] = curr_state_dict[curr_key]

    next_controlled_state_vec, next_controlled_datetime_vec = state_vec_from_state_dict(state_dict = next_controlled_state_dict, KPI_list = KPI_list, history_minutes = history_minutes)
    return next_controlled_state_dict, next_controlled_state_vec, next_controlled_datetime_vec

if __name__ == '__main__':
    args = parse_args()

    # load in all args
    ###########################
    base_results_dir = args.base_results_dir
    KPI_file = args.KPI_file
    MAST_CELX_fname = args.MAST_CELX_fname
    experiment_prefix = 'TELSTRA'

    CELX_df = pandas.read_csv(MAST_CELX_fname)
    thpt_var = 'THPT'
    MAX_THPT_VALUE = 100000
    print_mode = True
    KPI_list = list_from_textfile(KPI_file)
    cell_id = '136046093'

    # plot the KPI
    ###########################
    datetime_mode = True
    time_variable = 'DATETIME'
    fmt = '%Y-%m-%d %H:%M:%S'
    marker_plot_mode = True
    start_time = 'EMPTY'
    end_time = 'EMPTY'
    title_prefix = 'MAST_CELX_unperturbed'
    
    KPI_list = ['CELLT_AVG_NUM_SESS', 'CELLT_AGG_COLL_PER_TTI_DL']
    plot_loop_KPI(KPI_list, MAST_CELX_fname, datetime_mode,
                  time_variable, fmt, base_results_dir, cell_id,
                  title_prefix, marker_plot_mode, start_time, end_time,
                  experiment_prefix)
    ###########################

    # get_state_from_CELX(CELX_df = CELX_df, current_datetime_obj)

    # read in dataset
    dateparse = lambda dates: [pandas.datetime.strptime(d, fmt) for d in dates]
    df = pandas.read_csv(MAST_CELX_fname, parse_dates=['DATETIME'], date_parser=dateparse)
    df.set_index('DATETIME', inplace = True)

    # loop thru time by minutes based on loop time decide which matrix to load, then sample randomly from that matrix
    history_minutes = 5
    # need to let at least 'history' timepoints pass before we make state
    buffer_minutes = 2

    # construct state vector for a given time
    
    impulse_interval_minutes = 7

    print('looping by minutes')
    control_list = [0.0, 0.20, 1.0]

    state_propagation_mode_list = ['feed-forward', 'random_unperturbed', 'add_to_next_state']
    #state_propagation_mode_list = ['add_to_next_state']

    for state_propagation_method in state_propagation_mode_list:
        experiment_path = base_results_dir + '_' + state_propagation_method + '/' 
        remove_and_create_dir(experiment_path)

        for control in control_list:
            state_history_df = pandas.DataFrame()
            minute_iterator_obj, start_date, end_date = get_dateTime_iterator_from_CELX(CELX_df = df, buffer_minutes = buffer_minutes, history_minutes = history_minutes)

            # per minute calculation
            for datetime_obj in minute_iterator_obj:
                if print_mode:
                    print(' ')
                    print('DATETIME_OBJ: ', datetime_obj.__str__())

                # s_u: current unperturbed_state 
                ###############################

                # this makes a huge difference
                # if no if clause, there is no a minor difference between next state add and feed-forward

                # if the clause exists, and we feedback the state, only next state additive looks correct 
                if(datetime_obj == start_date):
                    state_dict, state_vec, subset_df = get_state_with_history_from_CELX(df = df, end_datetime_obj = datetime_obj, history_minutes = history_minutes, KPI_list = KPI_list)

                KPI = 'CELLT_AGG_COLL_PER_TTI_DL'

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

                if print_mode:
                    print('KPI_DICT_LAG_0: ', state_dict[KPI + '_LAG_0'])

                activity_factor_multiplier = 5
                noise_mean_std = [0, .1]

                next_controlled_state_dict, next_state_vec, next_controlled_state_vec = get_controlled_state_transition_per_scheme(datetime_obj = datetime_obj, df = df, history_minutes = history_minutes, KPI_list = KPI_list, activity_factor_multiplier = activity_factor_multiplier,  state_propagation_method = state_propagation_method, state_dict = state_dict, action = action, noise_mean_std = noise_mean_std)
              
                state_vec = next_controlled_state_vec

                state_dict = next_controlled_state_dict
                if print_mode:
                    print('state_propagation_method')
                    print(state_propagation_method)
                    print('state_vec')
                    print(state_vec)
                    print('next_state_vec')
                    print(next_state_vec)
                    print('next_controlled_state_vec')
                    print(next_controlled_state_vec)
                print(' ')


                state_history_df, KPI_df = write_results(
                    datetime_obj=datetime_obj,
                    state_history_df=state_history_df,
                    action=action,
                    KPI_dict=state_dict)
   
            # DONE WITH SIMULATION, PLOT!
            #############################################################
            # mean of collision metrics
            cleaned_state_df = state_history_df.replace([np.inf, -np.inf],
                                                        np.nan).dropna()

            control_KPI = 'CELLT_AGG_COLL_PER_TTI_DL'
            control_KPI_key = control_KPI + '_LAG_0'
            sum_collision = cleaned_state_df.sum()[control_KPI_key]

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


            KPI_list_with_lags = []
        
            for lag in range(history_minutes):
                for KPI in KPI_list: 
                    control_KPI_key = control_KPI + '_LAG_' + str(lag)
                    KPI_list_with_lags.append(control_KPI_key)

            plot_loop_KPI(KPI_list_with_lags, state_history_csv, datetime_mode,
                          time_variable, fmt, experiment_path, cell_id,
                          title_prefix, marker_plot_mode, start_time, end_time,
                          experiment_prefix)
