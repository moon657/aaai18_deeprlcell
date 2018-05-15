import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import cPickle
import json
import pandas
import sys, os

RL_ROOT_DIR = os.environ['RL_ROOT_DIR']
cell_sim_utils_dir = RL_ROOT_DIR + '/simulate_env/cell_sim_utils/'
sys.path.append(cell_sim_utils_dir)

from reward_computation_utils import *
from simple_cell_utils import report_rewards, get_simplified_reward

class SimpleCellSimulatorEnv(gym.Env):

    """ Simple openAI cell simulator with dynamics c' = c + A + noise
    Args:
        - env_params_dict: dictionary of env parameters

    Returns:
        - None

    """

    """ Init the environment
    """
    def __init__(self, env_params_dict=None):

        # env_params_dict: place all arguments for the simulator here
        ##############################################
        self.print_mode = env_params_dict['print_mode']
        self.deterministic_reset_mode = env_params_dict[
            'deterministic_reset_mode']

        self.continuous_action_mode = env_params_dict['continuous_action_mode']
        # parameters for reward computation
        self.reward_params_dict = env_params_dict['reward_params_dict']

        # for discrete actions, map a numeric action from 1 to N to a burst_prob from 0 to 1
        self.burst_prob_params_dict = env_params_dict['burst_prob_params_dict']
        # [C, A, N, E] = [0, 0, 0, 0]
        self.min_state_vector = env_params_dict['min_state_vector']
        # [C, A, N, E] = [1, 1, 1, 1]
        self.max_state_vector = env_params_dict['max_state_vector']
        # value to reset to, such as min_state_vector
        self.reset_state_vector = env_params_dict['reset_state_vector']
        self.thpt_var = env_params_dict['thpt_var']

        # if we have a very low thpt for num_last_entries after min_iterations_before_done steps of the simulation is over, send a done and reset
        self.min_iterations_before_done = env_params_dict[
            'min_iterations_before_done']
        self.num_last_entries = env_params_dict['num_last_entries']
        self.bad_thpt_threshold = env_params_dict['bad_thpt_threshold']
        # does reward have [K-B'] penalty for throughput before hard limit of K?
        self.hard_thpt_limit_flag = env_params_dict['hard_thpt_limit_flag']

        # a dataframe of state, action, reward history for logging
        self.reward_history_df = pandas.DataFrame()
        self.iteration_index = 0
        # batch = EPISODE
        self.batch_number = 0
        ##############################################
        # what range are the burst probabilities?
        # map discrete/continuous actions to bursts
        if (self.continuous_action_mode):
            self.action_space = spaces.Box(
                low=self.burst_prob_params_dict['min_burst_prob'],
                high=self.burst_prob_params_dict['max_burst_prob'],
                shape=(1,))
        else:
            # discrete actions
            self.burst_prob_params_dict = get_burstProb_actions(
                self.burst_prob_params_dict)
            self.action_space = self.burst_prob_params_dict['action_space']

        # state space
        self.observation_space = spaces.Box(low=self.min_state_vector,
                                            high=self.max_state_vector)

        if (self.print_mode):
            print('FUNCTION init')
            print('action space', self.action_space)
            print('observation space', self.observation_space)
        self._seed()

    """ 
        Set random seed
    """
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    """ 
        Dynamics: given s,a return s', reward, done
    """
    def _step(self, action):

        # convert action to a burst btwn [0,1]
        if (self.continuous_action_mode):
            burst = action
        else:
            action_to_burst_prob_dict = self.burst_prob_params_dict[
                'action_to_burst_prob']
            burst = action_to_burst_prob_dict[action]

# state is just C, collision
        coll_metric = self.state

        # noisy dynamics
        noise_mean = 0
        noise_std = .1
        num_samples = 1
        noise = np.random.normal(noise_mean, noise_std, num_samples)

        # state update
        # c' = c + a + noise
        next_state = coll_metric + burst + noise

        # original cell thpt is inversely related to C
        orig_thpt = float(self.reward_params_dict['B_0']) / coll_metric

        if (self.print_mode):
            print('state', self.state)
            print('action', action)
            print('burst', burst)
            print('next_state', next_state)
            print('orig_thpt', orig_thpt)

# compute the reward based on the action, current state
        reward, burst, PPC_data_MB_scheduled, user_lost_data_MB, new_cell_thpt = get_simplified_reward(
            action=burst,
            reward_params_dict=self.reward_params_dict,
            print_mode=self.print_mode,
            burst_prob_user_selector='same_as_PPC',
            orig_thpt=orig_thpt,
            hard_thpt_limit_flag=self.hard_thpt_limit_flag)

        # logging step
        # append the state information and reward to reward_history_df, which is
        # used for later plotting
        self.reward_history_df = report_rewards(
            state=self.state,
            burst=burst,
            reward=reward,
            reward_history_df=self.reward_history_df,
            iteration_index=self.iteration_index,
            PPC_data_MB_scheduled=PPC_data_MB_scheduled,
            user_lost_data_MB=user_lost_data_MB,
            print_mode=self.print_mode,
            thpt=orig_thpt,
            new_thpt=new_cell_thpt,
            thpt_var=self.thpt_var,
            batch_number=self.batch_number)

        # just for debugging
        self.reward = reward
        self.action_taken = burst

        # update the state
        self.state = next_state
        # update the state and simulation params
        done_flag = False
        self.iteration_index += 1

        # PREMATURE ABORT STEP IF TOO BAD A THROUGHPUT
        # if we reached enough timestamps and the last N thpts are below 
        # a threshold, abort
        if (self.iteration_index >= self.min_iterations_before_done):
            # if last N points had a thpt below threshold, abort the simulation
            num_entries_low_thpt = (
                self.reward_history_df[self.thpt_var][-self.num_last_entries:]
                <= self.bad_thpt_threshold).sum()
            fraction_bad_thpt_entries = float(num_entries_low_thpt) / float(
                self.num_last_entries)
            if (fraction_bad_thpt_entries >= .9):
                done_flag = True
                if (self.print_mode):
                    print('PREMATURE ABORT DUE TO LOW THRUPUT')
                    print('num_entries_low_thpt', num_entries_low_thpt)
                    print('bad_thpt_threshold', self.bad_thpt_threshold)
                    print('last', self.reward_history_df[self.thpt_var][
                        -self.num_last_entries:])

        if (self.print_mode):
            print('FUNCTION step')
            print('curr_state', self.state)
            print('next_state', next_state)
            print('reward', reward)
            print('done_flag', done_flag)

        return next_state, reward, done_flag, {}

    """ 
        Dynamics: given s,a return s', reward, done
    """
    # reset the collision metric for a new episode
    def _reset(self):
        if (self.deterministic_reset_mode):
            self.state = self.reset_state_vector

# biased sampling of C to lower congestion states for beginning of system
        else:
            self.state = self.np_random.uniform(
                low=self.min_state_vector,
                high=float(self.max_state_vector) / 5)

        self.batch_number += 1
        self.iteration_index = 0

        if (self.print_mode):
            print('FUNCTION reset')
            print('det reset mode', self.deterministic_reset_mode)
            print('state', self.state)
        return self.state

    def _render(self, mode='human', close=False):
        if (self.print_mode):
            print('FUNCTION render')
            print('render func not implemented yet')
        return
