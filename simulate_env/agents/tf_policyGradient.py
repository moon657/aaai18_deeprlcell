import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import sys,os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

RL_ROOT_DIR=os.environ['RL_ROOT_DIR']
util_dir = RL_ROOT_DIR + '/learning_suite/python/control_theory_code/state_space_generation'
sys.path.append(util_dir)

util_dir = RL_ROOT_DIR + '/learning_suite/python/utils/'
sys.path.append(util_dir)

from cell_simulator import *
from external_Q_agent_implementation import *
from panel_plot_timeseries import plot_loop_KPI
from double_panel_plot_timeseries import panel_KPI_plot
from utils import list_from_textfile
from external_Q_agent_implementation import QLearner
from reward_computation_utils import *
from time_variant_transition_matrix import state_to_vector, continuous_state_to_state_num 
from plot_reward_across_agents import *
from evaluate_agent_utils import *


# modified from: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))


def scale_state(unscaled_s, state_mean, state_scale):
	return (unscaled_s-state_mean)/state_scale

def evaluate_policy_gradient_agent(env_dict=None, burst_prob_params_dict=None, policy_agent_dict=None, plotting_info_dict = None):

	agent_type = 'policyGrad'
	plotting_info_dict['agent_type'] = agent_type

	# PARAMETERS FOR PLOTTING
	train_test_str = plotting_info_dict['TRAIN_TEST']
	experiment_prefix = plotting_info_dict['experiment_prefix']
	delay_reward_interval = plotting_info_dict['delay_reward_interval']
	base_results_dir = plotting_info_dict['base_results_dir']
	KPI_to_plot_file = plotting_info_dict['KPI_to_plot_file']
	cell_id = plotting_info_dict['cell_id']
	agent_type = plotting_info_dict['agent_type']
	datetime_mode = plotting_info_dict['datetime_mode']
	time_var = plotting_info_dict['time_var']
	date_var = plotting_info_dict['date_var']
	fmt = plotting_info_dict['fmt']
	marker_plot_mode = plotting_info_dict['marker_plot_mode']
	start_time = plotting_info_dict['start_time']
	end_time = plotting_info_dict['end_time']
	experiment_params = plotting_info_dict['experiment_params']
	cumulative_results_dir = plotting_info_dict['cumulative_results_dir']

	train_day_list = plotting_info_dict['train_day_list']
	test_day_list = plotting_info_dict['test_day_list']
	train_test_day_str = plotting_info_dict['train_test_day_str']

	experiment_params = '_'.join([experiment_prefix, cell_id, agent_type, 'delay', str(delay_reward_interval), train_test_str])

	print('evaluate policyGrad')
	# start the env
	observation, action_space, env = init_cell_environment(env_dict)

	# action space params
	action_space = burst_prob_params_dict['action_space'] 
	burst_prob_to_action = burst_prob_params_dict['burst_prob_to_action'] 
	action_to_burst_prob = burst_prob_params_dict['action_to_burst_prob']
	burst_prob_vector = burst_prob_params_dict['burst_prob_vector']
	optimal_discretized_burst = burst_prob_params_dict['optimal_discretized_burst']

	# LEARN
	####################################
	# Clear the Tensorflow graph.
	tf.reset_default_graph() 

	learning_rate = policy_agent_dict['learning_rate']
	num_hidden_layers = policy_agent_dict['num_hidden_layers']
	state_space_dimension = policy_agent_dict['state_space_dimension']
	MAX_STEP = policy_agent_dict['MAX_STEP']
	EPISODES = policy_agent_dict['EPISODES']
	PRINT_LEN = policy_agent_dict['PRINT_LEN']

	GRADIENT_UPDATE_FREQUENCY = policy_agent_dict['GRADIENT_UPDATE_FREQUENCY']
	EPISODE_REPORT_FREQUENCY = 2

	if(env.continuous_action_mode == True):
		num_actions = 1
	else:
		num_actions = burst_prob_params_dict['num_actions'] 

	# load the agent
	myAgent = agent(lr=learning_rate, s_size=state_space_dimension, a_size=num_actions, h_size=num_hidden_layers)

	# init tf
	################# 
	init = tf.initialize_all_variables()
	################# 
	state_space = env.observation_space 
	state_mean, state_scale = .5*(state_space.low+state_space.high), .5*(state_space.high-state_space.low)

	print('state mean', state_mean)
	print('state scale', state_scale)

	# Launch the tensorflow graph
	with tf.Session() as sess:
	    sess.run(init)
	    i = 0
	    total_reward = []
	    total_length = []
		
	    gradBuffer = sess.run(tf.trainable_variables())
	    for ix,grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0

	    # episodes: 
	    s = observation

	    trial = 0
	    for i in range(EPISODES):
	    	progress_str = ' '.join(['NEW EPISODE agent', agent_type, 'episode', str(i)])
	    	print(progress_str) 

		unscaled_s = env.reset()
		s = scale_state(unscaled_s, state_mean, state_scale)

		running_reward = 0
		ep_history = []

		for j in range(MAX_STEP):
		    #Choose either a random action or one from our network.
		    a_dist = sess.run(myAgent.output,feed_dict={myAgent.state_in:[s]})
		    a = np.random.choice(a_dist[0],p=a_dist[0])
		    a = np.argmax(a_dist == a)


		    # observation, reward, done, info = env.step(action)
		    unscaled_s1,r,d,_ = env.step(a) # Get our reward for taking an action given a bandit.
		    s1 = scale_state(unscaled_s1, state_mean, state_scale)

		    # create a step
		    if(j % PRINT_LEN == 0):
			    print('episode ', i, 'step: ', j)
			    burst_prob_value = action_to_burst_prob[a]
			    print('policyGrad unscaled_state ', unscaled_s, ' state ',  s)
			    print('burst prob value ', burst_prob_value)
			    print('action ', a, 'continuous', env.continuous_action_mode, 'a_dist', a_dist, 'a_dist[0]', a_dist[0])

		    ep_history.append([s,a,r,s1])
		    s = s1
		    running_reward += r

		    trial = trial + 1

		# UPDATE THE AGENT
		progress_str = ' '.join(['UPDATE NETWORK agent', agent_type, 'episode', str(i), 'trial', str(trial), 'j', str(j)])
		print(progress_str) 

		# Update the network.
		ep_history = np.array(ep_history)
		ep_history[:,2] = discount_rewards(ep_history[:,2])
		feed_dict={myAgent.reward_holder:ep_history[:,2],
			myAgent.action_holder:ep_history[:,1],myAgent.state_in:np.vstack(ep_history[:,0])}
		grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
		for idx,grad in enumerate(grads):
		    gradBuffer[idx] += grad

		if i % GRADIENT_UPDATE_FREQUENCY == 0 and i != 0:
		    feed_dict= dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
		    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
		    for ix,grad in enumerate(gradBuffer):
			gradBuffer[ix] = grad * 0
		
		total_reward.append(running_reward)
		total_length.append(j)

		    #Update our running tally of scores.
		if i % EPISODE_REPORT_FREQUENCY == 0:
		    mean_reward = np.mean(total_reward[-EPISODE_REPORT_FREQUENCY:])
		    progress_str = ' '.join(['mean reward', str(mean_reward), 'episode', str(i), 'trial', str(trial), 'j', str(j)])
		    print(progress_str) 
		i += 1

	# now save the reward results to a file
	experiment_reward_results_csv = base_results_dir + '/' + experiment_params + '.rewards.csv'
	reward_history_df = env.reward_history_df
	reward_history_df.to_csv(experiment_reward_results_csv, index=False)

	# now do a plot of reward vs time, C vs time, all metrics vs time
	batch_var = 'BATCH_NUM'
	last_batch_df, complete_experiments_df = get_last_batch_data(reward_history_df)

	# plot reward vs BATCH_NUM
	agent_list = [agent_type]
	agent_results_dir = base_results_dir + '/' + agent_type
	plot_all_agent_reward(base_results_dir, agent_list, plotting_info_dict)

	# update this to have the dataframe of interest
	panel_KPI_plot(last_batch_df, plotting_info_dict)

	return policy_agent_dict

