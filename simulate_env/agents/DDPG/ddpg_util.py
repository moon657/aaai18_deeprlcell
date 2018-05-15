import tensorflow as tf
import numpy as np

# the fanin function for initialization
def fanin(shape, fanin=None):
	fanin = fanin or shape[0]
	v = 1/np.sqrt(fanin)
	return tf.random_uniform(shape,minval=-v,maxval=v)

def actor_network(state_dim, action_dim, layer1_size, layer2_size):
	with tf.variable_scope('theta_p'):
		return [tf.Variable(fanin([state_dim,layer1_size]),name='1w'),
	          tf.Variable(fanin([layer1_size],state_dim),name='1b'),
	          tf.Variable(fanin([layer1_size,layer2_size]),name='2w'),
	          tf.Variable(fanin([layer2_size],layer1_size),name='2b'),
	          tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3),name='3w'),
	          tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3),name='3b')]

def get_action(state, theta):
	with tf.variable_scope('policy', values=[state]):
		h0 = tf.identity(state,name='h0-state')
    	h1 = tf.nn.relu( tf.matmul(h0,theta[0]) + theta[1],name='h1')
    	h2 = tf.nn.relu( tf.matmul(h1,theta[2]) + theta[3],name='h2')
    	h3 = tf.identity(tf.matmul(h2,theta[4]) + theta[5],name='h3')
    	action = tf.nn.tanh(h3,name='h4-action')
    	return action

def critic_network(state_dim, action_dim, layer1_size, layer2_size):
	with tf.variable_scope('theta_q'):
		return [tf.Variable(fanin([state_dim,layer1_size]),name='1w'),
            tf.Variable(fanin([layer1_size],state_dim),name='1b'),
            tf.Variable(fanin([layer1_size+action_dim,layer2_size]),name='2w'),
            tf.Variable(fanin([layer2_size],layer1_size+action_dim),name='2b'),
            tf.Variable(tf.random_uniform([layer2_size,1],-3e-4,3e-4),name='3w'),
            tf.Variable(tf.random_uniform([1],-3e-4,3e-4),name='3b')]

def get_qvalue(state, action, theta):
	with tf.variable_scope('qvalue', values=[state,action]):
		h0 = tf.identity(state,name='h0-state')
		h0a = tf.identity(action,name='h0-action')
		h1  = tf.nn.relu( tf.matmul(h0,theta[0]) + theta[1],name='h1')
		h1a = tf.concat([h1,action],1)
		h2  = tf.nn.relu( tf.matmul(h1a,theta[2]) + theta[3],name='h2')
		qs  = tf.matmul(h2,theta[4]) + theta[5]
		q = tf.squeeze(qs,[1],name='h3-q')
		return q

def policy_network(state_dim, action_dim, layer1_size):
	with tf.variable_scope('theta_pi'):
		return [tf.Variable(tf.random_normal([state_dim,layer1_size]),name='1w'),
	          tf.Variable(tf.zeros([layer1_size]),name='1b'),
	          tf.Variable(tf.random_normal([layer1_size,action_dim],stddev=0.1),name='2w'),
	          tf.Variable(tf.zeros([action_dim]),name='2b')]

def value_network(state_dim, layer1_size):
	with tf.variable_scope('theta_val'):
		return [tf.Variable(tf.random_normal([state_dim,layer1_size]),name='1w'),
	          tf.Variable(tf.zeros([layer1_size]),name='1b'),
	          tf.Variable(tf.random_normal([layer1_size,1],stddev=0.1),name='2w'),
	          tf.Variable(tf.zeros([1]),name='2b')]

def get_policy(state, theta):
	with tf.variable_scope('policy', values=[state]):
		h1 = tf.nn.tanh( tf.matmul(state,theta[0]) + theta[1],name='h1')
		action = tf.matmul(h1,theta[2]) + theta[3]
		return action

def get_value(state, theta):
	with tf.variable_scope('value', values=[state]):
		h1 = tf.nn.tanh( tf.matmul(state,theta[0]) + theta[1],name='h1')
		value = tf.matmul(h1,theta[2]) + theta[3]
		value = tf.squeeze(value, [1])
		return value

#########################
# extension to k-head NNs
#########################
def actor_network_khead(state_dim, action_dim, layer1_size, layer2_size, num_head):
	with tf.variable_scope('theta_p'):
		variables = [tf.Variable(fanin([state_dim,layer1_size]),name='1w'),
		      tf.Variable(fanin([layer1_size],state_dim),name='1b'),
		      tf.Variable(fanin([layer1_size,layer2_size]),name='2w'),
		      tf.Variable(fanin([layer2_size],layer1_size),name='2b')]
		for i in xrange(num_head):
			variables += [tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3),name='3w-'+str(i)),
		      tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3),name='3b-'+str(i))]
		return variables

def get_action_khead(state, theta, k):
	with tf.variable_scope('policy', values=[state,k]):
		k_mask = list(k.eval())
		actions = []
		h1 = tf.nn.relu( tf.matmul(state,theta[0]) + theta[1],name='h1')
		h2 = tf.nn.relu( tf.matmul(h1,theta[2]) + theta[3],name='h2')
		for i, mask in enumerate(k_mask):
			if mask:
				actions.append(tf.nn.tanh(tf.matmul(h2,theta[2*i+4]) + theta[2*i+5]))
		return actions

def critic_network_khead(state_dim, action_dim, layer1_size, layer2_size, num_head):
	with tf.variable_scope('theta_q'):
		variables = [tf.Variable(fanin([state_dim,layer1_size]),name='1w'),
		    tf.Variable(fanin([layer1_size],state_dim),name='1b'),
		    tf.Variable(fanin([layer1_size+action_dim,layer2_size]),name='2w'),
		    tf.Variable(fanin([layer2_size],layer1_size+action_dim),name='2b')]
		for i in xrange(num_head):
			variables += [tf.Variable(tf.random_uniform([layer2_size,1],-3e-4,3e-4),name='3w-'+str(i)),
		    tf.Variable(tf.random_uniform([1],-3e-4,3e-4),name='3b-'+str(i))]
		return variables

def get_qvalue_khead(state, action, theta, k):
	with tf.variable_scope('qvalue', values=[state,action,k]):
		k_mask = list(k.eval())
		qvalues = []
		h1  = tf.nn.relu( tf.matmul(state,theta[0]) + theta[1],name='h1')
		h1a = tf.concat(1,[h1,action])
		h2  = tf.nn.relu( tf.matmul(h1a,theta[2]) + theta[3],name='h2')
		for i, mask in enumerate(k_mask):
			if mask:
				qvalues.append(tf.squeeze(tf.matmul(h2,theta[2*i+4]) + theta[2*i+5],[1]))
		return qvalues

def ppc_actor_network_khead(state_dim0, state_dim1, action_dim, layer1_size, layer2_size, num_head):
	with tf.variable_scope('theta_p'):
		variables = [tf.Variable(fanin([state_dim0,layer1_size]),name='1w-1'),
		      tf.Variable(fanin([layer1_size],state_dim0),name='1b-1'),
		      tf.Variable(fanin([state_dim1,layer1_size]),name='1w-2'),
		      tf.Variable(fanin([layer1_size],state_dim1),name='1b-2'),
		      tf.Variable(fanin([2*layer1_size,layer2_size]),name='2w'),
		      tf.Variable(fanin([layer2_size],2*layer1_size),name='2b')]
		for i in xrange(num_head):
			variables += [tf.Variable(tf.random_uniform([layer2_size,action_dim],-3e-3,3e-3),name='3w-'+str(i)),
		      tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3),name='3b-'+str(i))]
		return variables

def get_ppc_action_khead(state0, state1, theta, k):
	with tf.variable_scope('policy', values=[state0,state1,k]):
		k_mask = list(k.eval())
		actions = []
		h11 = tf.nn.relu( tf.matmul(state0,theta[0]) + theta[1],name='h1-1')
		h12 = tf.nn.relu( tf.matmul(state1,theta[2]) + theta[3],name='h1-2')
		h1a = tf.concat(1,[h11,h12])
		h2 = tf.nn.relu( tf.matmul(h1a,theta[4]) + theta[5],name='h2')
		for i, mask in enumerate(k_mask):
			if mask:
				actions.append(tf.nn.tanh(tf.matmul(h2,theta[2*i+6]) + theta[2*i+7]))
		return actions

def ppc_critic_network_khead(state_dim0, state_dim1, action_dim, layer1_size, layer2_size, num_head):
	with tf.variable_scope('theta_q'):
		variables = [tf.Variable(fanin([state_dim0,layer1_size]),name='1w-1'),
		    tf.Variable(fanin([layer1_size],state_dim0),name='1b-1'),
		    tf.Variable(fanin([state_dim1,layer1_size]),name='1w-2'),
		    tf.Variable(fanin([layer1_size],state_dim1),name='1b-2'),
		    tf.Variable(fanin([2*layer1_size+action_dim,layer2_size]),name='2w'),
		    tf.Variable(fanin([layer2_size],2*layer1_size+action_dim),name='2b')]
		for i in xrange(num_head):
			variables += [tf.Variable(tf.random_uniform([layer2_size,1],-3e-4,3e-4),name='3w-'+str(i)),
		    tf.Variable(tf.random_uniform([1],-3e-4,3e-4),name='3b-'+str(i))]
		return variables

def get_ppc_qvalue_khead(state0, state1, action, theta, k):
	with tf.variable_scope('qvalue', values=[state0,state1,action,k]):
		k_mask = list(k.eval())
		qvalues = []
		h11  = tf.nn.relu( tf.matmul(state0,theta[0]) + theta[1],name='h1-1')
		h12 = tf.nn.relu( tf.matmul(state1,theta[2]) + theta[3],name='h1-2')
		h1a = tf.concat(1,[h11,h12,action])
		h2  = tf.nn.relu( tf.matmul(h1a,theta[4]) + theta[5],name='h2')
		for i, mask in enumerate(k_mask):
			if mask:
				qvalues.append(tf.squeeze(tf.matmul(h2,theta[2*i+6]) + theta[2*i+7],[1]))
		return qvalues

def policy_network_khead(state_dim, action_dim, layer1_size, K):
	with tf.variable_scope('theta_pi'):
		variables = [tf.Variable(tf.random_normal([state_dim,layer1_size]),name='1w'),
		      tf.Variable(tf.zeros([layer1_size]),name='1b')]
		for i in xrange(K):
			variables += [tf.Variable(tf.random_normal([layer1_size,action_dim],stddev=0.1),name='2w-'+str(i)),
		      	tf.Variable(tf.zeros([action_dim]),name='2b-'+str(i))]
		return variables

def value_network_khead(state_dim, layer1_size, K):
	with tf.variable_scope('theta_val'):
		variables = [tf.Variable(tf.random_normal([state_dim,layer1_size]),name='1w'),
		      tf.Variable(tf.zeros([layer1_size]),name='1b')]
		for i in xrange(K):
		    variables += [tf.Variable(tf.random_normal([layer1_size,1],stddev=0.1),name='2w-'+str(i)),
		      tf.Variable(tf.zeros([1]),name='2b-'+str(i))]
		return variables

def get_policy_khead(state, theta, k):
	with tf.variable_scope('policy', values=[state,k]):
		h1 = tf.nn.tanh( tf.matmul(state,theta[0]) + theta[1],name='h1')
		k_mask = list(k.eval())
		actions = []
		for i, mask in enumerate(k_mask):
			if mask:
				actions.append(tf.matmul(h1,theta[2*i+2]) + theta[2*i+3])
		return actions

def get_value_khead(state, theta, k):
	with tf.variable_scope('value', values=[state,k]):
		h1 = tf.nn.tanh( tf.matmul(state,theta[0]) + theta[1],name='h1')
		k_mask = list(k.eval())
		values = []
		for i, mask in enumerate(k_mask):
			if mask:
				value = tf.matmul(h1,theta[2*i+2]) + theta[2*i+3] 
				value = tf.squeeze(value, [1])
				values.append(value)
		return values

# def policy_network(state_dim, action_dim, layer1_size):
# 	W1 = tf.get_variable('W1', [state_dim,layer1_size], initializer=tf.random_normal_initializer())
# 	b1 = tf.get_variable('b1', [layer1_size], initializer=tf.constant_initializer(0))
# 	h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
# 	W2 = tf.get_variable('W2', [layer1_size,action_dim], initializer=tf.random_normal_initializer(stddev=0.1))
# 	b2 = tf.get_variable('b2', [action_dim], initializer=tf.constant_initializer(0))
# 	return tf.matmul(h1,W2) + b2

def exponential_moving_averages(theta, tau):
	ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
	update = ema.apply(theta)  # also creates shadow vars
	averages = [ema.average(x) for x in theta]
	return averages, update

class TFFun:
	"""
	class for calling tf functions.
	"""
	def __init__(self, inputs, outputs):
		self._inputs = inputs if type(inputs)==list else [inputs]
		self._outputs = outputs
		self._session = tf.get_default_session()

	def __call__(self, *args):
		feeds = {}
		for (argpos, arg) in enumerate(args):
			feeds[self._inputs[argpos]] = arg
		return self._session.run(self._outputs, feeds)


