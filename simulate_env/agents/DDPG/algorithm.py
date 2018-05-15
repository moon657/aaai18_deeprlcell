"""
This package contains standard RL models. 

The key functions for each model include:
act(state): get the action suggested by the model from state (which is observation in gym)
explore(state): get the noised action for exploration
learn(state, action, reward, next_state, done): update the model after receiving transition data (s,a,r,s',t)

The key hyper paramters include:
GAMMA: discount factor
EPSILON: epsilon-greedy exploration rate
BATCH_SIZE: minibatch size 
REPLAY_BUFFER_SIZE: replay size (literature value: 1000000)
REPLAY_START_SIZE: minimum size of replay required for training (literature value: 10000)
"""

import numpy as np
import tensorflow as tf 
# from ou_noise import OUNoise
# from critic_network import CriticNetwork 
# from actor_network import ActorNetwork
from ddpg_util import *
from replay_buffer import ReplayBuffer, RolloutBuffer
import cPickle
import random

class Greedy:
    """
    Greedy policy finds the action that maximizes the current reward. This is the general version that requires env. state: continuous, action: continuous
    """
    def __init__(self, state_space, action_space, num_action=20):
        self.name = 'Greedy'
        self.action_space = np.linspace(action_space.low, action_space.high, num_action)

    def act(self, env):
        max_reward, max_action = float('-inf'), 0
        state, state_dict = env.state, env.state_dict
        for action in self.action_space:
            _, reward, _, _ = env.step(np.array([action]))
            if reward > max_reward:
                max_action = action
                max_reward = reward
            env.step_back(state, state_dict)
        # print max_action, max_reward
        return np.array([max_action])

class Heuristic:
    def __init__(self, thpt_limit, rf_model, thpt_max, discount=0.8):
        self.name = 'H0'
        self.rf = rf_model
        self.thpt_limit = thpt_limit
        self.scale = thpt_max - self.thpt_limit
        self.discount = discount
        # print '%s initialized! thpt_limit:%r, thpt_max:%r' % (self.name, self.thpt_limit, thpt_max)

    def act(self, state):
        next_B = self.rf.predict([state[-3:]])[0]
        # print state[-1], next_B
        if next_B <= self.thpt_limit:
            action = 0
        else:
            action = min(1, (next_B-self.thpt_limit)*1.0/self.scale)
        return np.array([action])

class LRQ:
    """
    LR(Q-learning) with online learning and epsilon-greedy exploration. state: continuous, action: discrete.
    """
    def __init__(self, state_space, action_space, ETA=0.01, GAMMA=0, EPSILON=[0.9,0.1]):
        self.name = 'LRQ'
        self.state_dim = state_space.shape[0]
        self.n_action = action_space.n
        self.W = np.zeros((self.state_dim, self.n_action))
        self.gamma = GAMMA
        self.epsilon = EPSILON[0]
        self.final_epsilon = EPSILON[1]
        self.eta = ETA
        # initialize mean and scale for scaling input to [-1,1], [-1,1] to output
        self.state_mean, self.state_scale = .5*(state_space.low+state_space.high), .5*(state_space.high-state_space.low)

    def update_epsilon(self, depsilon):
        self.epsilon= max(self.final_epsilon, self.epsilon-depsilon)

    def get_abs_wt(self):
        return np.abs(self.W).mean()

    def explore(self, state):
        if np.random.random() < self.EPSILON:
            return np.random.randint(self.n_action)
        else:
            return self.act(state)

    def act(self, state):
        scale_state = self.scale_state(state)
        return np.argmax(np.dot(scale_state, self.W))

    def get_value(self, state):
        scale_state = self.scale_state(state)
        return np.max(np.dot(scale_state, self.W))

    def scale_state(self, state):
        return (state-self.state_mean)/self.state_scale
    def get_qvalue(self, state, action):
        scale_state = self.scale_state(state)
        return np.inner(scale_state, self.W[:,action])

    def learn(self, state, action, reward, next_state, done):
        scale_state = self.scale_state(state)
        if done:
            y = reward
        else:
            y = reward + self.GAMMA*self.get_value(next_state)

        self.W[:,action] = self.W[:,action] +self.eta*(y-self.get_qvalue(state, action)) * state
        self.W[:,action] = self.W[:,action] + self.eta*(y-self.get_qvalue(state, action)) * scale_state


    def load_model(self, path='./saved_lrq/wt'):
        save_file = open(path, 'rb')
        self.W = cPickle.load(save_file)
        save_file.close()

    def save_model(self, path='./saved_lrq/wt'):
        save_file = open(path, 'wb')
        cPickle.dump(self.W, save_file, -1)
        save_file.close()


class DDPG:
    """
    DDPG with experience replay and OU noise exploration. state: continuous, action: continuous.
    """
    def __init__(self, state_space, action_space, GAMMA=0.99, REPLAY_BUFFER_SIZE=200000, WARMUP=10000, BATCH_SIZE=32, LAYER1_SIZE=400, LAYER2_SIZE=300, TAU=0.001, ETA_ACTOR=1e-4, ETA_CRITIC=1e-3, L2_ACTOR=0, L2_CRITIC=0.01, OU_NOISE=[0.10, 0.10], clip_action_explore='TRUE', scale=True, discrete=False):
        # ou_noise = [theta, sigma]
        # state_space (observation_space) and action_space are input from gym Env
        self.name = 'DDPG' 
        self.state_dim = list(state_space.shape)
        self.action_dim = list(action_space.shape) if not discrete else [action_space.n]
        self.layer1_size = LAYER1_SIZE
        self.layer2_size = LAYER2_SIZE
        self.eta_actor = ETA_ACTOR
        self.eta_critic = ETA_CRITIC
        self.l2_actor = L2_ACTOR
        self.l2_critic = L2_CRITIC
        self.tau = TAU
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.warmup = WARMUP
        self.clip_action_explore = clip_action_explore
        self.scale = scale
        self.discrete = discrete

        ####################################
        # Clear the Tensorflow graph.
        tf.reset_default_graph() 

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))

        # Randomly initialize actor network and critic networks and both their target networks
        self.actor = actor_network(self.state_dim[0], self.action_dim[0], self.layer1_size, self.layer2_size)
        self.critic = critic_network(self.state_dim[0], self.action_dim[0], self.layer1_size, self.layer2_size)
        self.target_actor, update_target_actor = exponential_moving_averages(self.actor, self.tau)
        self.target_critic, update_target_critic = exponential_moving_averages(self.critic, self.tau)

        # initialize mean and scale for scaling input to [-1,1], [-1,1] to output
        self.state_mean, self.state_scale = .5*(state_space.low+state_space.high), .5*(state_space.high-state_space.low)
        if not discrete:
            self.action_mean, self.action_scale = .5*(action_space.low+action_space.high), .5*(action_space.high-action_space.low)
        else:
            self.action_mean, self.action_scale = np.array([.5] * action_space.n), np.array([.5] * action_space.n)
        print 'action normalize:', self.action_mean, self.action_scale
        print 'state normalize:', self.state_mean, self.state_scale

        print 'OU_NOISE:', OU_NOISE
        print 'clip_action_explore:', clip_action_explore

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random Ornstein-Uhlenbeck process 
        self.ou_theta, self.ou_sigma = OU_NOISE
        state = tf.placeholder(tf.float32, [None]+self.state_dim, 'state')
        action_test = get_action(state, self.actor)
        noise_init = tf.zeros([1]+self.action_dim)
        noise_var = tf.Variable(noise_init)
        ou_reset = noise_var.assign(noise_init)
        noise = noise_var.assign_sub((self.ou_theta) * noise_var - tf.random_normal(self.action_dim, stddev=self.ou_sigma))
        exploration = action_test + noise

        # actor update
        qvalue = get_qvalue(state, action_test, self.critic)
        loss_p = -tf.reduce_mean(qvalue, 0) + tf.add_n([self.l2_actor * tf.nn.l2_loss(var) for var in self.actor])
        optim_p = tf.train.AdamOptimizer(learning_rate=self.eta_actor)
        grad_p = optim_p.apply_gradients(optim_p.compute_gradients(loss_p, var_list=self.actor))
        with tf.control_dependencies([grad_p]):
            train_p = tf.group(update_target_actor)

        # critic update
        action_train = tf.placeholder(tf.float32, [self.batch_size] + self.action_dim, 'action')
        reward = tf.placeholder(tf.float32, [self.batch_size], 'reward')
        done = tf.placeholder(tf.bool, [self.batch_size], 'done')
        next_state = tf.placeholder(tf.float32, [self.batch_size] + self.state_dim, 'next_state')
        # calculate one-step ahead TQ
        q_train = get_qvalue(state, action_train, self.critic)
        next_action = get_action(next_state, self.target_actor)
        next_q = get_qvalue(next_state, next_action, self.target_critic)
        tq_train = tf.stop_gradient(tf.where(done,reward,reward + self.gamma*next_q))
        loss_q = tf.reduce_mean(tf.square(q_train - tq_train), 0) + tf.add_n([self.l2_critic * tf.nn.l2_loss(var) for var in self.critic])
        optim_q = tf.train.AdamOptimizer(learning_rate=self.eta_critic)
        grad_q = optim_q.apply_gradients(optim_q.compute_gradients(loss_q, var_list=self.critic))
        with tf.control_dependencies([grad_q]):
            train_q = tf.group(update_target_critic)

        # tf functions
        with self.sess.as_default():
            self._act = TFFun(state,action_test)
            self._explore = TFFun(state,exploration)
            self._reset = TFFun([],ou_reset)
            self._train = TFFun([state,action_train,reward,done,next_state],[train_p,train_q])
            self._train_q = TFFun([state,action_train,reward,done,next_state],[train_q])

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

    def reset_noise(self):
        self._reset()

    def scale_state(self, state):
        return (state-self.state_mean)/self.state_scale
        # return state

    def scale_action(self, action):
        return (action-self.action_mean)/self.action_scale
        # return action

    def unscale_action(self, action):
        return action*self.action_scale+self.action_mean
        # return action

    def get_abs_wt(self):
        # not implement yet
        return [np.abs(var.eval()).mean() for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)][0]

    def train(self):
        # Sample a random minibatch of M transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_batch(self.batch_size)
        
        # for action_dim = 1
        action_batch = np.resize(action_batch,[self.batch_size,self.action_dim[0]])

        #print(action_batch)
        #print(reward_batch)
        #print(done_batch)

        if self.replay_buffer.count() > self.warmup:
            self._train(state_batch,action_batch,reward_batch,done_batch,next_state_batch)
        else:
            self._train_q(state_batch,action_batch,reward_batch,done_batch,next_state_batch)

    def explore(self, state):
        if not self.scale:
            action = self._explore([state])[0]
        else:
            action = self._explore([self.scale_state(state)])[0]
        # could be beyond limits
        unbounded_action_with_noise = self.unscale_action(action)

        if(self.clip_action_explore == 'TRUE'):
            MIN_ACTION = self.action_mean - self.action_scale
            MAX_ACTION = self.action_mean + self.action_scale
            # print MIN_ACTION, MAX_ACTION, unbounded_action_with_noise
            bounded_action_with_noise = np.minimum(np.maximum(MIN_ACTION, unbounded_action_with_noise), MAX_ACTION)
        else:
            bounded_action_with_noise = unbounded_action_with_noise
        
        #if(unbounded_action_with_noise != bounded_action_with_noise):
        #    print('unbounded explore ', unbounded_action_with_noise)
        #    print('bounded explore ', bounded_action_with_noise)
        if not self.discrete:
            return bounded_action_with_noise
        else:
            return np.argmax(bounded_action_with_noise)

    def act(self, state):
        if not self.scale:
            action = self._act([state])[0]
        else:
            action = self._act([self.scale_state(state)])[0]
        #print('tchu action ', action)
        #print('tchu scale ', self.unscale_action(action))
        if not self.discrete:
            return self.unscale_action(action)
        else:
            return np.argmax(self.unscale_action(action)[0])

    def learn(self, state, action, reward, next_state, done):
        # Store transition (s,a,r,s',t) in replay buffer
        if self.discrete:
            arr = np.zeros(self.action_dim)
            arr[action] = 1
            action = arr
        # print 'action: %r' % action
        # print 'reward: %r' % reward
        if not self.scale:
            self.replay_buffer.add(state,self.scale_action(action),reward,next_state,done)
        else:
            self.replay_buffer.add(self.scale_state(state),self.scale_action(action),reward,self.scale_state(next_state),done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >= 5*self.batch_size:
            self.train()            

    def load_model(self, episode=None, file_path='./saved_ddpg', env_name='car'):
        checkpoint = tf.train.get_checkpoint_state(file_path)
        if episode is None:
            checkpoint_path = checkpoint.model_checkpoint_path
        else:
            checkpoint_path = '%s/%s-%d' % (file_path, env_name, episode)
        print 'load model %s' % checkpoint_path
        self.saver.restore(self.sess, checkpoint_path)

    def save_model(self, episode, file_path='./saved_ddpg', env_name='car'):
        print 'save model at episode %d' % episode
        self.saver.save(self.sess, '%s/%s' % (file_path, env_name), global_step=episode)

    def print_status(self):
        print 'current weights in actor network:\n'
        actor_wts = []
        critic_wts = []
        i = 0
        for g in self.actor:
            gval = np.array(g.eval())
            gavg = np.mean(np.abs(gval))
            actor_wts.append(gavg)
            print '%d:%r' % (i, gavg)
            i += 1
        i = 0
        print 'current weights in critic network:\n'
        for g in self.critic:
            gval = np.array(g.eval())
            gavg = np.mean(np.abs(gval))
            critic_wts.append(gavg)
            print '%d:%r' % (i, gavg)
            i += 1
        return actor_wts, critic_wts

class PolicyGradient(DDPG):
    """
    Policy gradient (NN) with REINFORCE. state: continuous, action: discrete.
    """
    def __init__(self, state_space, action_space, GAMMA=0.99, ETA=0.0001, EPS=1e-9, LAYER1_SIZE=20, L2_REG=0, EPSILON=[0.5,0], norm_len=1000000, gradient_clip=40, scale=False):
        self.name = 'PG'
        self.state_dim = list(state_space.shape)
        self.action_dim = action_space.n
        self.layer1_size = LAYER1_SIZE
        #self.val_reg = VAL_REG
        self.l2_reg = L2_REG
        self.gamma = GAMMA
        self.epsilon = EPSILON[0]
        self.final_epsilon = EPSILON[1]
        self.gradient_clip = gradient_clip
        self.scale = scale
        self.state_mean, self.state_scale = .5*(state_space.low+state_space.high), .5*(state_space.high-state_space.low)
        self.rollout_buffer = RolloutBuffer(norm_len)

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.pg = policy_network(self.state_dim[0], self.action_dim, self.layer1_size)
        
        # exploration for collecting training trajectories
        state = tf.placeholder(tf.float32, [None] + self.state_dim, 'state')
        action_logits = get_policy(state, self.pg)
        action_test = tf.argmax(tf.nn.softmax(action_logits), 1)
        # tf.multinomial may not be good, so disabled
        exploration = tf.multinomial(action_logits - tf.reduce_max(action_logits, 1, keep_dims=True), 1)

        # network update
        action_train = tf.placeholder(tf.int32, [None], 'action')
        discounted_reward = tf.placeholder(tf.float32, [None], name='discounted_reward')
        # policy loss as cross entropy/ removed value
        pg_loss = tf.reduce_mean((discounted_reward) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=action_train))
        # baseline regression loss as MSE
        #value_loss = self.val_reg * tf.reduce_mean(tf.square(discounted_reward - value))
        # L2 regularization
        reg_loss = tf.add_n([self.l2_reg * tf.nn.l2_loss(var) for var in self.pg])
        # action_log_prob = tf.nn.log_softmax(action_logits)
        # entropy_loss = -self.en_reg * tf.reduce_sum(action_log_prob*tf.exp(action_log_prob))
        loss = pg_loss + reg_loss
        optim = tf.train.RMSPropOptimizer(learning_rate=ETA, epsilon=EPS)
        #grads = optim.compute_gradients(loss, var_list=self.pg)
        grads = tf.gradients(loss, self.pg)
        grads, _ =  tf.clip_by_global_norm(grads, self.gradient_clip) 
        train_gp = optim.apply_gradients(list(zip(grads, self.pg)))
        #train_gp = optim.apply_gradients(grads)

        # tf functions
        with self.sess.as_default():
            self._act = TFFun(state,action_test)
            self._explore = TFFun(state,exploration)
            self._train = TFFun([state,action_train,discounted_reward],train_gp)
            self._act_logit = TFFun(state,action_logits)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

    def update_epsilon(self, depsilon):
        self.epsilon = max(self.final_epsilon, self.epsilon-depsilon)

    def scale_state(self, state):
        return (state-self.state_mean)/self.state_scale

    def act(self, state):
        #act_prob = self._act_logit([self.scale_state(state)])[0]
        #act_prob = softmax(act_prob) - 1e-5
        #return np.argmax(np.random.multinomial(1, act_prob))
        if self.scale:
            return self._act([self.scale_state(state)])[0]
        else:
            return self._act([state])[0]

    def explore(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            if self.scale:
                return self._explore([self.scale_state(state)])[0][0]
            else:
                return self._explore([state])[0][0]
            #return self.act(state)

    def store(self, state, action, reward):
        if self.scale:
            self.rollout_buffer.add(self.scale_state(state), action, reward) 
        else:
            self.rollout_buffer.add(state, action, reward) 

    def learn(self):
        state, action, discounted_reward = self.rollout_buffer.rollout(self.gamma)
        self.rollout_buffer.erase()
        self._train(state, action, discounted_reward)
            
        #grad = self._grad(state, action, discounted_reward)
        # print [g.shape for g in grad]
        # print [abs(g).mean() for g in grad]

    def print_status(self):
        print 'current weights in policy network:\n'
        pg_wts = []
        for g in self.pg:
            gval = np.array(g.eval())
            gavg = np.mean(np.abs(gval))
            pg_wts.append(gavg)
            print gavg
        return pg_wts

class ActorCritic(PolicyGradient):
    """
    Actor critic (NN for policy network and value network) with REINFORCE. state: continuous, action: discrete.
    """
    def __init__(self, state_space, action_space, GAMMA=0.99, ETA_ACTOR=0.01, ETA_CRITIC=0.05, EPS=1e-9, LAYER1_SIZE=50, L2_REG=0, EPSILON=[0.5,0], norm_len=1000000, gradient_clip=5, scale=False):
        self.name = 'AC'
        self.state_dim = list(state_space.shape)
        self.action_dim = action_space.n
        self.layer1_size = LAYER1_SIZE
        #self.val_reg = VAL_REG
        self.l2_reg = L2_REG
        self.gamma = GAMMA
        self.epsilon = EPSILON[0]
        self.final_epsilon = EPSILON[1]
        self.gradient_clip = gradient_clip
        self.scale = scale
        self.state_mean, self.state_scale = .5*(state_space.low+state_space.high), .5*(state_space.high-state_space.low)
        self.rollout_buffer = RolloutBuffer(norm_len)

        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.pg = policy_network(self.state_dim[0], self.action_dim, self.layer1_size)
        self.vn = value_network(self.state_dim[0], self.layer1_size)
        
        # exploration for collecting training trajectories
        state = tf.placeholder(tf.float32, [None] + self.state_dim, 'state')
        action_logits = get_policy(state, self.pg)
        action_test = tf.argmax(tf.nn.softmax(action_logits), 1)
        # tf.multinomial may not be good, so disabled
        exploration = tf.multinomial(action_logits - tf.reduce_max(action_logits, 1, keep_dims=True), 1)

        # policy gradient update
        value = get_value(state, self.vn)
        discounted_reward = tf.placeholder(tf.float32, [None], name='discounted_reward')
        action_train = tf.placeholder(tf.int32, [None], 'action')
        # policy loss as cross entropy/ removed value
        pg_ce_loss = tf.reduce_mean((discounted_reward-value) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=action_train))
        # baseline regression loss as MSE
        #value_loss = self.val_reg * tf.reduce_mean(tf.square(discounted_reward - value))
        # L2 regularization
        pg_reg_loss = tf.add_n([self.l2_reg * tf.nn.l2_loss(var) for var in self.pg])
        # action_log_prob = tf.nn.log_softmax(action_logits)
        # entropy_loss = -self.en_reg * tf.reduce_sum(action_log_prob*tf.exp(action_log_prob))
        pg_loss = pg_ce_loss + pg_reg_loss
        pg_optim = tf.train.RMSPropOptimizer(learning_rate=ETA_ACTOR, epsilon=EPS)
        pg_grads = tf.gradients(pg_loss, self.pg)
        pg_grads, _ =  tf.clip_by_global_norm(pg_grads, self.gradient_clip) 
        train_pg = pg_optim.apply_gradients(list(zip(pg_grads, self.pg)))
        pg_grad_norms = tf.sqrt(tf.add_n([tf.nn.l2_loss(var)*2 for var in pg_grads]))

        # value network update
        vn_mse_loss = tf.reduce_mean(tf.square(discounted_reward - value))
        vn_reg_loss = tf.add_n([self.l2_reg * tf.nn.l2_loss(var) for var in self.vn])
        vn_loss = vn_mse_loss + vn_reg_loss
        vn_optim = tf.train.RMSPropOptimizer(learning_rate=ETA_CRITIC, epsilon=EPS)
        vn_grads = tf.gradients(vn_loss, self.vn)
        vn_grads, _ = tf.clip_by_global_norm(vn_grads, self.gradient_clip) 
        train_vn = vn_optim.apply_gradients(list(zip(vn_grads, self.vn)))
        vn_grad_norms = tf.sqrt(tf.add_n([tf.nn.l2_loss(var)*2 for var in vn_grads]))

        # tf functions
        with self.sess.as_default():
            self._act = TFFun(state,action_test)
            self._explore = TFFun(state,exploration)
            self._train = TFFun([state,action_train,discounted_reward],[train_pg,train_vn])
            self._value = TFFun(state,value)
            self._act_logit = TFFun(state,action_logits)
            # self._grad_norm = TFFun([state,action_train,discounted_reward],[pg_grad_norms,vn_grad_norms])

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

    def print_status(self):
        print 'current weights in policy network:\n'
        pg_wts = []
        vn_wts = []
        for g in self.pg:
            gval = np.array(g.eval())
            gavg = np.mean(np.abs(gval))
            pg_wts.append(gavg)
            print gavg
        print 'current weights in value network:\n'
        for g in self.vn:
            gval = np.array(g.eval())
            gavg = np.mean(np.abs(gval))
            vn_wts.append(gavg)
            print gavg
        return pg_wts, vn_wts

    # def learn(self):
    #     state, action, discounted_reward = self.rollout_buffer.rollout(self.gamma)
    #     self.rollout_buffer.erase()
    #     self._train(state, action, discounted_reward)
    #     print 'current grad norms\n'
    #     pg_norm, vn_norm = self._grad_norm(state, action, discounted_reward)
    #     print pg_norm, vn_norm


class MultiHeadActorCritic(PolicyGradient):
    """
    Actor critic (there are multiple heads of NN for different bootstraped training data) with REINFORCE. state: continuous, action: discrete.
    """
    def __init__(self, state_space, action_space, NUM_HEAD=5, GAMMA=0.99, ETA_ACTOR=0.001, ETA_CRITIC=0.005, EPS=1e-9, LAYER1_SIZE=20, L2_REG=0.001, EPSILON=[0.5,0], norm_len=1000000, gradient_clip=40, scale=True, head_filter=None):
        self.name = 'MAC'
        self.state_dim = list(state_space.shape)
        self.action_dim = action_space.n
        self.layer1_size = LAYER1_SIZE
        #self.val_reg = VAL_REG
        self.l2_reg = L2_REG
        self.gamma = GAMMA
        self.epsilon = EPSILON[0]
        self.final_epsilon = EPSILON[1]
        self.gradient_clip = gradient_clip
        self.scale = scale
        self.state_mean, self.state_scale = .5*(state_space.low+state_space.high), .5*(state_space.high-state_space.low)
        self.rollout_buffer = RolloutBuffer(norm_len)
        self.K = NUM_HEAD
        self.head_filter = head_filter # to do: use cell/day to filter out heads
        print 'create Actor Critic with %d differnet heads...' % NUM_HEAD
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.pg_optim = tf.train.RMSPropOptimizer(learning_rate=ETA_ACTOR, epsilon=EPS)
        self.vn_optim = tf.train.RMSPropOptimizer(learning_rate=ETA_CRITIC, epsilon=EPS)
        self.pg = policy_network_khead(self.state_dim[0], self.action_dim, self.layer1_size, self.K)
        self.vn = value_network_khead(self.state_dim[0], self.layer1_size, self.K)
        self.k_head = tf.Variable(tf.constant([1]*self.K), dtype=tf.int32, trainable=False)
        self.sess.run(tf.variables_initializer([self.k_head]))
        self.init_graph(reset=True)
        self.saver = tf.train.Saver()

    def init_graph(self, reset=False, k_head=None):      
        # input variables
        if k_head is None:
            k_head = [1]*self.K
        self.sess.run(self.k_head.assign(k_head))
        state = tf.placeholder(tf.float32, [None] + self.state_dim, 'state')
        discounted_reward = tf.placeholder(tf.float32, [None], name='discounted_reward')
        action_train = tf.placeholder(tf.int32, [None], 'action')
        #ks = tf.placeholder(tf.int32, [None] + self.K_pick, name='head_filter') # the filter for heads
        
        action_logits_list = get_policy_khead(state, self.pg, self.k_head)
        action_test_list = [tf.argmax(tf.nn.softmax(action_logits), 1) for action_logits in action_logits_list]
        # tf.multinomial may not be good, so disabled
        exploration_list = [tf.multinomial(action_logits - tf.reduce_max(action_logits, 1, keep_dims=True), 1) for action_logits in action_logits_list]

        # policy gradient update
        value_list = get_value_khead(state, self.vn, self.k_head)
        n_head = len(action_logits_list)

        #print 'list dim: %d, %d' % (len(action_logits_list), len(value_list))
        
        # policy loss as cross entropy/ removed value
        pg_ce_loss_list = [tf.reduce_mean((discounted_reward-value_list[i]) * tf.nn.sparse_softmax_cross_entropy_with_logits(action_logits_list[i], action_train)) for i in xrange(n_head)]
        # baseline regression loss as MSE
        #value_loss = self.val_reg * tf.reduce_mean(tf.square(discounted_reward - value))
        # get the current active heads
        pg_list = []
        vn_list = []
        for i, mask in enumerate(list(self.k_head.eval())):
            if mask:
                pg_list.append(self.pg[:2] + [self.pg[2*i+2],self.pg[2*i+3]])
                vn_list.append(self.vn[:2] + [self.vn[2*i+2],self.vn[2*i+3]])
        # L2 regularization
        pg_reg_loss_list = [tf.add_n([self.l2_reg * tf.nn.l2_loss(var) for var in cur_pg]) for cur_pg in pg_list]
        # action_log_prob = tf.nn.log_softmax(action_logits)
        # entropy_loss = -self.en_reg * tf.reduce_sum(action_log_prob*tf.exp(action_log_prob))
        
        pg_loss_list = [pg_ce_loss_list[i] + pg_reg_loss_list[i] for i in xrange(n_head)]
        pg_grads_list = [tf.gradients(pg_loss_list[i], pg_list[i]) for i in xrange(n_head)] 
        pg_grads_list = [tf.clip_by_global_norm(pg_grads, self.gradient_clip)[0] for pg_grads in pg_grads_list] 
        train_pg_list = [self.pg_optim.apply_gradients(list(zip(pg_grads_list[i], pg_list[i]))) for i in xrange(n_head)]

        # value network update
        vn_mse_loss_list = [tf.reduce_mean(tf.square(discounted_reward - value)) for value in value_list]            
        vn_reg_loss_list = [tf.add_n([self.l2_reg * tf.nn.l2_loss(var) for var in cur_vn]) for cur_vn in vn_list]
        vn_loss_list = [vn_mse_loss_list[i] + vn_reg_loss_list[i] for i in xrange(n_head)]
        vn_grads_list = [tf.gradients(vn_loss_list[i], vn_list[i]) for i in xrange(n_head)]
        vn_grads_list = [tf.clip_by_global_norm(vn_grads, self.gradient_clip)[0] for vn_grads in vn_grads_list] 
        train_vn_list = [self.vn_optim.apply_gradients(list(zip(vn_grads_list[i], vn_list[i]))) for i in xrange(n_head)]

        # tf functions
        with self.sess.as_default():
            self._act = TFFun([state], action_test_list)
            self._explore = TFFun([state], exploration_list)
            self._train = TFFun([state,action_train,discounted_reward], [train_pg_list,train_vn_list])
            self._value = TFFun([state], value_list)
            self._act_logit = TFFun([state], action_logits_list)
        if reset:
            self.sess.run(tf.global_variables_initializer())
        
        # self.sess.graph.finalize()
        # self.sess.run(tf.variables_initializer([self.pg_optim, self.vn_optim]))

    def act(self, state, verbose=False):
        if self.scale:
            state = self.scale_state(state)
        actions = self._act([state])
        actions = [action[0] for action in actions]
        # print actions
        # if verbose:
        #     print 'head actions: %r' % actions
        return np.argmax(np.bincount(actions)) if not verbose else actions

    def update_epsilon(self, depsilon, filter_info=None):
        self.epsilon = max(self.final_epsilon, self.epsilon-depsilon)
        # if self.head_filter is None:
        #     cur_k = np.random.randint(self.K) # the head id for exploration
        # else:
        #     cur_k = self.head_filter[filter_info]
        # cur_head = [0] * self.K
        # cur_head[cur_k] = 1
        # #self.sess.run(self.k_head.assign(cur_head))
        # self.init_graph(k_head=cur_head)

    def update_head(self, filter_info=None, phase='test'):
        if self.head_filter is None:
            if phase == 'test':
                cur_head = [1] * self.K
            elif phase == 'explore':
                cur_k = np.random.randint(self.K)
                cur_head = [0] * self.K
                cur_head[cur_k] = 1
            else:
                cur_head = [0] * self.K
                for k in xrange(self.K):
                    if np.random.rand() >= 0.5:
                        cur_head[k] = 1
        else:
            cur_k = self.head_filter[filter_info]
            cur_head = [0] * self.K
            cur_head[cur_k] = 1
            #self.sess.run(self.k_head.assign(cur_head))
        self.init_graph(k_head=cur_head)

    def explore(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        else:
            if self.scale:
                state = self.scale_state(state)
            explores = self._explore([state])
            explores = [explore[0][0] for explore in explores]
            # print explores
            return explores[0]
            #return self.act(state)

    def store(self, state, action, reward):
        if self.scale:
            self.rollout_buffer.add(self.scale_state(state), action, reward) 
        else:
            self.rollout_buffer.add(state, action, reward)

    # def filter_head(self, filter_info=None):
    #     """ select the trianing heads """
    #     ks = [0] * self.K
    #     for k in xrange(self.K):
    #         if self.head_filter is None:
    #             if np.random.rand() >= 0.5:
    #                 ks[k] = 1 
    #         else:
    #             cur_k = self.head_filter[filter_info]
    #             ks[cur_k] = 1
    #     return ks if sum(ks) else [1] + ks[1:]

    def learn(self, filter_info=None):
        state, action, discounted_reward = self.rollout_buffer.rollout(self.gamma)
        self.rollout_buffer.erase()
        #ks = self.filter_head(filter_info)
        # ks = [0,1,1,0,1]
        #self.sess.run(self.k_head.assign(ks))
        #self.init_graph(k_head=ks)

        self._train(state, action, discounted_reward)

    def print_status(self):
        print 'selected heads for training: %r' % self.k_head.eval()
        print 'current weights in policy network:\n'
        pg_wts = []
        vn_wts = []
        i = 0
        for g in self.pg:
            gval = np.array(g.eval())
            gavg = np.mean(np.abs(gval))
            pg_wts.append(gavg)
            print '%d:%r' % (i,gavg)
            i += 1
        i = 0
        print 'current weights in value network:\n'
        for g in self.vn:
            gval = np.array(g.eval())
            gavg = np.mean(np.abs(gval))
            vn_wts.append(gavg)
            print gavg
        return pg_wts, vn_wts

class MultiHeadDDPG(DDPG):
    """
    DDPG with multiple heads.
    """
    def __init__(self, state_space, action_space, GAMMA=0.99, NUM_HEAD=2, REPLAY_BUFFER_SIZE=200000, WARMUP=10000, BATCH_SIZE=32, LAYER1_SIZE=400, LAYER2_SIZE=300, TAU=0.001, ETA_ACTOR=1e-4, ETA_CRITIC=1e-3, L2_ACTOR=0, L2_CRITIC=0.01, OU_NOISE=[0.10, 0.10], clip_action_explore='TRUE'):
        self.name = 'MDDPG' 
        self.state_dim = list(state_space.shape)[0] 
        self.action_dim = list(action_space.shape)[0]
        self.layer1_size = LAYER1_SIZE
        self.layer2_size = LAYER2_SIZE
        self.eta_actor = ETA_ACTOR
        self.eta_critic = ETA_CRITIC
        self.l2_actor = L2_ACTOR
        self.l2_critic = L2_CRITIC
        self.tau = TAU
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.warmup = WARMUP
        self.clip_action_explore = clip_action_explore
        self.num_head = NUM_HEAD

        # initialize mean and scale for scaling input to [-1,1], [-1,1] to output
        self.state_mean, self.state_scale = .5*(state_space.low+state_space.high), .5*(state_space.high-state_space.low)
        self.action_mean, self.action_scale = .5*(action_space.low+action_space.high), .5*(action_space.high-action_space.low)
        print 'action normalize:', self.action_mean, self.action_scale
        print 'state normalize:', self.state_mean, self.state_scale
        # print 'OU_NOISE:', OU_NOISE
        # print 'clip_action_explore:', clip_action_explore

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random Ornstein-Uhlenbeck process 
        self.ou_theta, self.ou_sigma = OU_NOISE
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))

        # Randomly initialize actor network and critic networks and both their target networks
        self.actor = actor_network_khead(self.state_dim, self.action_dim, self.layer1_size, self.layer2_size, self.num_head)
        self.critic = critic_network_khead(self.state_dim, self.action_dim, self.layer1_size, self.layer2_size, self.num_head)
        self.target_actor, self.update_target_actor = exponential_moving_averages(self.actor, self.tau)
        self.target_critic, self.update_target_critic = exponential_moving_averages(self.critic, self.tau)
        self.optim_p = tf.train.AdamOptimizer(learning_rate=self.eta_actor)
        self.optim_q = tf.train.AdamOptimizer(learning_rate=self.eta_critic)
        self.k_head = tf.Variable(tf.constant([1]*self.num_head), dtype=tf.int32, trainable=False)
        self.sess.run(tf.variables_initializer([self.k_head]))
        self.noise_init = tf.zeros([1]+[self.action_dim])
        self.noise_var = tf.Variable(self.noise_init)
        self.init_graph(reset=True)
        self.saver = tf.train.Saver()

    def init_graph(self, active_head=None, reset=False): 
        if active_head is None:
            k_head = [1]*self.num_head
        else:
            k_head = [0]*self.num_head
            k_head[active_head] = 1
        self.sess.run(self.k_head.assign(k_head))  
        # action and exploration     
        state = tf.placeholder(tf.float32, [None]+[self.state_dim], 'state')
        action_test = get_action_khead(state, self.actor, self.k_head)
        ou_reset = self.noise_var.assign(self.noise_init)
        noise = self.noise_var.assign_sub((self.ou_theta) * self.noise_var - tf.random_normal([self.action_dim], stddev=self.ou_sigma))
        exploration = action_test[0] + noise

        # actor update
        qvalue = get_qvalue_khead(state, action_test[0], self.critic, self.k_head)[0]
        if active_head is None:
            cur_actor = self.actor
        else:
            cur_actor = self.actor[:4] + [self.actor[2*active_head+4],self.actor[2*active_head+5]]
        loss_p = -tf.reduce_mean(qvalue, 0) + tf.add_n([self.l2_actor * tf.nn.l2_loss(var) for var in cur_actor])
        
        grad_p = self.optim_p.apply_gradients(self.optim_p.compute_gradients(loss_p, var_list=cur_actor))
        with tf.control_dependencies([grad_p]):
            train_p = tf.group(self.update_target_actor)

        # critic update
        action_train = tf.placeholder(tf.float32, [self.batch_size] + [self.action_dim], 'action')
        reward = tf.placeholder(tf.float32, [self.batch_size], 'reward')
        done = tf.placeholder(tf.bool, [self.batch_size], 'done')
        next_state = tf.placeholder(tf.float32, [self.batch_size] + [self.state_dim], 'next_state')

        # calculate one-step ahead TQ
        q_train = get_qvalue_khead(state, action_train, self.critic, self.k_head)[0]
        next_action = get_action_khead(next_state, self.target_actor, self.k_head)[0]
        next_q = get_qvalue_khead(next_state, next_action, self.target_critic, self.k_head)[0]
        tq_train = tf.stop_gradient(tf.select(done,reward,reward + self.gamma*next_q))
        if active_head is None:
            cur_critic = self.critic
        else:
            cur_critic = self.critic[:4] + [self.critic[2*active_head+4],self.critic[2*active_head+5]]
        loss_q = tf.reduce_mean(tf.square(q_train - tq_train), 0) + tf.add_n([self.l2_critic * tf.nn.l2_loss(var) for var in cur_critic])
        
        grad_q = self.optim_q.apply_gradients(self.optim_q.compute_gradients(loss_q, var_list=cur_critic))
        with tf.control_dependencies([grad_q]):
            train_q = tf.group(self.update_target_critic)

        # tf functions
        with self.sess.as_default():
            self._act = TFFun([state],action_test)
            self._explore = TFFun([state],exploration)
            self._reset = TFFun([],ou_reset)
            self._train = TFFun([state,action_train,reward,done,next_state],[train_p,train_q])
            self._train_q = TFFun([state,action_train,reward,done,next_state],[train_q])
        if reset:
            self.sess.run(tf.global_variables_initializer())
        #self.sess.graph.finalize()

    def train(self):
        # Sample a random minibatch of M transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_batch(self.batch_size)
        
        # for action_dim = 1
        action_batch = np.resize(action_batch,[self.batch_size,self.action_dim])

        if self.replay_buffer.count() > self.warmup:
            self._train(state_batch,action_batch,reward_batch,done_batch,next_state_batch)
        else:
            self._train_q(state_batch,action_batch,reward_batch,done_batch,next_state_batch)

    def explore(self, state):
        action = self._explore([self.scale_state(state)])[0]
        # could be beyond limits
        unbounded_action_with_noise = self.unscale_action(action)

        if(self.clip_action_explore == 'TRUE'):
            MIN_ACTION = self.action_mean - self.action_scale
            MAX_ACTION = self.action_mean + self.action_scale
            # print MIN_ACTION, MAX_ACTION, unbounded_action_with_noise
            bounded_action_with_noise = np.minimum(np.maximum(MIN_ACTION, unbounded_action_with_noise), MAX_ACTION)
        else:
            bounded_action_with_noise = unbounded_action_with_noise      
        return bounded_action_with_noise

    def act(self, state, verbose=False):
        actions = self._act([self.scale_state(state)])
        #print('tchu action ', action)
        #print('tchu scale ', self.unscale_action(action))
        if not verbose:
            return self.unscale_action(actions[0][0])
        else:
            return [self.unscale_action(action[0]) for action in actions]

    def learn(self, state, action, reward, next_state, done):
        self.replay_buffer.add(self.scale_state(state),self.scale_action(action),reward,self.scale_state(next_state),done)
        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >= 5*self.batch_size:
            self.train()  


class MultiHeadPPCDDPG(MultiHeadDDPG):
    """
    DDPG for job simulator (there are multiple heads, and separate input hidden layer). state: continuous, action: continuous.
    """
    def __init__(self, state_split_point, state_space, action_space, GAMMA=0.99, NUM_HEAD=5, REPLAY_BUFFER_SIZE=200000, WARMUP=10000, BATCH_SIZE=32, LAYER1_SIZE=200, LAYER2_SIZE=300, TAU=0.001, ETA_ACTOR=1e-4, ETA_CRITIC=1e-3, L2_ACTOR=0, L2_CRITIC=0.01, OU_NOISE=[0.10, 0.10], clip_action_explore='TRUE', head_filter=None):
        self.name = 'MPDDPG' 
        self.state_dim0 = state_split_point
        self.state_dim1 = list(state_space.shape)[0] - self.state_dim0
        self.action_dim = list(action_space.shape)[0]
        self.layer1_size = LAYER1_SIZE
        self.layer2_size = LAYER2_SIZE
        self.eta_actor = ETA_ACTOR
        self.eta_critic = ETA_CRITIC
        self.l2_actor = L2_ACTOR
        self.l2_critic = L2_CRITIC
        self.tau = TAU
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.warmup = WARMUP
        self.clip_action_explore = clip_action_explore
        self.num_head = NUM_HEAD
        if head_filter is None:
            head_filter = {('cell1','day1'):0,('cell2','day1'):1,('cell3','day1'):2,('cell1','day2'):0}
        self.head_filter = head_filter

        # initialize mean and scale for scaling input to [-1,1], [-1,1] to output
        self.state_mean, self.state_scale = .5*(state_space.low+state_space.high), .5*(state_space.high-state_space.low)
        self.action_mean, self.action_scale = .5*(action_space.low+action_space.high), .5*(action_space.high-action_space.low)
        print 'action normalize:', self.action_mean, self.action_scale
        print 'state normalize:', self.state_mean, self.state_scale
        # print 'OU_NOISE:', OU_NOISE
        # print 'clip_action_explore:', clip_action_explore

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random Ornstein-Uhlenbeck process 
        self.ou_theta, self.ou_sigma = OU_NOISE
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))

        # Randomly initialize actor network and critic networks and both their target networks
        self.actor = ppc_actor_network_khead(self.state_dim0, self.state_dim1, self.action_dim, self.layer1_size, self.layer2_size, self.num_head)
        self.critic = ppc_critic_network_khead(self.state_dim0, self.state_dim1, self.action_dim, self.layer1_size, self.layer2_size, self.num_head)
        self.target_actor, self.update_target_actor = exponential_moving_averages(self.actor, self.tau)
        self.target_critic, self.update_target_critic = exponential_moving_averages(self.critic, self.tau)
        self.optim_p = tf.train.AdamOptimizer(learning_rate=self.eta_actor)
        self.optim_q = tf.train.AdamOptimizer(learning_rate=self.eta_critic)
        self.k_head = tf.Variable(tf.constant([1]*self.num_head), dtype=tf.int32, trainable=False)
        self.sess.run(tf.variables_initializer([self.k_head]))
        self.noise_init = tf.zeros([1]+[self.action_dim])
        self.noise_var = tf.Variable(self.noise_init)
        self.init_graph(reset=True)
        self.saver = tf.train.Saver()

    def init_graph(self, active_head=None, reset=False): 
        if active_head is None:
            k_head = [1]*self.num_head
        else:
            k_head = [0]*self.num_head
            k_head[active_head] = 1
        self.sess.run(self.k_head.assign(k_head))  
        # action and exploration     
        state0 = tf.placeholder(tf.float32, [None]+[self.state_dim0], 'state0')
        state1 = tf.placeholder(tf.float32, [None]+[self.state_dim1], 'state1')
        action_test = get_ppc_action_khead(state0, state1, self.actor, self.k_head)
        ou_reset = self.noise_var.assign(self.noise_init)
        noise = self.noise_var.assign_sub((self.ou_theta) * self.noise_var - tf.random_normal([self.action_dim], stddev=self.ou_sigma))
        exploration = action_test[0] + noise

        # actor update
        qvalue = get_ppc_qvalue_khead(state0, state1, action_test[0], self.critic, self.k_head)[0]
        if active_head is None:
            cur_actor = self.actor
        else:
            cur_actor = self.actor[:6] + [self.actor[2*active_head+6],self.actor[2*active_head+7]]
        loss_p = -tf.reduce_mean(qvalue, 0) + tf.add_n([self.l2_actor * tf.nn.l2_loss(var) for var in cur_actor])
        
        grad_p = self.optim_p.apply_gradients(self.optim_p.compute_gradients(loss_p, var_list=cur_actor))
        with tf.control_dependencies([grad_p]):
            train_p = tf.group(self.update_target_actor)

        # critic update
        action_train = tf.placeholder(tf.float32, [self.batch_size] + [self.action_dim], 'action')
        reward = tf.placeholder(tf.float32, [self.batch_size], 'reward')
        done = tf.placeholder(tf.bool, [self.batch_size], 'done')
        next_state0 = tf.placeholder(tf.float32, [self.batch_size] + [self.state_dim0], 'next_state0')
        next_state1 = tf.placeholder(tf.float32, [self.batch_size] + [self.state_dim1], 'next_state1')
        # calculate one-step ahead TQ
        q_train = get_ppc_qvalue_khead(state0, state1, action_train, self.critic, self.k_head)[0]
        next_action = get_ppc_action_khead(next_state0, next_state1, self.target_actor, self.k_head)[0]
        next_q = get_ppc_qvalue_khead(next_state0, next_state1, next_action, self.target_critic, self.k_head)[0]
        tq_train = tf.stop_gradient(tf.select(done,reward,reward + self.gamma*next_q))
        if active_head is None:
            cur_critic = self.critic
        else:
            cur_critic = self.critic[:6] + [self.critic[2*active_head+6],self.critic[2*active_head+7]]
        loss_q = tf.reduce_mean(tf.square(q_train - tq_train), 0) + tf.add_n([self.l2_critic * tf.nn.l2_loss(var) for var in cur_critic])
        
        grad_q = self.optim_q.apply_gradients(self.optim_q.compute_gradients(loss_q, var_list=cur_critic))
        with tf.control_dependencies([grad_q]):
            train_q = tf.group(self.update_target_critic)

        # tf functions
        with self.sess.as_default():
            self._act = TFFun([state0,state1],action_test)
            self._explore = TFFun([state0,state1],exploration)
            self._reset = TFFun([],ou_reset)
            self._train = TFFun([state0,state1,action_train,reward,done,next_state0,next_state1],[train_p,train_q])
            self._train_q = TFFun([state0,state1,action_train,reward,done,next_state0,next_state1],[train_q])
        if reset:
            self.sess.run(tf.global_variables_initializer())
        #self.sess.graph.finalize()

    def train(self):
        # Sample a random minibatch of M transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample_batch(self.batch_size)
        
        # for action_dim = 1
        action_batch = np.resize(action_batch,[self.batch_size,self.action_dim])
        state0_batch = state_batch[:,:self.state_dim0]
        state1_batch = state_batch[:,self.state_dim0:]
        next_state0_batch = next_state_batch[:,:self.state_dim0]
        next_state1_batch = next_state_batch[:,self.state_dim0:]

        if self.replay_buffer.count() > self.warmup:
            self._train(state0_batch,state1_batch,action_batch,reward_batch,done_batch,next_state0_batch,next_state1_batch)
        else:
            self._train_q(state0_batch,state1_batch,action_batch,reward_batch,done_batch,next_state0_batch,next_state1_batch)

    def explore(self, state):
        state = self.scale_state(state)
        action = self._explore([state[:self.state_dim0]],[state[self.state_dim0:]])[0]
        # could be beyond limits
        unbounded_action_with_noise = self.unscale_action(action)

        if(self.clip_action_explore == 'TRUE'):
            MIN_ACTION = self.action_mean - self.action_scale
            MAX_ACTION = self.action_mean + self.action_scale
            # print MIN_ACTION, MAX_ACTION, unbounded_action_with_noise
            bounded_action_with_noise = np.minimum(np.maximum(MIN_ACTION, unbounded_action_with_noise), MAX_ACTION)
        else:
            bounded_action_with_noise = unbounded_action_with_noise      
        return bounded_action_with_noise

    def act(self, state, verbose=False):
        state = self.scale_state(state)
        actions = self._act([state[:self.state_dim0]],[state[self.state_dim0:]])
        #print('tchu action ', action)
        #print('tchu scale ', self.unscale_action(action))
        if not verbose:
            return self.unscale_action(actions[0][0])
        else:
            return [self.unscale_action(action[0]) for action in actions]           



    
