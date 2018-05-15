import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16

def plot_train_wts(wts, fig_path='./figs/train_wt', xlabel='Training episodes', ylabel='Average weight magnitudes', x=None):
	fig = plt.figure()
	if x is None:
		plt.plot(wts, linewidth=2)
	else:
		plt.plot(x, wts, linewidth=2)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	fig.savefig(fig_path)
	plt.close()

def plot_train_rewards(rmeans, rstds, fig_path='./figs/train_reward', xlabel='Training episodes', ylabel='Total rewards', x=None):
	if x is None:
		x = range(rmeans.shape[0])
	fig = plt.figure()
	plt.plot(x, rmeans, linewidth=2)
	plt.fill_between(x, rmeans-rstds, rmeans+rstds, edgecolor='none', alpha=0.4)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	fig.savefig(fig_path)
	plt.close()

def train(algo, env, MAX_STEP=500, EPISODES=2000, TEST_EPISODES=20, TEST_TERM=50, SAVE_TERM=200):
	test_num = EPISODES/TEST_TERM
	rewards = np.zeros((2, test_num))
	print 'train %s...' % algo.name
	for i in xrange(EPISODES):
		state = env.reset()
		# train
		for t in xrange(MAX_STEP):
			action = algo.explore(state)
			next_state, reward, done, _ = env.step(action)
			algo.learn(state, action, reward, next_state, done)
			state = next_state
			if done:
				break
		# test
		if (i+1) % TEST_TERM == 0:
			cur_rewards = np.zeros(TEST_EPISODES)
			for j in xrange(TEST_EPISODES):
				state = env.reset()
				cur_reward = 0
				state = env.reset()
				for t in xrange(MAX_STEP):
					action = algo.act(state)
					next_state, reward, done, _ = env.step(action)
					cur_reward += reward
					if done:
						break
				cur_rewards[j] = cur_reward
			rmean = cur_rewards.mean()
			rstd = cur_rewards.std()
			print 'episode: %d, Average total Reward: %.4f, Std total Reward: %.4f' % (i+1, rmean, rstd)
			rewards[:,i/TEST_TERM] = [rmean, rstd]

		# save
		if (i+1) % SAVE_TERM == 0:
			algo.save_model((i+1))
	np.savetxt('./data/train_reward', rewards, fmt='%r')
	plot_train_rewards(rewards[:1], rewards[1:], x=range(1,EPISODES+1,TEST_TERM))

def evaluate(algos, env, MAX_STEP=500, EPISODES=100):
	pass

