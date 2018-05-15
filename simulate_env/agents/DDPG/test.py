import gym
from algorithm import PolicyGradient, DDPG, ActorCritic
from train_util import *
import os
import argparse
#continuous env_list = ['MountainCarContinuous-v0']
#discrete env_list = [MountainCar-v0', 'Acrobot-v1', 'CartPole-v0', 'CartPole-v1']
def parse_args():
    plot_dir_default = '/Users/tchu/Documents/gym_test'
    #plot_dir_default = '/Users/csandeep/Documents/work/uhana/work/DDPG/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_dir', type=str, required=False, 
                        default=plot_dir_default, help="directory of reward plot")
    parser.add_argument('--discrete_action', type=str, required=False, 
                        default='TRUE', help="if the action space is discrete or not")
    parser.add_argument('--env_name', type=str, required=False, 
                        default='CartPole-v0', help="gym environment name")
    parser.add_argument('--agent_name', type=str, required=False, 
                        default='AC', help="algorithm name: PG, DDPG, AC")
    parser.add_argument('--train_episode', type=int, required=False, 
                        default = 1000,
                        help="number of training episodes")
    parser.add_argument('--test_episode', type=int, required=False, 
                        default = 20,
                        help="number of test episodes")
    parser.add_argument('--test_term', type=int, required=False, 
                        default = 100,
                        help="test frequency in training episodes")
    parser.add_argument('--save_term', type=int, required=False, 
                        default = 20000,
                        help="model-save frequency in training episodes")
    parser.add_argument('--MAX_STEP', type=int, required=False, 
                        default = 100,
                        help="max steps per episode")
    args = parser.parse_args()
    return args
args = parse_args()
discrete_action_flag = True if args.discrete_action == 'TRUE' else False
env_name = args.env_name
print 'simulate %s' % env_name
env = gym.make(env_name)
print env.action_space
agent_name = args.agent_name
if agent_name == 'DDPG':
	if not discrete_action_flag:
		algo = DDPG(env.observation_space, env.action_space)
	else:
		algo = DDPG(env.observation_space, env.action_space, discrete=True)
	exp_mode = 'ou_noise'
else:
	if agent_name == 'PG':
		# algo = PolicyGradient(env.observation_space, env.action_space)
		algo = PolicyGradient(env.observation_space, env.action_space, ETA=0.001)
	else:
		algo = ActorCritic(env.observation_space, env.action_space, ETA_ACTOR=0.001, ETA_CRITIC=0.005)
	exp_mode = 'epsilon_greedy'
plot_path = args.plot_dir
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

train(algo, env, env_name=env_name, MAX_STEP=args.MAX_STEP, EPISODES=args.train_episode, TEST_EPISODES=args.test_episode, TEST_TERM=args.test_term, SAVE_TERM=args.save_term, exp_mode=exp_mode, plot_path = plot_path)
