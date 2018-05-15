import numpy as np
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 16
import matplotlib.pyplot as plt
import math
import pandas as pd

def plot_train_wts(wts,
                   fig_path='./figs/train_wt',
                   xlabel='Training episodes',
                   ylabel='Average weight magnitudes',
                   x=None):
    fig = plt.figure()
    if x is None:
        plt.plot(wts, linewidth=2)
    else:
        plt.plot(x, wts, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(fig_path)
    plt.close()

def plot_train_rewards(rmeans,
                       rstds,
                       fig_path='./figs/train_reward',
                       xlabel='Training episodes',
                       ylabel='Total rewards',
                       x=None):
    if x is None:
        x = range(rmeans.shape[0])
    fig = plt.figure()
    plt.plot(x, rmeans, linewidth=2)
    plt.fill_between(
        x, rmeans - rstds, rmeans + rstds, edgecolor='none', alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.savefig(fig_path)
    plt.close()

def plot_multi_train_rewards(rmeans,
                       rstds,
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
        plt.fill_between(x, rmeans[i]-rstds[i], rmeans[i]+rstds[i], edgecolor='none', alpha=0.5, color=colors[i], where=np.isfinite(rmeans[i]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    fig.savefig(fig_path)
    plt.close()

# THIS TRAIN CODE WORKS WELL
#def train(algo,
#          env,
#          MAX_STEP=500,
#          EPISODES=2000,
#          TEST_EPISODES=20,
#          TEST_TERM=50,
#          SAVE_TERM=200,
#          exp_mode='ou_noise',
#          env_name='car',
#          random_reset_during_train=False):
#    #def train(algo, env, MAX_STEP=500, EPISODES=2000, TEST_EPISODES=20, TEST_TERM=50, SAVE_TERM=200):
#    test_num = EPISODES / TEST_TERM
#    rewards = np.zeros((2, test_num + 1))
#    wts = np.zeros(EPISODES + 1)
#    if not MAX_STEP:
#        MAX_STEP = env.spec.timestep_limit
#    if exp_mode == 'epsilon_greedy':
#        epsilon = 1.0
#        depsilon = .9 * 4 / EPISODES
#    print 'train %s...' % algo.name
#    for i in xrange(EPISODES + 1):
#        # test
#        if i % TEST_TERM == 0:
#            cur_rewards = np.zeros(TEST_EPISODES)
#            for j in xrange(TEST_EPISODES):
#                env.random_reset_mode = False
#                print('test', env.random_reset_mode, ' EPISODES ', EPISODES, ' i ', i)
#                state = env.reset()
#                cur_reward = 0
#                for t in xrange(MAX_STEP):
#                    action = algo.act(state)
#                    next_state, reward, done, _ = env.step(action)
#                    cur_reward += reward
#                    if done:
#                        break
#                    state = next_state
#
#                #last_batch = env.batch_number
#                #df = env.reward_history_df
#                #late_batch_df = df[df['BATCH_NUM'] == (last_batch - 1)]
#                #print(late_batch_df['ACTION'].describe())
#                #print('reward ', cur_reward)
#                cur_rewards[j] = cur_reward
#
#            rmean = cur_rewards.mean()
#            rstd = cur_rewards.std()
#            print 'episode: %d, Average total Reward: %.4f, Std total Reward: %.4f' % (
#                i, rmean, rstd)
#            rewards[:, i / TEST_TERM] = [rmean, rstd]
#        wts[i] = algo.get_abs_wt()
#        if i == EPISODES:
#            break
#
#        # train
#        if exp_mode == 'epsilon_greedy':
#            algo.set_epsilon(epsilon)
#        elif exp_mode == 'ou_noise':
#            algo.reset_noise()
#        env.random_reset_mode = random_reset_during_train
#        state = env.reset()
#        print('train ', env.random_reset_mode, ' EPISODES ', EPISODES, ' i ', i)
#        for t in xrange(MAX_STEP):
#            # print('i ', i,'t ', t)
#            action = algo.explore(state)
#            next_state, reward, done, _ = env.step(action)
#            algo.learn(state, action, reward, next_state, done)
#            state = next_state
#            if done:
#                break
#
#        if exp_mode == 'epsilon_greedy':
#            epsilon = max(0.1, epsilon - depsilon)
#
#        # print(rewards)
#        # save
#        # np.savetxt('./data/train_reward_invertpend', rewards, fmt='%r')
#        #plot_train_rewards(rewards[0], rewards[1], x=range(0,EPISODES+1,TEST_TERM), fig_path='/Users/csandeep/figs/train_reward_'+env_name)
#        #plot_train_wts(wts, x=range(EPISODES+1), fig_path='/Users/csandeep/figs/train_wt_'+env_name)
#    return rewards, wts
#

def read_train_info(file_path):
    train_inds = []
    test_inds = []
    head_filter = {}
    df = pd.read_csv(file_path)
    # print df.columns
    for i in xrange(df.shape[0]):
        cell, day = str(df.CELL_ID[i]), str(df.DATE_LOCAL[i])
        if df.TRAIN_TEST_INDICATOR[i] == 'TEST':
            test_inds.append((cell,day))
        else:
            train_inds.append((cell,day))
        head = int(df.HEAD_ID[i])
        head_filter[(cell,day)] = head
    # print train_inds, test_inds, head_filter
    return train_inds, test_inds, head_filter

def evaluate(algos, env, MAX_STEP=500, EPISODES=10):
    K = len(algos)
    rewards = np.zeros((K,EPISODES,MAX_STEP))
    # this can also be done by calling env.reward_history_df
    eg_actions = np.zeros((K,MAX_STEP))
    eg_rewards = np.zeros((K,MAX_STEP))
    env.TRAIN_MODE = False
    env.random_reset_mode = False
    for i in xrange(EPISODES):
        for j, algo in enumerate(algos):
            # env.step_back(state0, state_dict0, reset=True)
            state = env.reset()
            for t in xrange(MAX_STEP):
                if algo.name == 'Greedy':
                    action = algo.act(env)
                else:
                    action = algo.act(state)
                if not i:
                    actions[j,t] = action
                
                next_state, reward, done, _ = env.step(action)
                # print action, reward, done, t, env.iteration_index
                rewards[j,i,t] = reward
                if done:
                    break
                state = next_state
            print 'test episode %d, algo %s, total reward: %.4f' %(i, algo.name, np.sum(rewards[j,i]))
    stat_rewards = np.zeros((K,2,MAX_STEP))
    for j in xrange(K):
        stat_rewards[j,0] = np.mean(rewards[j], axis=0)
        stat_rewards[j,1] = np.std(rewards[j], axis=0)
    return stat_rewards, actions

# here an episode is a set of days, that are ALWAYS fully cycled through
def train_daily_episode(algo, env, DDPG_dict):
    print 'daily episode train %s...' % algo.name
    MAX_STEP = DDPG_dict['MAX_STEP']
    EPISODES = DDPG_dict['EPISODES']
    TEST_EPISODES = DDPG_dict['TEST_EPISODES']
    TEST_TERM = DDPG_dict['TEST_TERM']
    SAVE_TERM = DDPG_dict['SAVE_TERM']
    PRINT_LEN = DDPG_dict['PRINT_LEN']

    test_num = EPISODES / TEST_TERM
    rewards = np.zeros((2, test_num))

    for i in xrange(EPISODES):
        # resets to first timepoint
        algo.reset_noise()
        state = env.reset()

        # execute the daily batch
        # train
        print('executing episode ', i, 'length: ', MAX_STEP)
        for t in xrange(MAX_STEP):

            action = algo.explore(state)
            next_state, reward, done, _ = env.step(action)
            algo.learn(state, action, reward, next_state, done)
            state = next_state

            if (t % PRINT_LEN == 0):
                print('episode ', i, 'step: ', t, 'state ', state, 'action ',
                      action, 'reward ', reward)
            if done:
                break

        # plot train episode i
        # panel_KPI_plot(last_batch_df, plotting_info_dict)

        print('DONE TRAIN weights: ', algo.get_abs_wt())
        # test
        if (i + 1) % TEST_TERM == 0:
            print('TEST at episode ', i, 'length: ', MAX_STEP)
            cur_rewards = np.zeros(TEST_EPISODES)
            for j in xrange(TEST_EPISODES):
                algo.reset_noise()
                state = env.reset()
                cur_reward = 0
                for t in xrange(MAX_STEP):
                    action = algo.act(state)
                    next_state, reward, done, _ = env.step(action)
                    if (t % PRINT_LEN == 0):
                        print('TEST episode ', i + 1, 'step: ', t)

                        print('state ', state, 'action: ', action)
                        print('reward: ', reward)
                        print('weights: ', algo.get_abs_wt())

                    state = next_state
                    cur_reward += reward
                    if done:
                        break
                cur_rewards[j] = cur_reward

                # plot test episode j
                # panel_KPI_plot(last_batch_df, plotting_info_dict)

            rmean = cur_rewards.mean()
            rstd = cur_rewards.std()
            print 'episode: %d, Average total Reward: %.4f, Std total Reward: %.4f' % (
                i + 1, rmean, rstd)
            rewards[:, i / TEST_TERM] = [rmean, rstd]

        print(rewards)
    return rewards


def print_train_progress_str(env = None, cur_reward = None, curr_episode = None, TOTAL_EPISODES = None):
    #last_batch = env.batch_number
    #df = env.reward_history_df
    #late_batch_df = df[df['BATCH_NUM'] == (last_batch - 1)]
    #print(late_batch_df['ACTION'].describe())
    #print('reward ', cur_reward)

    pass

def split_data_head(head_filter, train_inds, test_inds):
    num_head = max(head_filter.values()) + 1
    train_head_inds = [[] for _ in xrange(num_head)]
    test_head_inds = [[] for _ in xrange(num_head)]
    for ind in train_inds:
        active_head = head_filter[ind]
        train_head_inds[active_head].append(ind)
    for ind in test_inds:
        active_head = head_filter[ind]
        test_head_inds[active_head].append(ind)
    return num_head, train_head_inds, test_head_inds

def test_algo(algo, env, num_head, test_inds, test_head_inds, MAX_STEP, TEST_DAY_REPEAT):
    rewards = np.zeros((2, len(test_inds)))
    for test_head in xrange(num_head):
        env.rf_ind = test_head
        cur_test_inds = test_head_inds[test_head]
        # env.reward_params_dict['hard_thpt_limit'] = hard_thpt_limits[test_head]
        if algo.name == 'MDDPG':
            algo.init_graph(active_head=test_head)
        for test_ind in cur_test_inds:
            cur_rewards = np.zeros(TEST_DAY_REPEAT)
            for j in xrange(TEST_DAY_REPEAT): 
                env.ind = test_ind   
                state = env.reset() #TODO
                cur_reward = 0
                for t in xrange(MAX_STEP):
                    action = algo.act(state)
                    next_state, reward, done, _ = env.step(action)
                    cur_reward += reward
                    if done:
                        break
                    state = next_state
                cur_rewards[j] = cur_reward
            rmean = cur_rewards.mean()
            rstd = cur_rewards.std()
            print 'head: %d, test data: %r, average total reward: %.4f, std total reward: %.4f.' % (test_head, test_ind, rmean, rstd)
            ind = test_inds.index(test_ind)
            rewards[:,ind] = [rmean, rstd]
    return rewards

def train_ppc_old(algo, env, MAX_STEP=500, TRAIN_DAY_REPEAT=10, TEST_DAY_REPEAT=1, TEST_TERM=5, SAVE_TERM=50,
    env_name='cell', save_path=None, verbose_wt=False, head_filter=None, plot_path='/Users/tchu/Documents/gym_test'):
    
    train_inds = env.train_inds
    test_inds = env.test_inds
    # train_inds = [('cell1','day1'),('cell2','day1'),('cell3','day1'),('cell1','day4')]
    # test_inds = [('cell1','day2'),('cell1','day3')]
    num_train = len(train_inds)
    num_test = len(test_inds)
    EPISODES = num_train * TRAIN_DAY_REPEAT
    tot_num_test = EPISODES / TEST_TERM
    rewards = np.zeros((2, num_test, tot_num_test+1)) # mean and std rewards for each test data
    if not MAX_STEP:
        MAX_STEP = env.spec.timestep_limit
    # split the train test data according to heads
    if head_filter is None:
        head_filter = {('cell1','day1'):0,('cell2','day1'):0,('cell3','day1'):1,('cell1','day2'):0,('cell1','day3'):1,('cell1','day4'):1} 
    num_head, train_head_inds, test_head_inds = split_data_head(head_filter, train_inds, test_inds)
    # hard_thpt_limits = [10000, 4000]
    print 'train %s...' % algo.name
     
    i = 0
    for head in xrange(num_head):
        if algo.name == 'MDDPG':
            algo.init_graph(active_head=head)
        cur_train_inds = train_head_inds[head]
        #cur_test_inds = test_head_inds[head]
        print 'train head %d...' % head
        for _ in xrange(TRAIN_DAY_REPEAT):
            for train_ind in cur_train_inds:
                # test whenever necessary
                if i % TEST_TERM == 0:
                    env.TRAIN_MODE = False
                    print 'test at episode: %d' % i
                    rewards[:,:,i/TEST_TERM] = test_algo(algo, env, num_head, test_inds, test_head_inds, MAX_STEP, TEST_DAY_REPEAT)
                    if algo.name == 'MDDPG':
                        algo.init_graph(active_head=head)
                    if verbose_wt:
                        algo.print_status()
                # train one episode
                i += 1
                algo.reset_noise()
                env.TRAIN_MODE = True
                env.ind = train_ind
                env.rf_ind = head
                # env.reward_params_dict['hard_thpt_limit'] = hard_thpt_limits[head]
                state = env.reset() # TODO
                for t in xrange(MAX_STEP):
                    action = algo.explore(state)
                    next_state, reward, done, _ = env.step(action)
                    algo.learn(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break
                # save whenever necessary
                if i % SAVE_TERM == 0:
                    algo.save_model(i, save_path, env_name = env_name)
    env.TRAIN_MODE = False
    print 'test at episode: %d' % i
    rewards[:,:,-1] = test_algo(algo, env, num_head, test_inds, test_head_inds, MAX_STEP, TEST_DAY_REPEAT)
    # make sure to save the final model
    if algo.name != 'H0':
        algo.save_model(i, save_path, env_name = env_name)
        #algo.replay_buffer.save_buffer(save_path+'/replay.txt')
        plot_multi_train_rewards(rewards[0], rewards[1], x=range(0,EPISODES+1,TEST_TERM), legend=test_inds, fig_path=plot_path+'/'+env_name+'_reward.png')
    np.savetxt(plot_path+'/'+env_name+'_reward_mean', rewards[0])
    np.savetxt(plot_path+'/'+env_name+'_reward_std', rewards[1])
    return rewards

def train_ppc(algo, env, MAX_STEP=500, TRAIN_DAY_REPEAT=10, TEST_DAY_REPEAT=1, TEST_TERM=5, SAVE_TERM=50,
    env_name='cell', save_path=None, verbose_wt=False, head_filter=None, plot_path='/Users/tchu/Documents/gym_test'):
    
    train_inds = env.train_inds
    test_inds = env.test_inds
    # train_inds = [('cell1','day1'),('cell2','day1'),('cell3','day1'),('cell1','day4')]
    # test_inds = [('cell1','day2'),('cell1','day3')]
    num_train = len(train_inds)
    num_test = len(test_inds)
    EPISODES = num_train * TRAIN_DAY_REPEAT
    tot_num_test = EPISODES / TEST_TERM
    rewards = np.zeros((2, num_test, tot_num_test+1)) # mean and std rewards for each test data
    if not MAX_STEP:
        MAX_STEP = env.spec.timestep_limit
    # split the train test data according to heads
    if head_filter is None:
        head_filter = {('cell1','day1'):0,('cell2','day1'):0,('cell3','day1'):1,('cell1','day2'):0,('cell1','day3'):1,('cell1','day4'):1} 
    num_head, train_head_inds, test_head_inds = split_data_head(head_filter, train_inds, test_inds)
    # hard_thpt_limits = [10000, 4000]
    print 'train %s...' % algo.name
     
    i = 0
    for _ in xrange(TRAIN_DAY_REPEAT):
        for head in xrange(num_head):
            if algo.name == 'MDDPG':
                algo.init_graph(active_head=head)
            cur_train_inds = train_head_inds[head]
            print 'train head %d...' % head
            for train_ind in cur_train_inds:
                # test whenever necessary
                if i % TEST_TERM == 0:
                    env.TRAIN_MODE = False
                    print 'test at episode: %d' % i
                    rewards[:,:,i/TEST_TERM] = test_algo(algo, env, num_head, test_inds, test_head_inds, MAX_STEP, TEST_DAY_REPEAT)
                    if algo.name == 'MDDPG':
                        algo.init_graph(active_head=head)
                    if verbose_wt:
                        algo.print_status()
                # train one episode
                i += 1
                algo.reset_noise()
                env.TRAIN_MODE = True
                env.ind = train_ind
                env.rf_ind = head
                # env.reward_params_dict['hard_thpt_limit'] = hard_thpt_limits[head]
                state = env.reset() # TODO
                for t in xrange(MAX_STEP):
                    action = algo.explore(state)
                    next_state, reward, done, _ = env.step(action)
                    algo.learn(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break
                # save whenever necessary
                if i % SAVE_TERM == 0:
                    algo.save_model(i, save_path, env_name = env_name)
    env.TRAIN_MODE = False
    print 'test at episode: %d' % i
    rewards[:,:,-1] = test_algo(algo, env, num_head, test_inds, test_head_inds, MAX_STEP, TEST_DAY_REPEAT)
    test_algo(algo, env, num_head, train_inds, train_head_inds, MAX_STEP, TEST_DAY_REPEAT)
    if algo.name != 'H0':
        # make sure to save the final model
        algo.save_model(i, save_path, env_name = env_name)
        #algo.replay_buffer.save_buffer(save_path+'/replay.txt')
        plot_multi_train_rewards(rewards[0], rewards[1], x=range(0,EPISODES+1,TEST_TERM), legend=test_inds, fig_path=plot_path+'/'+env_name+'_reward.png')
    np.savetxt(plot_path+'/'+env_name+'_reward_mean', rewards[0])
    np.savetxt(plot_path+'/'+env_name+'_reward_std', rewards[1])
    return rewards

def train(algo,
          env,
          MAX_STEP=500,
          EPISODES=2000,
          TEST_EPISODES=20,
          EXP_EPISODES=-1,
          TEST_TERM=50,
          SAVE_TERM=200,
          exp_mode='ou_noise',
          env_name='cell',
          random_reset_during_train=False,
          save_path=None,
          verbose_train_mode=False,
          verbose_wt=False,
          plot_path='/Users/tchu/Documents/gym_test'):

    test_num = EPISODES / TEST_TERM
    rewards = np.zeros((2, test_num + 1))
    wts = np.zeros(EPISODES + 1)
    if not MAX_STEP:
        MAX_STEP = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    if exp_mode == 'epsilon_greedy':
        # exploration method
        if EXP_EPISODES < 0:
            depsilon = algo.epsilon * 2.0 / EPISODES
        else:
            depsilon = algo.epsilon * 1.0 / EXP_EPISODES
    
    print 'train %s...' % algo.name
    for i in xrange(EPISODES + 1):
        # test
        if i % TEST_TERM == 0:
            cur_rewards = np.zeros(TEST_EPISODES)
            if algo.name == 'MAC':
                algo.update_head(phase='test')
            for j in xrange(TEST_EPISODES):
                env.TRAIN_MODE = False
                state = env.reset()
                cur_reward = 0
                # verbose = True if j == 0 else False
                for t in xrange(MAX_STEP):
                    action = algo.act(state)
                    next_state, reward, done, _ = env.step(action)
                    cur_reward += reward
                    if done:
                        break
                    state = next_state

                if(verbose_train_mode):
                    print('test', env.random_reset_mode, ' EPISODES ', EPISODES, ' i ', i)
                    last_batch = env.batch_number
                    df = env.reward_history_df
                    late_batch_df = df[df['BATCH_NUM'] == (last_batch - 1)]
                    print(late_batch_df['ACTION'].describe())
                    print('reward ', cur_reward)
                cur_rewards[j] = cur_reward

            rmean = cur_rewards.mean()
            rstd = cur_rewards.std()
            print 'episode: %d, Average total Reward: %.4f, Std total Reward: %.4f' % (
                i, rmean, rstd)
            rewards[:, i / TEST_TERM] = [rmean, rstd]
            if verbose_wt:
                algo.print_status()
        wts[i] = algo.get_abs_wt()

        # print('SAVE TERM', SAVE_TERM)
        if i > 0 and i % SAVE_TERM == 0:
            # save the neural net params
            algo.save_model(i, save_path, env_name = env_name)

        if i == EPISODES:
            break

        # train
        if exp_mode == 'epsilon_greedy':
            algo.update_epsilon(depsilon)
        elif exp_mode == 'ou_noise':
            algo.reset_noise()
        if algo.name == 'MAC':
            algo.update_head(phase='explore')
        env.TRAIN_MODE = True
        state = env.reset()
        if(verbose_train_mode):
            print('train ', env.random_reset_mode, ' EPISODES ', EPISODES, ' i ', i)
        for t in xrange(MAX_STEP):
            # print('i ', i,'t ', t)
            action = algo.explore(state)
            next_state, reward, done, _ = env.step(action)
            if algo.name in ['PG','AC','MAC']:
                algo.store(state, action, reward)
            else:
                algo.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if algo.name == 'MAC':
            algo.update_head(phase='train')
        if algo.name in ['PG','AC','MAC']:
            algo.learn()
        # print(rewards)
        # save
        # np.savetxt('./data/train_reward_invertpend', rewards, fmt='%r')
    plot_train_rewards(rewards[0], rewards[1], x=range(0,EPISODES+1,TEST_TERM), fig_path=plot_path+'/'+env_name+'_reward.png')
    plot_train_wts(wts, x=range(EPISODES+1), fig_path=plot_path+'/'+env_name+'_wt.png')
    return rewards, wts

# def train_pg(algo,
#           env,
#           MAX_STEP=None,
#           EPISODES=200,
#           TRAIN_EPISODES=20,
#           TEST_EPISODES=10,
#           TEST_TERM=20,
#           SAVE_TERM=200,
#           random_reset_during_train=False,
#           env_name='car',
#           save_path=None,
#           verbose_train_mode=False):
#     """
#     For training policy gradient, we perform REINFORCE instead of experience replay.
#     """
#     test_num = EPISODES / TEST_TERM
#     train_rewards = np.zeros((2, test_num + 1))
#     wts = np.zeros(test_num + 1)
#     if not MAX_STEP:
#         MAX_STEP = env.spec.timestep_limit
    
#     print 'train %s...' % algo.name
#     i = 0
#     while i <= EPISODES:
#         # test
#         if i % TEST_TERM == 0:
#             cur_rewards = np.zeros(TEST_EPISODES)
#             for j in xrange(TEST_EPISODES):
#                 env.random_reset_mode = False
#                 env.TRAIN_MODE = False
#                 state = env.reset()
#                 cur_reward = 0
#                 for t in xrange(MAX_STEP):
#                     action = algo.act(state)
#                     next_state, reward, done, _ = env.step(action)
#                     cur_reward += reward
#                     if done:
#                         break
#                     state = next_state

#                 if(verbose_train_mode):
#                     print('test', env.random_reset_mode, ' EPISODES ', EPISODES, ' i ', i)
#                     last_batch = env.batch_number
#                     df = env.reward_history_df
#                     late_batch_df = df[df['BATCH_NUM'] == (last_batch - 1)]
#                     print(late_batch_df['ACTION'].describe())
#                     print('reward ', cur_reward)
#                 cur_rewards[j] = cur_reward

#             rmean = cur_rewards.mean()
#             rstd = cur_rewards.std()
#             print 'test episode: %d, Average total Reward: %.4f, Std total Reward: %.4f' % (
#                 i, rmean, rstd)
#             train_rewards[:,(i/TEST_TERM)] = [rmean, rstd]
#             wts[i/TEST_TERM] = algo.get_abs_wt()

#         if i > 0 and i % SAVE_TERM == 0:
#             # save the neural net params
#             algo.save_model(i, save_path, env_name = env_name)

#         if i == EPISODES:
#             break

#         # train
#         env.random_reset_mode = random_reset_during_train
#         env.TRAIN_MODE = True
#         states, actions, rewards, discounted_rewards = rollout(algo, env, MAX_STEP, TRAIN_EPISODES)
#         if(verbose_train_mode):
#             print('train ', env.random_reset_mode, ' EPISODES ', EPISODES, ' i ', i)
#         algo.learn(states, actions, discounted_rewards)
#         print 'train episode: %d, Average total Reward: %.4f, Std total Reward: %.4f' % (
#                 i, np.mean(rewards), np.std(rewards))
#         i += TRAIN_EPISODES

#         # print(rewards)
#         # save
#         # np.savetxt('./data/train_reward_invertpend', rewards, fmt='%r')
#     plot_train_rewards(train_rewards[0], train_rewards[1], x=range(0,EPISODES+1,TEST_TERM), fig_path='/Users/tchu/Documents/gym_test/'+env_name+'_reward.png')
#     plot_train_wts(wts, x=range(0,EPISODES+1,TEST_TERM), fig_path='/Users/tchu/Documents/gym_test/'+env_name+'_wt.png')
#     return rewards, wts

