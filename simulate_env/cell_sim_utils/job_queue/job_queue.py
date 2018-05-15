import numpy as np
from os import path
import sys, os
import time
import random
from collections import OrderedDict
import copy


class job_queue_sim():

    def __init__(self):
        self.ppc_queue = []    

    def generate_PPC_queue_dictionary(self, num_PPC_clients = None, job_mean = None, job_std = None, job_deadline = None, preset_job=None):

        """
        Create PPC Job Queue according to mean and standard deviation of log-normal distribution
        Currently all jobs have the same deadline
        Input:
        number of PPC clients,
        mean value of PPC job in Bytes,
        standard deviation of PPC job in Bytes,
        time of job_deadline in float format in seconds since 1970,
        optional preset number of bytes for each job. It will override the random function
        Output:
        Returns maximum possible value for dealine and PPC bytes
        None. Changes are made in ppc_queue
        """

        # use ordered dictionary to keep items in order
        ppc_queue_user = OrderedDict()
        for i in range(num_PPC_clients):
            ppc_queue_user['deadline'] = job_deadline
            if preset_job == None:
                ppc_queue_user['ppc_bytes'] = int(random.lognormvariate(job_mean,job_std))
            else:
                ppc_queue_user['ppc_bytes'] = preset_job[i]
            ppc_queue_user['remaining_bytes'] = ppc_queue_user['ppc_bytes']
            self.ppc_queue.append(copy.deepcopy(ppc_queue_user))
        #print self.ppc_queue
        # shall we use the real timestamp? use T instead?
        ts = time.time()
        return self.generate_PPC_state(ts)


    def update_PPC_queue_per_timestep(self, time, selected_user_i, bandwidth_i, dt_i, activity_factor_i):

        """
        Update PPC queue according to time and selected user information
        Current all activities are updated at the same time
        Input:
        time: the time of this timestep in float seconds since 1970
        selected_user_i: a list of selected user, eg. [1 2 4 6]
        bandwidth_i: a list of bandwidth for each user, eg. [10 1 5 3]
        dt_i: a list of time slots for each user, eg. [1 2 3 4]
        activity_factor_i: a list of activity factor for each user, eg. [0.1 0.5 0.1 0.2]
        Output:
        None. Changes are made in ppc_queue
        """

        ppc_history = OrderedDict()
        # Test if selected_user_i is a int. Covert int to list
        if type(selected_user_i) == int:
            selected_user_i = [selected_user_i]
            bandwidth_i = [bandwidth_i]
            dt_i = [dt_i]
            activity_factor_i = [activity_factor_i]
        for i in range(len(selected_user_i)):
            user_id = selected_user_i[i]
            if 'tx_round' in self.ppc_queue[user_id]:
                self.ppc_queue[user_id]['tx_round'] +=1
            else:
                self.ppc_queue[user_id]['tx_round'] = 1
            # add suffix to each round. otherwise same key will be overwritten
            suffix = self.ppc_queue[user_id]['tx_round'];
            suffix = '_r_'+str(suffix)
            ppc_history['tx_time'+suffix] = time
            bytes_tx = int(bandwidth_i[i]*dt_i[i]*activity_factor_i[i])
            ppc_history['tx_bytes'+suffix] = bytes_tx
            self.ppc_queue[user_id].update(copy.deepcopy(ppc_history))
            self.ppc_queue[user_id]['remaining_bytes'] -= bytes_tx
            # prevent negative ppc bytes values
            if self.ppc_queue[user_id]['remaining_bytes'] < 0:
               self.ppc_queue[user_id]['remaining_bytes'] = 0 
        #print self.ppc_queue


    def generate_jobset_reward(self, reward_time):

        """
        Calculate rewards before a given time. Print warning if time is later than deadline.
        Input: 
        time: may be earlier or later than deadline.
        Output:
        reward: boolean values of reward/no reward.
        """

        finished = []
        for i in range(len(self.ppc_queue)):
            # Check if reward time exceed deadline
            if i==0 and reward_time > self.ppc_queue[i]['deadline']:
                print "Warning! Reward time exceed deadline!"
            if 'tx_round' in self.ppc_queue[i]:
                # if user i has transmitted, sum all tx_round before deadline
                tx_round  = self.ppc_queue[i]['tx_round']
                tx_bytes = 0
                # iterate all tx_rounds
                for k in range(tx_round):
                    if reward_time>=self.ppc_queue[i]['tx_time_r_' + str(k+1)]:
                        tx_bytes += self.ppc_queue[i]['tx_bytes_r_' + str(k+1)]
                # if transmited more ppc bytes than needed
                if tx_bytes >= self.ppc_queue[i]['ppc_bytes']:
                    finished.append(self.ppc_queue[i]['ppc_bytes'])
                else:
                    finished.append(0)
            else:
                # if user i hasn't transmitted, set reward to 0
                finished.append(0)
        return finished


    def generate_PPC_jobset(self, time):
    
        """
        Generate PPC jobset at the given time
        Input:
        time: current time. Will be used to calculate remaining time of each user
        Output:
        PPC_client_time_bytes_np: remaining time and PPC bytes of each user in numpy array
        """

        PPC_client_time_bytes = [[],[]]
        for i in range(len(self.ppc_queue)):
            PPC_client_time_bytes[0].append( self.ppc_queue[i]['deadline'] - time )
            PPC_client_time_bytes[1].append( self.ppc_queue[i]['remaining_bytes'])
        PPC_client_time_bytes_np = np.array(PPC_client_time_bytes)
        return PPC_client_time_bytes_np


    def generate_PPC_state(self, time):

        """
        Generate PPC states at the given time, similar to generate_PPC_jobset, but output
        is interleaved to match the input of openAI.
        Input:
        time: current time. Will be used to calculate remaining time of each user
        Output:
        PPC_client_time_bytes_np: remaining time and PPC bytes of each user in numpy array,
        arranged in [size_1, time_1, size_2, time_2 ...]
        """

        PPC_client_state = np.array([])
        for i in range(len(self.ppc_queue)):
            PPC_client_state = np.append(PPC_client_state, self.ppc_queue[i]['deadline'] - time)
            PPC_client_state = np.append(PPC_client_state, self.ppc_queue[i]['remaining_bytes'])
        return PPC_client_state

       

