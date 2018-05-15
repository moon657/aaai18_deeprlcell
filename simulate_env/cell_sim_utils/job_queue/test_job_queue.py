#import job_queue
from job_queue import job_queue_sim 
job_queue_sim = job_queue_sim()
from datetime import datetime
import time
import random

if __name__ == "__main__":

    #====Test generate PPC queue function====

    # set deadline to 10 seconds from now
    ts = time.time()
    #job_queue_sim = job_queue.job_queue_sim()
    #job_queue_sim = job_queue_sim()

    #==== Test generate PPC queue function with and without preset bytes value====
    job_queue_sim.generate_PPC_queue_dictionary(num_PPC_clients = 10, job_mean = 10, job_std = 1, job_deadline = ts+10, preset_job = range(10000,110000,10000))
    print '\n'.join(map(str,job_queue_sim.ppc_queue))

    ##print job_queue_sim.ppc_queue

    ##====Test update PPC queue function====
    #print job_queue_sim.ppc_queue[1]
    #job_queue_sim.update_PPC_queue_per_timestep(ts+2,[1,3],[100000,10000],[1,1],[0.5,0.1])
    #job_queue_sim.update_PPC_queue_per_timestep(ts+4,[2,4],[100000,10000],[1,1],[0.5,0.1])
    #print job_queue_sim.ppc_queue[1]

    ##===Test generate job reward function==== 
    ## Calculate reward for only first update
    #reward = job_queue_sim.generate_jobset_reward(ts+3)
    #print reward
    ## Calculate reward for first and second update
    #reward = job_queue_sim.generate_jobset_reward(ts+5)
    #print reward
    ## Calculate reward beyond deadline
    #reward = job_queue_sim.generate_jobset_reward(ts+11)
    #print reward 
    ## Test generated PPC jobset
    #print job_queue_sim.generate_PPC_jobset(ts+4)

    ## Test generated PPC states
    #print job_queue_sim.generate_PPC_state(ts+4)
