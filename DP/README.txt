This code uses dynamic programming (DP) to compute an optimal control strategy
for a finite horizon problem by constructing a sequence of Q tables: S x A x T

The network congestion trace for horizon T is fully known ahead of time and we compute the 
optimal action value tables using discretized states S, discretized actions A, and full 
knowledge of Markov Decision Process (MDP) dynamics.

NOTE: PPC (pre-positioned content) is the same as IOT and HVFT for the purposes of this paper.

A. code outline
#######################################################
1. discretization_utils.py
    - generic code to discretize a system

2. Q_DP_utils.py
    helper utils for doing DP with discretized states, actions, finite horizon T
        - operate on a Q table of Q: S x A x T
        - compute optimal Q table

3. IOT_DP_utils.py
    - env specific utils for trans probs and reward

    - key fncs:
        - where x is PPC for e.g.

        - x_env_get_reward_vector()
        - x_env_get_single_transition_reward()
        - x_env_get_trans_prob_vector(): 

4. PPC_DP_wrapper_test.py
    - setup env for PPC
    - create problem_params_dict etc with info on env.
    - called by test_DP.sh 

MAIN WRAPPER TO RUN A TEST OF DP:
5. test_DP.sh
    - run DP code for IOT example, plot results
    - calls PPC_DP_wrapper_test.py

B. running on a novel env  
#######################################################
how to run DP with quantized Q table on a new env, call it x

1. modify test_DP.sh and write a new x_wrapper_test.py which creates the following dicts:
    - problem_params_dict
    - trans_prob_params
    - reward_params_dict
    - see section C

2. create x_DP_utils.py like IOT_DP_utils.py
    - problem specific reward/transition logic

C. useful dictionary structure
#######################################################

1. problem_params_dict: 
    - info on discretization

    - fields:
        - state_space_dim: scalar
        - action_space_dim: scalar
        - states: list of discretized states
        - actions: list of discretized actions
        - base_results_dir: path for where to store Q table as h5 file
        - problem_number: naming files
        - cell_day: just to delineate and name files
        - GAMMA: discount factor

    - output files:
        - Qtable.{problem_number}.{cell_day}.h5

2. trans_prob_params:
    - fields:

        - stateIndex_to_state: dict mapping to discretized state
        - actionIndex_to_action: dict mapping to discretized action value
        - states: list of states

    - for PPC case, not needed for other envs

        - master_cell_records: df of timeseries
        - congestion_var: str
        - time_index_var: str
        - numerical_tolerance: float
        - discretized_state_values: since master_cell_records has continuous congestion values
        - epsilon: for state trans randomness params
        - M: IOT multiplication factor

3. reward_params_dict:
    For IOT app case:
        - whether to use RF to predict B from C_E
        - CE_mode: str of 'INSTANTANEOUS' or not (PPC specific)
        - RF_mode: bool of False, uses B = 1/C model for thpt 
        - RF_features etc.

