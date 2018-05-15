
PLOT_DIR=/Users/csandeep/Documents/work/uhana/work/DDPG
rm -rf $PLOT_DIR
mkdir -p $PLOT_DIR

ENV_NAME=CartPole-v1

AGENT_NAME=AC

MAX_STEP=100

CODE_DIR=$RL_ROOT_DIR/simulate_env/agents/DDPG

python $CODE_DIR/test.py --plot_dir $PLOT_DIR --env_name $ENV_NAME --agent_name $AGENT_NAME --MAX_STEP $MAX_STEP

