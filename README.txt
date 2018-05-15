# My project's README
This repository is for running RL algorithms on a cell simulator
for the AAAI 2018 paper: "Cellular Network Traffic Scheduling using Deep Reinforcement Learning".

# Setup environment variables
1. Required python2.7 packages:
Pandas, Numpy, Scipy, Matplotlib, json, openAI gym, Tensorflow, keras, seaborn

2. How to get openAI gym (from https://gym.openai.com/docs):

	git clone https://github.com/openai/gym
	cd gym
	pip install -e . 

3. Ensure $AAAI_ROOT_DIR bash variable points to your installation. Ex:
	In ~/.bashrc:
	export AAAI_ROOT_DIR=/home/<user>/aaai18_deeprlcell

	Then source ~/.bashrc

4. Code results appear in $AAAI_WORK_DIR
   As in step 3, point this to an appropriate location for results to appear on your machine


