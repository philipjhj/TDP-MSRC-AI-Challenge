# TDP-MSRC-AI-Challenge

This project is created for The Malmo Collaborative AI Challenge, which concerns writing an AI-agent for a game called 
Pig Chase created in Minecraft.
Turn to https://www.microsoft.com/en-us/research/academic-program/collaborative-ai-challenge/ for a description about 
the game and the challenge.

## Running the code
1. Install the challenge code as described under installation [here](https://github.com/Microsoft/malmo-challenge#installation)
1. Go into the *ai_challenge* folder in the malmo-challenge and run
```
wget https://github.com/philipjhj/TDP-MSRC-AI-Challenge/archive/master.zip
mkdir danish_puppeteers
unzip master.zip -d danish_puppeteers
mv danish_puppeteers/master/* danish_puppeteers/
rm -r danish_puppeteers/master/
cd danish_puppeteers
```
Add the *pig_chase* and *danish_puppeteers* to your python path by using the command `export PYTHONPATH=$PYTHONPATH:<path to ai_challenge>/ai_challenge/danish_puppeteers`. After this you should be able to run the evaluation script `python pig_chase_eval_sample.py` or any of the other scripts.

To run a script with docker on an Azure machine created with docker-machine, run
```
./run_azure_docker.sh <machine-name> <python-script-name-without-file-extension>
```
See more about how to run the challenge with docker [here](https://github.com/Microsoft/malmo-challenge/tree/master/docker)
## Idea

Our agent makes a decision of whether a challenging agent is cooperative or not. After making this decision all
following actions are computed using traditional AI-techniques (A-star search in game-graph using the available
actions etc.). The determining factor is thus whether the agent can correctly identify the intentions of a 
challenging agent. We have made two different modes of how it does this.

#### Mode 1: Guided Danish Puppet
The first mode uses a simple heuristic to determine whether the challenging agent is cooperative. If the challenger 
moves towards the pig with 60% of its moves, then it is assumed cooperative otherwise it is considered non-cooperative.
This solution is very specific to the target game and does not include any machine learning methods. It is basically
an improved version of the A-star agent, in which A-star is run on multiple targets 

#### Mode 2: Stringless Danish Puppet (AKA Pinocchio)
The other mode uses a Hidden Markov Model in an attempt to model the intention of the challenger. It uses a few
observable variables as input:
1. The helmet color
1. Sign of difference in distance to the pig (takes values -1, 0 or 1)
1. Sign of difference in distance to nearest exit (takes values -1, 0 or 1)  

It then computes the marginal distribution over the usefulness of the following action and uses this probability to
determine whether the challengers intentions are cooperative or non-cooperative.  

More information can be found in the 
[Documentation](https://github.com/philipjhj/TDP-MSRC-AI-Challenge/blob/master/Documentation.md)-file.
## Demo
See a [video here](https://www.youtube.com/watch?v=T8afN7Kjo3E&feature=youtu.be) demonstrating our agent in action.

## Results
Our results based on the *pig_chase_eval_sample.py* script can be seen [here](https://malmo-leaderboard.azurewebsites.net/). The experiment name matches the method used. A few details on the results are given here:
