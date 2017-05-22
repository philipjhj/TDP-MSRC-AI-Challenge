# Documentation

## Idea
There were many interesting aspects of the challenge, such as reinforcement learning of the entire game and 
intention-detection of the other agent. Our group decided to focus on intention-detection, as opposed to learning the 
actual game. This allowed for us to write specialized methods designed specifically for the rules of Pig Chase, such 
as traditional search algorithms and basic decisions of actions given a wanted outcome. A learning system is then 
applied attempting to discover the intentions of the other agents and deciding a wanted outcome from there.

## AI-methods and basic planning

#### Action-planning
Our system uses a multi-target A-star algorithm for determining shortest paths in the game graph, given a set of 
possible actions. The implementation searches for paths to multiple targets at the same time, outputting the shortest 
path to each. It uses the Manhattan distance heuristic. This allows for our system to determine fastest paths between
both agents and all possible targets (neighbouring cells to the pig and both exits). 

#### Domain specific improvements

Some properties of the game were noticed through the design process, which motivated some basic planning decisions:
* The pig can be caught by a single agent, if it stands on an exit.  
  We therefore made the agent detect such a situation and catch the pig on his own.
* The pig can not be caught if it stands on a squared with more than two accessible neighbouring squares.
  Our system wait with all actions until the pig moves to a squared where it can be caught - then the agent moves. 
* The timer can run out.
  If waiting for the pig takes too long, we find the path to the nearest exit.

All these concepts are simple improvements applicable to an A-star-powered agent and could be used as is (without any 
machine learning etc.). 

#### Environment

A few problems were noticed and handled (although due to a bit of time-pressure we didn't create any issues). 
These problems had to be handled in order for the system to run smoothly. Examples are:
1. Y-coordinate of agents given by the environment is always 4. Just 4. No matter where you are - it's 4.  
  Using the state-matrix we corrected this error.
1. No timer given from the environment.  
  We found a bit of a hack-solution but it would be nice to know the time of the game.
1. Server looses connection once in a while.
  Specifically we had problems if the timer ran out.  
  The connection-failure resulted in the time-out noted in 
  [Issue #30](https://github.com/Microsoft/malmo-challenge/issues/30).
  
  
## Decision making
Using the above described planning systems, the agent simply needs to make one decision; is the challenging agent 
cooperative or not. All actions can from there be computed easily. For this task we made two modes of the agent. 

#### Mode 1: Guided Danish Puppet
The agent uses the traditional AI-systems and a simple, manually made heuristic to determine whether the challenging
agent is cooperative or not. It is something along the lines: if the challenger moves towards the pig with 60% of his 
actions or more, then he is cooperative. Otherwise he is not. If he is cooperative then we help him, otherwise we
exit the game. 

#### Mode 2: Stringless Danish Puppet
We wanted to make a system that could infer the intentions of another agent. Also we thought it would be interesting
to make the system unsupervised, so that it gets to know the agents, but not by maximizing its utility. Of cause
this implies that the solution may become very suboptimal with respect to the points of the game.  

Our agent uses a Hidden Markov Model, to model the observed reactions from the challenging agent. The observations are:
1. The helmet color (unknown-helmet, helmet-1 or helmet-2)
1. Sign of difference in distance to the pig (takes values -1, 0 or 1)
1. Sign of difference in distance to nearest exit (takes values -1, 0 or 1)  

By Observing these features in a range of games the model should be able to learn different intention of agents.  

In our decision-step we determine the probability-distribution over the states of the Markov model after seeing the 
past sequence. We then compute the marginal distribution of moving towards the pig, and not moving towards the pig 
(away from or not changing distance to pig), in the next step. These two probabilities are weighted with the 
prizes of the game to make a decision of whether the challenging agent is cooperative or not.  

Due to the limitations of the HMM-implementation, all three observable variables are combined into one observable
variable, with a total of 27 possible values.

## Other Notes

* We realized that the MineCraft server allows for the agents to strafe, and we allowed our agent to take advantage of this.
* Since the pig might be in a position where it is impossible to catch, we implemented the option for our agent to wait until it moved to a better position

## Improvements and future work

* Use the standstil and bad_guy (heading directly for the exit) agents from the implemented ChallengerFactory
* Design better features and generate better training data for the HMM
