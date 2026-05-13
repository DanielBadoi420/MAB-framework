# Environment Overview

This framework implements a simple multi-agent multi-armed bandit environment. 
 
Each timestep:
1. Each agent selects an arm according to its strategy.
2. The environment samples a reward for each arm that was chosen.
3. If multiple agents choose the same arm, a collision policy determines how the reward is split.
4. Each agent receives its reward and updates its internal state according to its strategy.

## Environment

**Parameters**
- n_agents
- list of Arm objects
- collision_policy (function)

The Environment class control the behaviour of the simulation.

## Arms
Each arm is an instance of the Arm class:

Arm(mean, sd=1.0, reward_fn=None)


An arm has:

mean – expected reward
sd – standard deviation (used by the default Gaussian reward function)
reward_fn – sampling function - defaulting to Gaussian(mean, sd)


## Reward Sharing Policies

The policies, implemented as functions, are passed to the environment.

The environment implements:
- linear_share
- zero_on_collision
- winner_takes_all


## Agents
Agents must implement choose_arm(), update(), and have a name - defaults to the class name if not passed.
Agent() class is an interface to create strategy types - see scripts of the implemented Agent subclasses in the strategies folder.
The environment implements:

- RandomAgent
- EpsilonGreedyAgent
- UCB_BaselineAgent

## ExperimentRunner
Runs experiments, logs results, plots optionally. A wrapper greatly simplifying the use of the environment.

## Package Installation and CLI Usage

After cloning your fork of the repository, install the package in editable mode:

pip install -e .

Note - this will create a distribution folder in your repo - save to ignore. This file is caught by our gitignore, so it won't be tracked by git.


This makes the framework available as the Python module multi_agent_bandits and installs the command-line tool mab

Then, you can execute experiments as 

mab run name_of_your_experiment_script

As long as your experiment script is located in the experiments folder.

There's available flags for this command, defining number of steps, logging and plotting. An example call could look like (from MAB-framework directory):

mab run example --steps 20000 --save results/example --plot-rewards --plot-frequencies

which would execute the experiments/example.py script, override the default number of steps to 20000, save results to results/example, and plot both kinds of plots.

To run batch experiments you can run this in the terminal in the MAB framework directury:
mab run circuit_breaker_batch --steps 10000 --save results/circuit_breaker_batch

To analyze these results:
python multi_agent_bandits/experiments/analyze_circuit_breaker_results.py       
