# This code script is used to test and debug Class code

# Import all classes
from utils.utilCosim import *
from utils.utilPolicy import *
from utils.utilRL import *

from policy.policy import *
from policyLearn.policyLearnQL import PolicyLearnQL

import numpy as np
import os
import pickle
import time

# Set the desired height for the mass
y_desired = 2.5

# ## Test initialize Cosim
# env = EnvironmentMSD(y_desired)
# env.InitializeCoSim() 
#PASS

# ## Test replicate main.py
# env = EnvironmentMSD(y_desired)
# env.InitializeCoSim()
# env.CoSimInstance.Simulate()
# env.CoSimInstance.PlotTimeSeries(separate_plots=True)
#PASS

# ## Test replication of main.py, change the mass, stiffness, and damping parameter
# # Parameter check
# mass=1
# stiff_coef=1
# damp_coef=1

# # Run the environment
# env = EnvironmentMSD(y_desired, mass=mass, stiff_coef=stiff_coef, damp_coef=damp_coef)
# env.InitializeCoSim()
# env.CoSimInstance.Simulate()
# env.CoSimInstance.PlotTimeSeries(separate_plots=True)
#PASS

## Reset, sample, and score test episode
# Instantiate environment
env = EnvironmentMSD(y_desired)

# FOR DEBUGGING
n_episode_value = 5  # Number of episodes required to compute the action value
n_episode_score = 1  # Number of episodes required for socring te policy
bin_numbers=10          # Bin numbers
alpha=0.1               # Learning rate
gamma=0.9               # Reward discount rate
max_n_steps=10        # Maximum number of steps in a single episodes
print_every=2         # The episodes when the score is printed
epsilon = 1.0            # Explorative coefficient
epsilon_decay = 0.9995   # Decays the epsilon as the episodes goes
epsilon_min = 0.01       # The minimum epsilon after decay

# Initialize the action-value table
bin_numbers = env.bin_numbers
observation_space_number = env.observation_space.n
action_space_number = env.action_space.n

init_q = np.zeros((bin_numbers ** observation_space_number, action_space_number))

# Initialize the policy, Epsilon Greedy Policy
QL_policy = EpsilonGreedyPolicy(env.action_space,
                                init_q,
                                epsilon,
                                epsilon_decay,
                                epsilon_min)

# Reset environment (Even at first try)
env.reset()

# Set the max_n_step and run sampling
max_n_steps = 1000

# Test Sampling
# disc_observations, observations, actions, rewards, dones = SampleEpisode(env, QL_policy, max_n_steps=max_n_steps)

# print(disc_observations)
# print(observations)
# print(actions)
# print(rewards)
# print(dones)

score, _ = ScorePolicy(env, QL_policy, n_episode_score, gamma)

print(env.terminal_state.MSDPositionTerminal)
print(env.y_desiredBound)
print(score)

env.CoSimInstance.PlotTimeSeries(separate_plots=True)
