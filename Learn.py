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

## Set the parameters for EnvironmentMST instantiation
y_desired           = 2.5       # The desired y position for the mass
obsPos              = 10        # Outer bound (+ & -) for position observation space
obsVel              = 10        # Outer bound (+ & -) for velocity observation space
mass                = 1         # Object mass in MSD system
stiffCoef           = 1         # Stiffness coefficient
dampCoef            = 1         # Damping coefficient
binNumbers          = 10        # The number of discretized bins
stopTime            = 10000     # Maximum final stopping time for CoSimulation
fps                 = 60        # fps count for timestep size
initMSDPos          = 0         # Initial MSD position
initMSDVel          = 0         # Initial MSD velocity
Force               = 10        # Magnitude of control force (fixed)
y_tsBound           = 5.0       # Terminal state bound (+ & -) measured from y_desired
y_desiredTolerance  = 0.25      # Height tolerance for (+ & -) measured from y_desired for reward zone

## Instantiate EnvironmentMSD as envMSD
envMSD = EnvironmentMSD(y_desired,
                        obsPos,
                        obsVel,
                        mass,
                        stiffCoef,
                        dampCoef,
                        binNumbers,
                        stopTime,
                        fps,
                        initMSDPos,
                        initMSDVel,
                        Force,
                        y_tsBound,
                        y_desiredTolerance)

## Parameters for Policy and Q-Learning
# FOR DEBUGGING
n_episode_value = 10     # Number of episodes required to compute the action value
n_episode_score = 10     # Number of episodes required for socring te policy
alpha=0.1                # Learning rate
gamma=0.9                # Reward discount rate
max_n_steps=10           # Maximum number of steps in a single episodes
print_every=2            # The episodes when the score is printed
epsilon = 1.0            # Explorative coefficient
epsilon_decay = 0.9995   # Decays the epsilon as the episodes goes
epsilon_min = 0.01       # The minimum epsilon after decay

# # FOR LEARNING
# n_episode_value = 20000  # Number of episodes required to compute the action value
# n_episode_score = 5000   # Number of episodes required for socring te policy
# alpha=0.05               # Learning rate
# gamma=0.9                # Reward discount rate
# max_n_steps=10000        # Maximum number of steps in a single episodes
# print_every=2500         # The episodes when t he score is printed
# epsilon = 1.0            # Explorative coefficient
# epsilon_decay = 0.9999   # Decays the epsilon as the episodes goes
# epsilon_min = 0.01       # The minimum epsilon after decay

# Initialize the action-value table
binNumbers = envMSD.binNumbers
obsSpaceNumber = envMSD.observation_space.n
actSpaceNumber = envMSD.action_space.n

init_q = np.zeros((binNumbers ** obsSpaceNumber, actSpaceNumber))

## Intantiate the policy: Epsilon Greedy Policy
QL_policy = EpsilonGreedyPolicy(envMSD.action_space,
                                init_q,
                                epsilon,
                                epsilon_decay,
                                epsilon_min)

## Train the Policy
# Set the policy to training mode
QL_policy.train()

# Do policy improvement
PolicyLearnQL(envMSD, 
              QL_policy,
              n_episode_value,
              n_episode_score,
              alpha,
              gamma,
              max_n_steps,
              print_every)
