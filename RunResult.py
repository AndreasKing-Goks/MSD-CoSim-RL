# Import all classes
from utils.utilCosim import *
from utils.utilPolicy import *
from utils.utilRL import *

from policy.policy import *
from policyLearn.policyLearnQL import PolicyLearnQL
from policyLearn.policyLearnSARSA import PolicyLearnSARSA

import numpy as np
import pickle
import time

# Load the Q-table from a specific pickle file
with open(r'02_Saved\Scores\best_q_table_4_learn_SARSA.pkl', 'rb') as f:
    saved_data = pickle.load(f)
    initial_q_table = saved_data['Q-table']  # Extract the Q-table from the saved data

## Set the parameters for EnvironmentMST instantiation
y_desired           = 2.5       # The desired y position for the mass
obsPos              = 10        # Outer bound (+ & -) for position observation space
obsVel              = 10        # Outer bound (+ & -) for velocity observation space
mass                = 1         # Object mass in MSD system
stiffCoef           = 1         # Stiffness coefficient
dampCoef            = 1         # Damping coefficient
binNumbers          = 10        # The number of discretized bins
stopTime            = 10000     # Maximum final stopping time for CoSimulation
fps                 = 10        # fps count for timestep size
initMSDPos          = 0         # Initial MSD position
initMSDVel          = 0         # Initial MSD velocity
force               = 15        # Magnitude of control force (fixed)
y_tsBound           = 5.0       # Terminal state bound (+ & -) measured from y_desired
y_desiredTolerance  = 0.5      # Height tolerance for (+ & -) measured from y_desired for reward zone

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
                        force,
                        y_tsBound,
                        y_desiredTolerance)

# ## Parameters for Policy and Q-Learning
# # FOR DEBUGGING
# n_episode_value = 10    # Number of episodes required to compute the action value
# n_episode_score = 1     # Number of episodes required for socring te policy
# alpha=0.1                # Learning rate
# gamma=0.9                # Reward discount rate
# max_n_steps=1000        # Maximum number of steps in a single episodes
# print_every=2            # The episodes when the score is printed
# epsilon = 1.0            # Explorative coefficient
# epsilon_decay = 0.99   # Decays the epsilon as the episodes goes
# epsilon_min = 0.01       # The minimum epsilon after decay

# # FOR LEARNING
# n_episode_value = 5000                 # Number of episodes required to compute the action value
# n_episode_score = 100                   # Number of episodes required for socring te policy
# alpha=0.1                               # Learning rate
# gamma=0.9                               # Reward discount rate
max_n_steps=2000                        # Maximum number of steps in a single episodes
# print_every=n_episode_value/10          # The episodes when t he score is printed
epsilon = 1.0                           # Explorative coefficient
epsilon_decay = 0.9995                  # Decays the epsilon as the episodes goes
epsilon_min = 0.01                      # The minimum epsilon after decay

# # Initialize the action-value table
# binNumbers = envMSD.binNumbers
# obsSpaceNumber = envMSD.observation_space.n
# actSpaceNumber = envMSD.action_space.n

init_q = initial_q_table

## Intantiate the policy: Epsilon Greedy Policy
QL_policy = EpsilonGreedyPolicy(envMSD.action_space,
                                init_q,
                                epsilon,
                                epsilon_decay,
                                epsilon_min)

## Train the Policy
# Set the policy to training mode
QL_policy.eval()
# QL_policy.train()

disc_observations, observations, actions, rewards, dones = SampleEpisode(envMSD, 
                                                                         QL_policy, 
                                                                         max_n_steps)

# y_desired = envMSD.y_desired
# y_desiredBoundHi = envMSD.y_desiredBoundHi
# y_desiredBoundMed = envMSD.y_desiredBoundMed
# y_desiredBoundLow = envMSD.y_desiredBoundLow
# y_desiredTolerance = envMSD.y_desiredTolerance

# print(y_desired)
# print(y_desiredBoundHi)
# print(y_desiredBoundMed)
# print(y_desiredBoundLow)
# print(y_desiredTolerance)

# envMSD.CoSimInstance.PlotTimeSeries(separate_plots=True)
envMSD.PlotTimeSeriesResult(separate_plots=False)
