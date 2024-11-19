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
# env = EnvironmentMSD(y_desired, mass=mass, stiffCoef=stiff_coef, dampCoef=damp_coef)
# env.InitializeCoSim()
# env.CoSimInstance.Simulate()
# env.CoSimInstance.PlotTimeSeries(separate_plots=True)
# # PASS

## Reset, sample, and score test episode
# Instantiate environment
force = 20
env = EnvironmentMSD(y_desired, force=force)

# FOR DEBUGGING
n_episode_value = 5  # Number of episodes required to compute the action value
n_episode_score = 1  # Number of episodes required for socring te policy
bin_numbers=10          # Bin numbers
alpha=0.1               # Learning rate
gamma=0.9               # Reward discount rate
max_n_steps=100000        # Maximum number of steps in a single episodes
print_every=2         # The episodes when the score is printed
epsilon = 1.0            # Explorative coefficient
epsilon_decay = 0.9995   # Decays the epsilon as the episodes goes
epsilon_min = 0.01       # The minimum epsilon after decay

# Initialize the action-value table
bin_numbers = env.binNumbers
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
max_n_steps = 100000

# # Test Sampling
step_taken_list = []
total_reward_received_list = []

eps = 5

QL_policy.eval()
QL_policy.train()

# QL_policy.q = mockup_q = np.random.rand(*init_q.shape)

for i in range(eps):
    disc_observations, observations, actions, rewards, dones = SampleEpisode(env, 
                                                                         QL_policy, 
                                                                         max_n_steps)
    step_taken_list.append(len(actions))
    total_reward_received_list.append(sum(rewards))

print(QL_policy.q)
print(step_taken_list)
print(np.mean(step_taken_list))
print(total_reward_received_list)
print(np.mean(total_reward_received_list))


# # print(disc_observations)
# # print(observations)
# # print(actions)
# # print(env.observation_space.MSDPosition)
# # print(env.observation_space.MSDVelocity)
# # print(env.terminal_state.MSDPositionTerminal) 
# # print(len(rewards))
# # print(sum(rewards))
# # print(len(actions))
# # print(len(actions.count(0)))
# # print(len(actions.count(1)))
# # print(dones)

# best_q_table_path = r'02_Saved\Scores\best_q_table_2_learn'


# # Load the best Q-table if it exists and compare scores
# if os.path.exists(best_q_table_path):
#     with open(best_q_table_path, 'rb') as f:
#         best_data = pickle.load(f)
#         best_q_table = best_data['Q-table']
#         best_score = best_data['Score']
    
#         # Overwrite the current Q-table with the best Q-table
#         QL_policy.q = best_q_table
# else:
#     print("Trained data is not found")

# # print(best_q_table)
# print(QL_policy.q)

# score, total_reward, total_step = ScorePolicy(env, 
#                                               QL_policy, 
#                                               max_n_steps, 
#                                               n_episode_score)

# print(f'{score:.2%}')
# print(total_reward)
# print(total_step)
# print(action_taken)
# # print(QL_policy.q)


env.CoSimInstance.PlotTimeSeries(separate_plots=True)
