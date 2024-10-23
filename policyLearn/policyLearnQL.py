from utils.utilCosim import *
from utils.utilPolicy import *
from utils.utilRL import *

import numpy as np
import os
import pickle

def PolicyLearnQL(env,
                  policy,
                  n_episode_value,
                  n_episode_score,
                  alpha:           float=0.1,
                  gamma:           float=0.9,
                  max_n_steps:       int=10000,
                  print_every:       int=500,
                  save_dir:          str="./02_Saved/Scores"):
    
    # Create directory if it doesn't exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Intializ the best score
    best_score = np.inf

    # Initiate training mode
    policy.train()
    
    ############## EVALUATION-EXPLOITATION CYCLE ##############
    # Through each episode
    for episode in range(n_episode_value):

        # Set the policy to handle the epsilon decay
        policy.begin_episode(episode)

        ############## SCORING START ##############
        # Print for every couple episode
        if not (episode % print_every):
            # Print current episode and epsilon
            print(f"Q-Learning episode: {episode}, epsilon: {policy.epsilon: .2f}")

            # Switch to evaluation mode
            policy.eval()

            # Do policy scoring
            score = ScorePolicy(env, policy, n_episode_score, gamma)

            ############## SAVING SCORE STARTS ##############
            # Save the best QL scores
            if score > best_score:
                best_score = score

                # Save the best scores and the Q-table
                save_path = os.path.join(save_dir, "best_q_table.pkl")
                with open(save_path, "wb") as f:
                    pickle.dump({
                        "Episode" : episode,
                        "Score"   : score,
                        "Q-table" : policy.q,
                    }, f)
            ############## SAVING SCORES ENDS ##############

            # Print the score in terminal
            print("Score: {s:.2%}".format(s=score))

            # Switch to training mode
            policy.train()

            # Print whitespace for readability
            print()
        ############## SCORING  ENDS ##############

    # THE SNIPPET BELOW IS ONLY FOR INITIALIZING BEFORE THE LOOP PROCESS
    # Reset the environment for RL and Co-Simulation
    env.reset()

    # Get the initial state and discretize it
    states = env.state
    discrete_states = env.DiscretizeState(states)

    # Sample an action based on the policy using the discrete state
    action = policy.sample(discrete_states)

    # Do action value estimation and improvement method using Q-Learning
    for step in range(max_n_steps):
        # Perform forward step for the agent
        next_states, reward, done = env.step(action)

        # Discretize the next state
        next_discrete_states = env.DiscretizeState(next_states)

        # Sample next action using next states
        next_action = policy.sample(next_discrete_states)

        ## Compute the state-action value
        # Obtained reward
        term_1 = reward

        # Maximum expected state-action pair value
        term_2 = gamma * np.max(policy.q[next_discrete_states])

        # Current state-action pair value
        term_3 = policy.q[discrete_states, action]

        # Update the Q-tables
        policy.q[discrete_states, action] += alpha * (term_1 + term_2 -term_3)

        # If the episode has finished, compute the action value and then break the loop
        if done:
            break

        # Set the next state-action pair as the current state-action pair for the next action-value update
        discrete_states = next_discrete_states
        action = next_action
    
    return