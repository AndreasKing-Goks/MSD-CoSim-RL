from utils.utilCosim import *
from utils.utilPolicy import *
from utils.utilRL import *

import numpy as np
import os
import pickle

def LogMessage(message,
               logID,
               log_dir):
    ## Helper function to log or print the message
    # Log file path
    file_name = f"log_result_{logID}.txt"
    log_file_path = os.path.join(log_dir, file_name)

    # Check directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # # If log_file_path exists
    # if os.path.exists(log_file_path):
    #     file_name = f"log_result_{logID+1}.txt"
    #     log_file_path = os.path.join(log_dir, file_name)

    # Log the message
    if log_file_path:
        with open(log_file_path, "a") as log_file:
            log_file.write(message + "\n")
    

def PolicyLearnQL(env,
                  policy,
                  n_episode_value,
                  n_episode_score,
                  logID,
                  logMode,
                  alpha:           float=0.1,
                  gamma:           float=0.9,
                  max_n_steps:       int=10000,
                  print_every:       int=500,
                  save_result:      bool=True,
                  save_dir:          str="./02_Saved/Scores",
                  test_log_dir:      str="./02_Saved/Log/Test",
                  learn_log_dir:     str="./02_Saved/Log/Learn"):
    # LOGGING INITIAL MESSAGE

    # Logging mode
    if logMode == "test":
        log_dir = test_log_dir
    elif logMode == "learn":
        log_dir = learn_log_dir
    elif logMode == "none":
        return
    else:
        raise ValueError(f"Log mode '{logMode}' is not available. Please choose from 'test', 'learn', or 'none'.")
    
    LogMessage("#################### LEARNING PARAMETERS #####################", logID, log_dir)
    LogMessage("", logID, log_dir)
    LogMessage(f"Policy improvement maximum episodes    : {n_episode_value}", logID, log_dir)
    LogMessage(f"Policy scoring maximum episodes        : {n_episode_score}", logID, log_dir)
    LogMessage(f"Log ID                                 : {logID}", logID, log_dir)
    LogMessage(f"Log mode                               : {logMode}", logID, log_dir)
    LogMessage(f"Learning rate                          : {alpha}", logID, log_dir)
    LogMessage(f"Maximum number of simulation step      : {max_n_steps}", logID, log_dir)
    LogMessage(f"Printing for every: {print_every} episodes", logID, log_dir)
    
    LogMessage("", logID, log_dir)
    
    LogMessage("################### Q-LEARNING RESULT LOG ####################", logID, log_dir)
    
    LogMessage("", logID, log_dir)

    # Create directory if it doesn't exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set q tables file name
    qt_filename = f"best_q_table_{logID}_{logMode}"

    # If qt_filename exists
    if os.path.exists(os.path.join(save_dir, qt_filename)):
        qt_filename = f"best_q_table_{logID+1}_{logMode}"

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
        if not ((episode + 1) % print_every):
            # Print current episode and epsilon
            LogMessage(f"Q-Learning episode: {episode + 1}, epsilon: {policy.epsilon: .2f}", logID, log_dir)

            # Switch to evaluation mode
            policy.eval()

            # Do policy scoring
            score, _ = ScorePolicy(env, policy, n_episode_score, gamma)

            ############## SAVING SCORE STARTS ##############
            if save_result:
                # Save the best QL scores
                if score > best_score:
                    best_score = score

                    # Save the best scores and the Q-table
                    save_path = os.path.join(save_dir, qt_filename)
                    with open(save_path, "wb") as f:
                        pickle.dump({
                            "Episode" : episode + 1,
                            "Score"   : score,
                            "Q-table" : policy.q,
                        }, f)
            ############## SAVING SCORES ENDS ##############

            # Print the score in terminal
            LogMessage("Score: {s:.2%}".format(s=score), logID, log_dir)

            # Switch to training mode
            policy.train()

            # Print whitespace for readability
            LogMessage("", logID, log_dir)
        ############## SCORING  ENDS ##############

    # THE SNIPPET BELOW IS ONLY FOR INITIALIZING BEFORE THE LOOP PROCESS
    # Reset the environment for RL and Co-Simulation
    env.reset()

    # Get the initial state and discretize it
    states = env.states
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

    # Switch to training mode
    policy.train()
    
    # ############## FINAL POLICY EVALUATION ##############
    # # Print last episode and epsilon
    # LogMessage(f"Q-Learning episode: {episode + 1}, epsilon: {policy.epsilon: .2f}", logID, log_dir)

    # # Turn evaluation mode on
    # policy.eval()
    
    # # Do last policy scoring
    # score, _ = ScorePolicy(env, policy, n_episode_score, gamma)

    # ############## SAVING SCORE STARTS ##############
    # if save_result:
    #     # Save the best QL scores
    #     if score > best_score:
    #         best_score = score

    #         # Save the best scores and the Q-table
    #         save_path = os.path.join(save_dir, qt_filename)
    #         with open(save_path, "wb") as f:
    #             pickle.dump({
    #                         "Episode" : episode + 1,
    #                         "Score"   : score,
    #                         "Q-table" : policy.q,
    #                     }, f)
    #         ############## SAVING SCORES ENDS ##############

    # # Print the score in terminal
    # LogMessage("Score: {s:.2%}".format(s=score), logID, log_dir)
    # LogMessage("", logID, log_dir)
    # LogMessage("##############################################################", logID, log_dir)
    return