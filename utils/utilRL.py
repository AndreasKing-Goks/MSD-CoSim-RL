import numpy as np

def SampleEpisode(env,            # Environment Class
                  policy,         # Policy class 
                  max_n_steps: int=10000,
                  reset:      bool=True):
    
    # Intitialize transition info container
    disc_observations = []
    observations      = []
    actions           = []
    rewards           = []
    dones             = []

    # If reset, initial state is now is the current state
    if reset:
        env.reset()                                         # Reset the environment
        init_states = env.states                            # Get the initial continuous states (from the resetted environment)
        disc_init_states = env.DiscretizeState(init_states) # Discretizes the initial states

    else:
        init_states = env.states                            # Directly get the initial continouos states
        disc_init_states = env.DiscretizeState(init_states) # Discretizes the initial states

    # Add the initial states to the observation container
    disc_observations.append(init_states)
    observations.append(init_states)

    # Go through the timestep
    for step in range(max_n_steps):
        # Sample an action based on the policy
        action = policy.sample(disc_observations[-1])

        # Obtain the next observation by stepping with action
        next_states, reward, done = env.state(action)

        # Discretizes the next_states
        disc_next_states = env.DiscretizeState(next_states)

        # Store the transition info into the containers
        disc_observations.append(disc_next_states)
        observations.append(next_states)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)

        # Terminate the episode if termination status is active
        if done:
            break
    
    return disc_observations, observations, actions, rewards, dones

def ScorePolicy(env,
                policy,
                n_episodes:  int=10000,
                gamma:     float=0.9):
    # Initialize variables and containers
    steps_to_goal = []

    # Go through the policy for each episodes
    for episode in range(n_episodes):
        # Sample the episodes
        _, _, _, rewards, _ = SampleEpisode(env, policy, reset=True)
 
        # Record the goal occurences and the steps to achieve it
        steps_to_goal.append(len(rewards))

        # Compute the scores: mean steps taken to reach goal for all episodes
        score = np.mean(steps_to_goal) if steps_to_goal else 0

    return score