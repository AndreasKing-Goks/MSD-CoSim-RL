import numpy as np

class UniformPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        self.n_action = action_space.n
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def probability(self, state, action):
        action_prob = 1/self.n_action # Uniform probability
        return action_prob
    
    def sample(self, state):
        action = self.action_space.sample() # Sample a random action

class GreedyPolicy(UniformPolicy):
    def __init__(self, action_space, q): # Initialize the parent class it is inheriting from
        super().__init__(action_space)
        self.q = q

    def max_value_action(self, state):
        action = np.argmax(self.q[state])
        return action
    
    def probability(self, state, action):
        # Select the highest value action
        action_max = self.max_value_action(state)

        # Return probability of 1 for the selected action 0 otherwise
        action_prob = float(action == action_max)

        return action_prob
    
    def sample(self, state):
        # Select the highest value action
        action_max = self.max_value_action(state)

        return action_max
    
class EpsilonGreedyPolicy(GreedyPolicy):
    def __init__(self, action_space, q, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.001):
        super().__init__(action_space, q)
        self.epsilon_start = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.epsilon = self.epsilon_start

    def sample(self, state):
        # print(self.training)
        # print(np.random.random() <= self.epsilon)
        # print(self.training and np.random.random() <= self.epsilon)
        # Sampling need to be done in the flat_index form
        if self.training and (np.random.random() <= self.epsilon):
            action = self.action_space.sample()
            # print(action)
        else:                                       # Evaluation mode
            action = np.argmax(self.q[state])
            # print(action)
        return action
    
    def begin_episode(self, episode_index):
        self.epsilon = self.epsilon_start * (self.epsilon_decay ** episode_index)
        self.epsilon = max(self.epsilon, self.epsilon_min)