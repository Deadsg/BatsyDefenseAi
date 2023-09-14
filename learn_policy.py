def learn_policy():
    # Define your training logic here
    pass

class QLearningEnvironment:
    def select_action(self, state):
        epsilon = 0.1  # Exploration-exploitation trade-off parameter
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)  # Random action for exploration
        else:
            return np.argmax(self.q_table[state])