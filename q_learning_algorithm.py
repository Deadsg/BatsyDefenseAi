class QLearningEnvironment:
    def __init__(self, n_states, n_actions, learning_rate, discount_factor):
        self.q_table = np.zeros((n_states, n_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        # ...
    
    def update_q_value(self, state, action, reward, next_state):
        # Q-value update equation
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[state, action])