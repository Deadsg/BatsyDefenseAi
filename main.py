import random
import numpy as np

class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_prob:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * self.q_table[next_state, best_next_action] - self.q_table[state, action])

def train_q_learning(agent, num_episodes):
    for episode in range(num_episodes):
        state = random.randint(0, agent.num_states - 1)

        while True:
            action = agent.choose_action(state)
            next_state = random.randint(0, agent.num_states - 1)
            reward = random.uniform(0, 1)

            agent.update_q_table(state, action, reward, next_state)

            state = next_state

            if episode == num_episodes - 1:
                return agent.q_table

# Example usage:
if __name__ == "__main__":
    num_states = 5
    num_actions = 3

    agent = QLearningAgent(num_states, num_actions)
    trained_q_table = train_q_learning(agent, num_episodes=1000)
    print(trained_q_table)