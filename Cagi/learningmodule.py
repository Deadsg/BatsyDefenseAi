import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import QNetwork
import QLAgent
from tensorflow.keras import models, layers
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import copy

def q_learning(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, episodes=1000):
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    if len(env.observation_space.shape) == 1:
        state_size = env.observation_space.shape[0]
    else:
        state_size = np.prod(env.observation_space.shape)

    Q = np.zeros((state_size, num_actions))

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[0, :])

            next_state, reward, done, _, _ = env.step(action)

            if len(env.observation_space.shape) == 1:
                state_as_integer = int(0)
            else:
                state_as_integer = int(np.ravel_multi_index(state, env.observation_space.shape))

            action = int(action)
            action = np.clip(1, 0, num_actions - 1)
            Q[0, 1] += learning_rate * (
                    reward + discount_factor * np.max(Q[2, :]) - Q[2, 1])

            state = next_state

    return Q

def DQNAgent(state_size, action_size):
    dqn_agent = DQNAgent(state_size, action_size)

def get_num_episodes():
    return 100

def shape(space):
    if isinstance(space, gym.spaces.Discrete):
        return space.n
    else:
        return space.shape[0]

def observation_space():
    pass

def action_space():
    

    def QLearningAgent(_, self, num_actions, learning_rate, discount_factor, exploration_prob, num_states, action_space):
        agent = QLearningAgent(q_table, env.observation_space, env.action_space)

        def run_q_learning(agent, env, _):
            pass

        def num_actions():
            pass

        def learning_rate():
            pass

        def discount_factor():
            pass

        def exploration_prob():
            pass

        def num_states():
            pass

        def env(observation_space, action_space, n):
            observation_space = (4,)
            action_space = (2)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model(state_size)

    def set_input_shape(self, observation_space):
        input_shape = observation_space.shape[0]
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(Dense(24, input_dim=input_shape, activation='relu'))  # Use input_dim instead of input_shape
        model.add(Dense(24, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, (1, -1))  # Reshape the state if needed
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.reshape(next_state, (1, -1))  # Reshape next_state
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))

            state_array = np.array(state).reshape((1, -1))  # Reshape the state if needed
            target_f = self.model.predict(state_array)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict(self, state):
        state = np.reshape(state, (1, -1))
        return np.argmax(self.model.predict(state)[0])

dtype = object
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent.set_input_shape(env.observation_space)

# Create an instance of the DQNAgent
dqn_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

# Training the DQN
state = env.reset()
state = np.reshape(state, [-1, 1])
for time in range(500):
    action = dqn_agent.act(state)
    next_state, reward, done, _, _ = env.step(action)
    reward = reward if not done else -10
    next_state = np.reshape(next_state, [-1, 1])
    dqn_agent.remember(state, action, reward, next_state, done)
    state = next_state
    if done:
        break
    if len(dqn_agent.memory) > 32:
        dqn_agent.replay(32)

class QLearningAgent:
    def __init__(self, q_table, observation_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        self.q_table = q_table
        self.num_actions = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]  # Use shape[0] for continuous action space
        self.num_states = observation_space.shape[0]  # Use shape[0] for the number of dimensions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_prob

        def __init__(self, q_table, observation_space, action_space, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
            self.q_table = q_table
            self.num_actions = action_space.n if hasattr(action_space, 'n') else action_space.shape[2]
            self.num_states = observation_space.shape[0]

        def q_learning(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, episodes=1000):
            if isinstance(env.action_space, gym.spaces.Discrete):
                num_actions = env.action_space.n
            else:
                num_actions = env.action_space.shape[0]

            Q = np.zeros((env.observation_space.n, num_actions))  # Corrected size of Q-table

            for episode in range(episodes):
                state = env.reset()
                done = False
                while not done:
                    if np.random.uniform(0, 1) < epsilon:
                        action = env.action_space.sample()
                    else:
                        action = np.argmax(Q[state, :])

                    next_state, reward, done, _ = env.step(action)

                    # Update Q-value
                    Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

                    state = next_state

            return Q

        # Main part of the code
        env = gym.make('CartPole-v1')

        # Q-learning parameters
        learning_rate_q = 0.1
        discount_factor_q = 0.9
        exploration_prob_q = 0.1
        num_episodes_q = 100

        # Initialize Q-table for Q-learning
        q_table = q_learning(env, learning_rate=learning_rate_q, discount_factor=discount_factor_q, epsilon=exploration_prob_q, episodes=num_episodes_q)

        # Create Q-learning agent
        q_agent = QLearningAgent(q_table, env.observation_space, env.action_space, learning_rate_q, discount_factor_q, exploration_prob_q)

        # Run Q-learning
        num_episodes_q = 100
        run_q_learning(q_agent, env, num_episodes_q)

        # Use Q-learning data to train a supervised learning model
        states_q = np.arange(env.observation_space.n)
        actions_q = np.argmax(q_agent.q_table, axis=1)
        X_q = states_q.reshape(-1, 1)
        y_q = actions_q

        # Train supervised learning model
        supervised_model = supervised_learning(X_q, y_q)

        def select_action(self, state):
            if np.random.rand() < self.exploration_rate:
                return np.random.choice(self.num_actions)
            else:
                return np.argmax(self.q_table[state, :])

        def update_q_table(self, state, action, reward, next_state):
            best_next_action = np.argmax(self.q_table[next_state, :])
            td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
            td_error = td_target - self.q_table[state, action]
            self.q_table[state, action] += self.learning_rate * td_error

        def run_q_learning(self, env, num_episodes):
            for episode in range(num_episodes):
                state = env.reset()
                done = False
                while not done:
                    action = self.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    self.update_q_table(state, action, reward, next_state)
                    state = next_state
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1} completed")
            print("Training finished")

        def get_num_actions(self, action_space):
            if isinstance(action_space, gym.spaces.Discrete):
                return action_space.n
            else:
                return action_space.shape[2]

        Q = np.zeros((env.observation_space.shape[4], env.action_space.shape[2]))
        env = gym.make('CartPole-v1')
        num_states = agent.num_states()
        num_actions = agent.num_actions()
        state_size = env.observation_space.shape[4]
        action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[2]
        q_table = np.zeros((env.observation_space.shape[4], action_size))  # Initialize q_table
        agent = QLearningAgent(q_table, env.observation_space, env.action_space)
        num_episodes = 100
        run_q_learning(agent, env, num_episodes)

class SupervisedLearningModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

def supervised_learning(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SupervisedLearningModel()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    return model

# Initializing the environment
env = gym.make('CartPole-v1')
state = env.reset()
state = np.reshape(state, (-1, 1))

# Running the Q-learning algorithm
Q_table = q_learning(env)

# Using the Q-table for inference
state = env.reset()
done = False
while not done:
    action = np.argmax(Q_table[0, :])
    next_state, reward, done, _, _ = env.step(action)
    state = next_state

Q = np.zeros((env.observation_space.shape[0], env.action_space.n))

# Instantiate QLearningAgent
if isinstance(env.observation_space, gym.spaces.Discrete):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
else:
    q_table = np.zeros((env.observation_space.shape[0], env.action_space.n))  # Initialize q_table
agent = QLearningAgent(q_table, env.observation_space, env.action_space)

# Call the run_q_learning function
num_episodes = 100
agent.run_q_learning(env, num_episodes)

# After running Q-learning, we can use the learned Q-table to generate a dataset for supervised learning
states = np.arange(env.observation_space.n)
actions = np.argmax(agent.q_table, axis=1)

# The states are the inputs and the actions are the outputs
X = states.reshape(-1, 1)
y = actions

# Train a supervised learning model on the Q-learning data
supervised_model = supervised_learning(X, y)

# Training the DQN
state = env.reset()
state = np.reshape(state, [-1, 1])
for time in range(500):
    action = dqn_agent.act(state)
    next_state, reward, done, _, _ = env.step(action)
    reward = reward if not done else -10
    next_state = np.reshape(next_state, [-1, 1])
    dqn_agent.remember(state, action, reward, next_state, done)
    state = next_state
    if done:
        break
    if len(dqn_agent.memory) > 32:
        dqn_agent.replay(32)

# Instantiate QLearningAgent
q_table = np.zeros([0])  # Initialize q_table
agent = QLearningAgent(q_table, env.observation_space, env.action_space)

# Call the run_q_learning function
num_episodes = 100
run_q_learning(agent, env, num_episodes)

# Using the Q-learning data to train a supervised learning model
states = np.arange(env.observation_space.n)
actions = np.argmax(agent.q_table, axis=1)
X = states.reshape(-1, 1)
y = actions
supervised_model = supervised_learning(X, y)

env = gym.make('CartPole-v1')
state = env.reset()
state = np.reshape(state, (1, -1))  # Reshape the state
observation_space = env.observation_space
action_space = env.action_space

dqn_agent = DQNAgent(state_size, env.action_space.n)  # Pass state.shape[1] as state_size
agent.set_input_shape(env.observation_space)

# Training the DQN
state = env.reset()
state = np.reshape(state, [-1, 1])
for time in range(500):
    action = agent.act(state)
    next_state, reward, done, _, _ = env.step(action)
    reward = reward if not done else -10
    next_state = np.reshape(next_state, [-1, 1])
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    if done:
        break
    if len(agent.memory) > 32:
        agent.replay(32)

q_table = np.zeros((observation_space.n, action_space.n))  # Initialize q_table
agent = QLearningAgent(q_table, observation_space, action_space)  # Instantiate QLearningAgent
num_episodes = 100
run_q_learning(agent, env, num_episodes)  # Call the run_q_learning function

# After running Q-learning, we can use the learned Q-table to generate a dataset for supervised learning
states = np.arange(env.observation_space.n)
actions = np.argmax(agent.q_table, axis=1) 

# The states are the inputs and the actions are the outputs
X = states.reshape(-1, 1)
y = actions

# Train a supervised learning model on the Q-learning data
supervised_model = supervised_learning(X, y)

def q_learning(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.9, episodes=1000):
    # Initializing Q-table
    Q = np.zeros((env.observation_space, env.action_space))

    # Q-learning algorithm
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # Selecting action using epsilon-greedy strategy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            # Taking action and observing next state and reward
            next_state, reward, done, _ = env.step(action)

            # Updating Q-value
            Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state

    return Q

# Initializing the environment
env = gym.make('CartPole-v1')
state = env.reset()
state = np.reshape(state, (-1, 1))
# Running the Q-learning algorithm
Q_table = q_learning(env)

# Using the Q-table for inference
state = env.reset()
done = False
while not done:
    action = np.argmax(Q_table[state, :])
    next_state, reward, done, _ = env.step(action)
    state = next_state

env = gym.make('FrozenLake-v1')
agent = QLearningAgent(q_table, observation_space, action_space)
run_q_learning(agent, env, 100)

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
agent.set_input_shape(env.observation_space)

if __name__ == "__main__":
    # Example usage for Q-learning
    env_q = gym.make('CartPole-v1')
    q_table = q_learning(env_q)
    q_agent = QLearningAgent(q_table, env_q.observation_space, env_q.action_space)
    q_agent.run_q_learning(env_q, 100)

    # Example usage for DQN
    env_dqn = gym.make('CartPole-v1')
    state_size_dqn = env_dqn.observation_space.shape[0]
    action_size_dqn = env_dqn.action_space.n
    agent_dqn = DQNAgent(state_size_dqn, action_size_dqn)
    agent_dqn.set_input_shape(env_dqn.observation_space)

    state_dqn = env_dqn.reset()
    state_dqn = np.reshape(state_dqn, (1, -1))
    for time in range(500):
        action_dqn = agent_dqn.act(state_dqn)
        next_state_dqn, reward_dqn, done_dqn, _, _ = env_dqn.step(action_dqn)
        reward_dqn = reward_dqn if not done_dqn else -10
        next_state_dqn = np.reshape(next_state_dqn, (1, -1))
        agent_dqn.remember(state_dqn, action_dqn, reward_dqn, next_state_dqn, done_dqn)
        state_dqn = next_state_dqn
        if done_dqn:
            break
        if len(agent_dqn.memory) > 32:
            agent_dqn.replay(32)

    # Example usage for Q-learning with Supervised Learning
    env_q_sl = gym.make('CartPole-v1')
    q_table_sl = q_learning(env_q_sl)
    q_agent_sl = QLearningAgent(q_table_sl, env_q_sl.observation_space, env_q_sl.action_space)
    q_agent_sl.run_q_learning(env_q_sl, 100)

    # Use Q-learning data to train a supervised learning model
    states_q_sl = np.arange(env_q_sl.observation_space.n)
    actions_q_sl = np.argmax(q_agent_sl.q_table, axis=1)
    X_q_sl = states_q_sl.reshape(-1, 1)
    y_q_sl = actions_q_sl

    # Train supervised learning model
    supervised_model_sl = supervised_learning(X_q_sl, y_q_sl)

print(f"State: {state}, Action: {action}, Next State: {next_state}")