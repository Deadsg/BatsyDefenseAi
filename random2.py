# random_q_state.py

import random

def generate_random_q_state(num_states, num_actions):
    """
    Generate a random Q-state with random Q-values.

    Args:
        num_states (int): Number of states in the environment.
        num_actions (int): Number of possible actions.

    Returns:
        dict: Random Q-state represented as a dictionary.
            Example: {'state1': {'action1': 0.1, 'action2': 0.2}, 'state2': {'action1': 0.3, 'action2': 0.4}}
    """
    q_state = {}

    for state in range(num_states):
        state_name = f'state{state + 1}'
        q_state[state_name] = {}

        for action in range(num_actions):
            action_name = f'action{action + 1}'
            q_state[state_name][action_name] = random.uniform(0, 1)

    return q_state