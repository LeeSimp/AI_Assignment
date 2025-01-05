import numpy as np
import random
import matplotlib.pyplot as plt

# Define the grid world environment
grid_size = 5
start_position = (0, 0)  # (1, 1) in 1-based indexing
goal_position = (4, 4)   # (5, 5) in 1-based indexing
obstacles = [(1, 1), (3, 3)]  # (2, 2) and (4, 4) in 1-based indexing

# Define the action space
actions = ['up', 'down', 'left', 'right']
action_dict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Initialize the Q-table with zeros
Q_table = np.zeros((grid_size, grid_size, len(actions)))


# Define the next state function
def get_next_state(state, action):
    next_state = (state[0] + action_dict[action][0], state[1] + action_dict[action][1])
    if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
        return next_state
    else:
        return state

# Define the reward function
def get_reward(state):
    if state == goal_position:
        return 100
    elif state in obstacles:
        return -10
    else:
        return 1

# Function to get the optimal path
def get_optimal_path():
    state = start_position
    path = [state]
    while state != goal_position:
        action = actions[np.argmax(Q_table[state[0], state[1]])]
        state = get_next_state(state, action)
        path.append(state)
    return path

# Evaluate the trained agent's performance
def evaluate_agent(num_episodes=100):
    total_rewards = []
    for episode in range(num_episodes):
        state = start_position
        total_reward = 0
        while state != goal_position:
            action = actions[np.argmax(Q_table[state[0], state[1]])]
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    average_reward = np.mean(total_rewards)
    return average_reward

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000

# Q-learning algorithm
for episode in range(num_episodes):
    state = start_position
    while state != goal_position:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)  # Explore
        else:
            action = actions[np.argmax(Q_table[state[0], state[1]])]  # Exploit

        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        best_next_action = np.argmax(Q_table[next_state[0], next_state[1]])
        td_target = reward + gamma * Q_table[next_state[0], next_state[1], best_next_action]
        td_error = td_target - Q_table[state[0], state[1], actions.index(action)]
        Q_table[state[0], state[1], actions.index(action)] += alpha * td_error

        state = next_state

# Print the Q-table
print("Q-table after training:")
print(Q_table)

# Get the optimal path
optimal_path = get_optimal_path()
print("Optimal path from start to goal:")
print(optimal_path)

# Evaluate the trained agent
average_reward = evaluate_agent()
print(f"Average total reward over 100 episodes: {average_reward}")



