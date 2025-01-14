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
qTable = np.zeros((grid_size, grid_size, len(actions)))

# Define the reward function
def GetReward(state):
    if state == goal_position:
        return 100
    elif state in obstacles:
        return -10
    else:
        return 1

# Define the next state function
def GetNextState(state, action):
    next_state = (state[0] + action_dict[action][0], state[1] + action_dict[action][1])
    if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
        return next_state
    else:
        return state

# Q-learning hyperparameters
alpha = 0.1  # Learning rate
beta = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
numOfEpisodes = 1000

# Q-learning algorithm
for episode in range(numOfEpisodes):
    state = start_position
    while state != goal_position:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)  # Explore
        else:
            action = actions[np.argmax(qTable[state[0], state[1]])]  # Exploit

        next_state = GetNextState(state, action)
        reward = GetReward(next_state)
        best_next_action = np.argmax(qTable[next_state[0], next_state[1]])
        td_target = reward + beta * qTable[next_state[0], next_state[1], best_next_action]
        td_error = td_target - qTable[state[0], state[1], actions.index(action)]
        qTable[state[0], state[1], actions.index(action)] += alpha * td_error

        state = next_state

# Print the Q-table
print("Q-table after training:")
print(qTable)

# Get the optimal path
def GetOptimalPath():
    state = start_position
    path = [state]
    while state != goal_position:
        action = actions[np.argmax(qTable[state[0], state[1]])]
        state = GetNextState(state, action)
        path.append(state)
    return path

optimalPath = GetOptimalPath()
print("Optimal path from start to goal:")
print(optimalPath)

# Evaluate the trained agent
def EvaluateAgent(num_episodes=100):
    total_rewards = []
    for episode in range(num_episodes):
        state = start_position
        total_reward = 0
        while state != goal_position:
            action = actions[np.argmax(qTable[state[0], state[1]])]
            next_state = GetNextState(state, action)
            reward = GetReward(next_state)
            total_reward += reward
            state = next_state
        total_rewards.append(total_reward)
    average_reward = np.mean(total_rewards)
    return average_reward

averageReward = EvaluateAgent()
print(f"Average total reward over 100 episodes: {averageReward}")

# Function to plot the Q-values
def PlotQValues(qTable):
    """
    Creates a heatmap for each action in the qTable.
    qTable shape is assumed to be (rows, cols, numActions).
    """
    rows, cols, num_actions = qTable.shape

    # Adjust action labels to match your actual actions' order if needed
    action_labels = ["Up", "Down", "Left", "Right"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i in range(num_actions):
        row_idx = i // 2
        col_idx = i % 2

        # Plot the Q-values for action i
        im = axs[row_idx, col_idx].imshow(qTable[:, :, i], cmap='hot', origin='upper')
        axs[row_idx, col_idx].set_title(f"Q-values for {action_labels[i]}")
        axs[row_idx, col_idx].set_xticks(np.arange(cols))
        axs[row_idx, col_idx].set_yticks(np.arange(rows))

        # Optional: label each cell with its Q-value
        for r in range(rows):
            for c in range(cols):
                q_val = qTable[r, c, i]
                axs[row_idx, col_idx].text(c, r, f"{q_val:.1f}",
                                           ha="center", va="center", color="black", fontsize=8)

        fig.colorbar(im, ax=axs[row_idx, col_idx])

    plt.tight_layout()
    plt.show()

# Plot the learned Q-table
PlotQValues(qTable)

# Function to plot the optimal path
def PlotOptimalPath(optimalPath):
    grid = np.zeros((grid_size, grid_size))
    for obstacle in obstacles:
        grid[obstacle] = -1  # Mark obstacles

    fig, ax = plt.subplots()
    ax.matshow(grid, cmap='gray')

    for (i, j) in optimalPath:
        ax.text(j, i, 'o', va='center', ha='center', color='red')

    ax.text(start_position[1], start_position[0], 'S', va='center', ha='center', color='green')
    ax.text(goal_position[1], goal_position[0], 'G', va='center', ha='center', color='blue')

    plt.title("Optimal Path from Start to Goal")
    plt.show()

# Plot the optimal path
PlotOptimalPath(optimalPath)
