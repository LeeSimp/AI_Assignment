import numpy as np
import random
import matplotlib.pyplot as plt

# Define the grid world environment
gridSize = 5
startPosition = (0, 0)  # (1, 1) in 1-based indexing
goalPosition = (4, 4)   # (5, 5) in 1-based indexing
obstacles = [(1, 1), (3, 3)]  # (2, 2) and (4, 4) in 1-based indexing

# Define the action space
actions = ['up', 'down', 'left', 'right']
actionDict = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

# Initialize the Q-table with zeros
qTable = np.zeros((gridSize, gridSize, len(actions)))

# Q-learning parameters
learningRate = 0.1  
discountFactor = 0.9  
explorationRate = 0.1  
numOfEpisodes = 1000

# Define the next state function
def GetNextState(state, action):
    nextState = (state[0] + actionDict[action][0], state[1] + actionDict[action][1])
    if 0 <= nextState[0] < gridSize and 0 <= nextState[1] < gridSize:
        return nextState
    else:
        return state

# Define the reward function
def GetReward(state):
    if state == goalPosition:
        return 100
    elif state in obstacles:
        return -10
    else:
        return 1

# Function to get the optimal path
def GetOptimalPath():
    state = startPosition
    path = [state]
    while state != goalPosition:
        action = actions[np.argmax(qTable[state[0], state[1]])]
        state = GetNextState(state, action)
        path.append(state)
    return path

# Evaluate the trained agent's performance
def EvaluateAgent(numOfEpisodes=100):
    total_rewards = []
    for episode in range(numOfEpisodes):
        state = startPosition
        totalReward = 0
        while state != goalPosition:
            action = actions[np.argmax(qTable[state[0], state[1]])]
            nextState = GetNextState(state, action)
            reward = GetReward(nextState)
            totalReward += reward
            state = nextState
        total_rewards.append(totalReward)
    averageReward = np.mean(total_rewards)
    return averageReward



# Q-learning algorithm
for episode in range(numOfEpisodes):
    state = startPosition
    while state != goalPosition:
        if random.uniform(0, 1) < explorationRate:
            action = random.choice(actions)  # Explore
        else:
            action = actions[np.argmax(qTable[state[0], state[1]])]  # Exploit

        next_state = GetNextState(state, action)
        reward = GetReward(next_state)
        best_next_action = np.argmax(qTable[next_state[0], next_state[1]])
        td_target = reward + discountFactor * qTable[next_state[0], next_state[1], best_next_action]
        td_error = td_target - qTable[state[0], state[1], actions.index(action)]
        qTable[state[0], state[1], actions.index(action)] += learningRate * td_error

        state = next_state

# Print the Q-table
print("Q-table after training:")
print(qTable)

# Get the optimal path
optimalPath = GetOptimalPath()
print("Optimal path from start to goal:")
print(optimalPath)

# Evaluate the trained agent
averageReward = EvaluateAgent()
print(f"Average total reward over 100 episodes: {averageReward}")



