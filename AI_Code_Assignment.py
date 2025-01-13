import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import heapq
import time

data = pd.read_csv('traversal_cost_data.csv')  

# Encode non-numerical features
non_numerical_features = ['type_of_terrain', 'zone_classification', 'time_of_day']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [col for col in data.columns if col not in non_numerical_features + ['traversal_cost']]),
        ('cat', OneHotEncoder(), non_numerical_features)
    ])

X = data.drop('traversal_cost', axis=1)
y = data['traversal_cost']
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

xTrainProcessed = preprocessor.fit_transform(xTrain)
xTestProcessed = preprocessor.transform(xTest)


# Linear regression model
linearRegressor = LinearRegression()
linearRegressor.fit(xTrainProcessed, yTrain)

# Predictions for training data
yPredictTrainLinearRegressor = linearRegressor.predict(xTrainProcessed)
mae_train_lr = mean_absolute_error(yTrain, yPredictTrainLinearRegressor)
mse_train_lr = mean_squared_error(yTrain, yPredictTrainLinearRegressor)
rmse_train_lr = np.sqrt(mse_train_lr)

# Predictions for test data
yPredictLinearRegressor = linearRegressor.predict(xTestProcessed)
mae_test_lr = mean_absolute_error(yTest, yPredictLinearRegressor)
mse_test_lr = mean_squared_error(yTest, yPredictLinearRegressor)
rmse_test_lr = np.sqrt(mse_test_lr)

print(f"Linear Regression - Training Data - MAE: {mae_train_lr}, MSE: {mse_train_lr}, RMSE: {rmse_train_lr}")
print(f"Linear Regression - Test Data - MAE: {mae_test_lr}, MSE: {mse_test_lr}, RMSE: {rmse_test_lr}")

# Polynomial regression model
polynomialFeatures = PolynomialFeatures(degree=4)
xTrainPoly = polynomialFeatures.fit_transform(xTrainProcessed)
xTestPoly = polynomialFeatures.transform(xTestProcessed)

polyRegressor = LinearRegression()
polyRegressor.fit(xTrainPoly, yTrain)

# Predictions for training data
yPredictTrainPolyRegressor = polyRegressor.predict(xTrainPoly)
mae_train_pr = mean_absolute_error(yTrain, yPredictTrainPolyRegressor)
mse_train_pr = mean_squared_error(yTrain, yPredictTrainPolyRegressor)
rmse_train_pr = np.sqrt(mse_train_pr)

# Predictions for test data
yPredictPolyRegressor = polyRegressor.predict(xTestPoly)
mae_test_pr = mean_absolute_error(yTest, yPredictPolyRegressor)
mse_test_pr = mean_squared_error(yTest, yPredictPolyRegressor)
rmse_test_pr = np.sqrt(mse_test_pr)

print(f"Polynomial Regression - Training Data - MAE: {mae_train_pr}, MSE: {mse_train_pr}, RMSE: {rmse_train_pr}")
print(f"Polynomial Regression - Test Data - MAE: {mae_test_pr}, MSE: {mse_test_pr}, RMSE: {rmse_test_pr}")

# Neural network model
mlp = MLPRegressor(max_iter=1000, random_state=42, hidden_layer_sizes=(64, 32), activation='relu', solver='adam')

mlp.fit(xTrainProcessed, yTrain)

# Predictions for training data
yPredictTrainMLP = mlp.predict(xTrainProcessed)
mae_train_mlp = mean_absolute_error(yTrain, yPredictTrainMLP)
mse_train_mlp = mean_squared_error(yTrain, yPredictTrainMLP)
rmse_train_mlp = np.sqrt(mse_train_mlp)

# Predictions for test data
yPredictMLP = mlp.predict(xTestProcessed)
mae_test_mlp = mean_absolute_error(yTest, yPredictMLP)
mse_test_mlp = mean_squared_error(yTest, yPredictMLP)
rmse_test_mlp = np.sqrt(mse_test_mlp)

print(f"Neural Network - Training Data - MAE: {mae_train_mlp}, MSE: {mse_train_mlp}, RMSE: {rmse_train_mlp}")
print(f"Neural Network - Test Data - MAE: {mae_test_mlp}, MSE: {mse_test_mlp}, RMSE: {rmse_test_mlp}")

#task 3
gridData = pd.read_csv('provided_grid.csv')

# Preprocess the grid data
gridDataProcessed = preprocessor.transform(gridData)

# Estimate traversal costs using the polynomial regression model
gridDataPoly = polynomialFeatures.transform(gridDataProcessed)
estimatedTraversalCosts = polyRegressor.predict(gridDataPoly)

# Save the results
gridData['estimated_traversal_cost'] = estimatedTraversalCosts
gridData.to_csv('Estimated_grid.csv', index=False)

# Pathfinding algorithms comparison

def DepthFirstSearch(grid, start, goal):
    stack = [(start, [start])]
    visited = set()
    DFSCount = 0
    while stack:
        (vertex, path) = stack.pop()
        if vertex in visited:
            continue
        if vertex == goal:
            print(f"DFS steps: {DFSCount}")
            return path
        visited.add(vertex)
        for neighbor in GetNeighbours(grid, vertex):
            stack.append((neighbor, path + [neighbor]))
        DFSCount += 1
    return None

def BreadthFirstSearch(grid, start, goal):
    queue = [(start, [start])]
    visited = set()
    BFSCount = 0
    while queue:
        (vertex, path) = queue.pop(0)
        if vertex in visited:
            continue
        if vertex == goal:
            print(f"BFS steps: {BFSCount}")
            return path
        visited.add(vertex)
        for neighbor in GetNeighbours(grid, vertex):
            queue.append((neighbor, path + [neighbor]))
        BFSCount += 1
    return None

def DijkstrasAlgorithm(grid, start, goal):
    queue = [(0, start, [])]
    visited = set()
    DijkstraCount = 0
    while queue:
        (cost, vertex, path) = heapq.heappop(queue)
        if vertex in visited:
            continue
        path = path + [vertex]
        if vertex == goal:
            print(f"Dijkstra steps: {DijkstraCount}")
            return path
        visited.add(vertex)
        for neighbor in GetNeighbours(grid, vertex):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + grid[neighbor], neighbor, path))
        DijkstraCount += 1
    return None

def AStarSearch(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    AStarCount = 0
    queue = [(0, start, [])]
    visited = set()
    while queue:
        (cost, vertex, path) = heapq.heappop(queue)
        if vertex in visited:
            continue
        path = path + [vertex]
        if vertex == goal:
            print(f"A* steps: {AStarCount}")
            return path
        visited.add(vertex)
        for neighbor in GetNeighbours(grid, vertex):
            if neighbor not in visited:
                priority = cost + grid[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(queue, (priority, neighbor, path))
        AStarCount += 1
    return None

def GetNeighbours(grid, vertex):
    rows, cols = len(grid), len(grid[0])
    x, y = vertex
    neighbors = []
    if x > 0:  # Up
        neighbors.append((x - 1, y))
    if x < rows - 1:  # Down
        neighbors.append((x + 1, y))
    if y > 0:  # Left
        neighbors.append((x, y - 1))
    if y < cols - 1:  # Right
        neighbors.append((x, y + 1))
    return neighbors

grid = np.array(gridData['estimated_traversal_cost']).reshape((20, 20))  # Example grid initialization
start = (0, 0)  # Example start position
goal = (len(grid) - 1, len(grid[0]) - 1)  # Example goal position

# Compare the algorithms
startTime = time.time()
DFSResult = DepthFirstSearch(grid, start, goal)
DFSTime = time.time() - startTime

startTime = time.time()
BFSResult = BreadthFirstSearch(grid, start, goal)
BFSTime = time.time() - startTime

startTime = time.time()
dijkstrasResult = DijkstrasAlgorithm(grid, start, goal)
dijkstrasTime = time.time() - startTime

startTime = time.time()
aStarResult = AStarSearch(grid, start, goal)
aStarTime = time.time() - startTime

# Print the results
print(f"DFS Time: {DFSTime}")
print(f"BFS Time: {BFSTime}")
print(f"Dijkstra Time: {dijkstrasTime}")
print(f"A* Time: {aStarTime}")