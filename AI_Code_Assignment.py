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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Linear regression model
""" linear_regressor = LinearRegression()
linear_regressor.fit(X_train_processed, y_train)

y_pred_lr = linear_regressor.predict(X_test_processed)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
print(f"Linear Regression - MAE: {mae_lr}, MSE: {mse_lr}, RMSE: {rmse_lr}") """

# Polynomial regression model
polynomial_features = PolynomialFeatures(degree=4)
X_train_poly = polynomial_features.fit_transform(X_train_processed)
X_test_poly = polynomial_features.transform(X_test_processed)

poly_regressor = LinearRegression()
poly_regressor.fit(X_train_poly, y_train)

y_pred_pr = poly_regressor.predict(X_test_poly)
mae_pr = mean_absolute_error(y_test, y_pred_pr)
mse_pr = mean_squared_error(y_test, y_pred_pr)
rmse_pr = np.sqrt(mse_pr)
print(f"Polynomial Regression - MAE: {mae_pr}, MSE: {mse_pr}, RMSE: {rmse_pr}")

# Neural network model
"""mlp = MLPRegressor(max_iter=1000, random_state =42, hidden_layer_sizes=(64, 32), activation='relu', solver='adam')


mlp.fit(X_train_processed, y_train)

y_pred_mlp = mlp.predict(X_test_processed)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
print(f"Neural Network (MLP) - MAE: {mae_mlp}, MSE: {mse_mlp}, RMSE: {rmse_mlp}")"""

grid_data = pd.read_csv('provided_grid.csv')

# Preprocess the grid data
grid_data_processed = preprocessor.transform(grid_data)

# Estimate traversal costs using the polynomial regression model
grid_data_poly = polynomial_features.transform(grid_data_processed)
estimated_traversal_costs = poly_regressor.predict(grid_data_poly)

# Save the results
grid_data['estimated_traversal_cost'] = estimated_traversal_costs
grid_data.to_csv('Estimated_grid.csv', index=False)

# Pathfinding algorithms comparison
def depth_first_search(grid, start, goal):
    stack = [(start, [start])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex in visited:
            continue
        if vertex == goal:
            return path
        visited.add(vertex)
        for neighbor in get_neighbors(grid, vertex):
            stack.append((neighbor, path + [neighbor]))
    return None

def breadth_first_search(grid, start, goal):
    queue = [(start, [start])]
    visited = set()
    while queue:
        (vertex, path) = queue.pop(0)
        if vertex in visited:
            continue
        if vertex == goal:
            return path
        visited.add(vertex)
        for neighbor in get_neighbors(grid, vertex):
            queue.append((neighbor, path + [neighbor]))
    return None

def dijkstra_algorithm(grid, start, goal):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        (cost, vertex, path) = heapq.heappop(queue)
        if vertex in visited:
            continue
        path = path + [vertex]
        if vertex == goal:
            return path
        visited.add(vertex)
        for neighbor in get_neighbors(grid, vertex):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + grid[neighbor], neighbor, path))
    return None

def a_star_search(grid, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    queue = [(0, start, [])]
    visited = set()
    while queue:
        (cost, vertex, path) = heapq.heappop(queue)
        if vertex in visited:
            continue
        path = path + [vertex]
        if vertex == goal:
            return path
        visited.add(vertex)
        for neighbor in get_neighbors(grid, vertex):
            if neighbor not in visited:
                priority = cost + grid[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(queue, (priority, neighbor, path))
    return None

def get_neighbors(grid, vertex):
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

rows = grid_data['row'].max() + 1
cols = grid_data['col'].max() + 1
grid = np.array(grid_data['estimated_traversal_cost']).reshape((rows, cols))  # Example grid initialization
start = (0, 0)  # Example start position
goal = (len(grid) - 1, len(grid[0]) - 1)  # Example goal position

# Compare the algorithms
start_time = time.time()
dfs_result = depth_first_search(grid, start, goal)
dfs_time = time.time() - start_time

start_time = time.time()
bfs_result = breadth_first_search(grid, start, goal)
bfs_time = time.time() - start_time

start_time = time.time()
dijkstra_result = dijkstra_algorithm(grid, start, goal)
dijkstra_time = time.time() - start_time

start_time = time.time()
a_star_result = a_star_search(grid, start, goal)
a_star_time = time.time() - start_time

# Print the results
print(f"DFS Result: {dfs_result}, Time: {dfs_time}")
print(f"BFS Result: {bfs_result}, Time: {bfs_time}")
print(f"Dijkstra Result: {dijkstra_result}, Time: {dijkstra_time}")
print(f"A* Result: {a_star_result}, Time: {a_star_time}")