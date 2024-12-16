import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('traversal_cost_data.csv')  

# Encode non-numerical features
non_numerical_features = ['type_of_terrain', 'zone_classification', 'time_of_day']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [col for col in data.columns if col not in non_numerical_features + ['traversal_cost']]),
        ('cat', OneHotEncoder(), non_numerical_features)
    ])

# Split the dataset into training and testing sets (80-20 split)
X = data.drop('traversal_cost', axis=1)
y = data['traversal_cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Linear regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_processed, y_train)

# Predict and evaluate linear regression
y_pred_lr = linear_regressor.predict(X_test_processed)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
print(f"Linear Regression - MAE: {mae_lr}, MSE: {mse_lr}, RMSE: {rmse_lr}")

# Polynomial regression model
polynomial_features = PolynomialFeatures(degree=4)
X_train_poly = polynomial_features.fit_transform(X_train_processed)
X_test_poly = polynomial_features.transform(X_test_processed)

poly_regressor = LinearRegression()
poly_regressor.fit(X_train_poly, y_train)

# Predict and evaluate polynomial regression
y_pred_pr = poly_regressor.predict(X_test_poly)
mae_pr = mean_absolute_error(y_test, y_pred_pr)
mse_pr = mean_squared_error(y_test, y_pred_pr)
rmse_pr = np.sqrt(mse_pr)
print(f"Polynomial Regression - MAE: {mae_pr}, MSE: {mse_pr}, RMSE: {rmse_pr}")

# Neural network model
mlp = MLPRegressor(max_iter=1000, random_state =42, hidden_layer_sizes=(64, 32), activation='relu', solver='adam')


mlp.fit(X_train_processed, y_train)

# Predict and evaluate neural network
y_pred_mlp = mlp.predict(X_test_processed)
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
print(f"Neural Network (MLP) - MAE: {mae_mlp}, MSE: {mse_mlp}, RMSE: {rmse_mlp}")