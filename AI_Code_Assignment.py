import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('traversal_cost_data.csv')

# Identify non-numerical features
non_numerical_features = ['type_of_terrain', 'zone_classification', 'time_of_day']

# Encode non-numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [col for col in data.columns if col not in non_numerical_features + ['traversal_cost']]),
        ('cat', OneHotEncoder(), non_numerical_features)
    ])

# Split the dataset into training and testing sets (80-20 split)
X = data.drop('traversal_cost', axis=1)
y = data['traversal_cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#linear regression model



# Create a pipeline for linear regression
linear_regression_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('regressor', LinearRegression())])

# Train the model
linear_regression_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred_lr = linear_regression_pipeline.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

print(f"Linear Regression - MAE: {mae_lr}, MSE: {mse_lr}, RMSE: {rmse_lr}")

#polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Create a pipeline for polynomial regression
polynomial_features = PolynomialFeatures(degree=4)
polynomial_regression_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                 ('poly', polynomial_features),
                                                 ('regressor', LinearRegression())])

# Train the model
polynomial_regression_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred_pr = polynomial_regression_pipeline.predict(X_test)
mae_pr = mean_absolute_error(y_test, y_pred_pr)
mse_pr = mean_squared_error(y_test, y_pred_pr)
rmse_pr = np.sqrt(mse_pr)

print(f"Polynomial Regression - MAE: {mae_pr}, MSE: {mse_pr}, RMSE: {rmse_pr}")

#neural network model

# Define the neural network model
def create_model(optimizer='adam', activation='relu', dropout_rate=0.0):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Wrap the model for use in scikit-learn
model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=0)

# Define the grid search parameters
param_grid = {
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.0, 0.2, 0.4],
    'batch_size': [10, 20],
    'epochs': [50, 100]
}
# Create the grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(preprocessor.fit_transform(X_train), y_train)

# Best parameters
print(f"Best parameters: {grid_result.best_params_}")

# Evaluate the best model
best_model = grid_result.best_estimator_
y_pred_nn = best_model.predict(preprocessor.transform(X_test))
mae_nn = mean_absolute_error(y_test, y_pred_nn)
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)

print(f"Neural Network - MAE: {mae_nn}, MSE: {mse_nn}, RMSE: {rmse_nn}")