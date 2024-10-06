# -*- coding: utf-8 -*-
"""Assignment1_re.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1drLQEiByFWKOPLkljA9q0FCEnlpkik46
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')
print(data.head())
print(data.info())
print(data.describe())

#Scatter plot of data
plt.figure(figsize=(10,6))
sns.scatterplot(x='X', y='y', data= data)
plt.title ('Scatter Plot of Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

#Histogram of target variable ('y')
plt.figure(figsize=(10,6))
sns.histplot(data['y'], bins=20, kde=True)
plt.title('Histogram of Target Variable(Y)')
plt.xlabel('Y')
plt.ylabel('Frequency')
plt.show()

#removing outliers - IQR based method
Q1 = data['y'].quantile(0.25)
Q3 = data['y'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data['y'] < lower_bound) | (data['y'] > upper_bound)]
print(outliers)
#remove outliers
data_cleaned = data[(data['y'] >= lower_bound) & (data['y'] <= upper_bound)]
print("Data shape after removing outliers:", data_cleaned.shape)

#Before outlier removal
plt.figure(figsize=(12, 6))
# Box plot for X
plt.subplot(1, 2, 1)
sns.boxplot(x=data['X'])
plt.title('Box plot of X')
# Box plot for y
plt.subplot(1, 2, 2)
sns.boxplot(x=data['y'])
plt.title('Box plot of y')


# After outlier removal
plt.figure(figsize=(12, 6))
# Box plot for X
plt.subplot(1, 2, 1)
sns.boxplot(x=data_cleaned['X'])
plt.title('Box plot of X')
# Box plot for y
plt.subplot(1, 2, 2)
sns.boxplot(x=data_cleaned['y'])
plt.title('Box plot of y')
plt.show()

#Segregating fatured and targets
X = data_cleaned[['X']].values  #Features (2D array)
y = data_cleaned['y'].values #target (1D array)
# Normalisation
scaler = StandardScaler() #initialise scalar
X_scaled = scaler.fit_transform(X) #fit and transform to features
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled] # Adding column of 1s as intercept term
# Splitting data into training and Validation Sets
# training = 0.8, validation(test) = 0.2  (random_state -> reproduciblity of split)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#shape of split data
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)

# Linear Regression with Batch Gradient Descent

theta = np.zeros(X_train.shape[1])           #initialise parameters (theta) with weights - 00 (intercept) 01(Slope)

def hypothesis(X, theta):                     # as a simple linear function : h(x) = theta_0 + theta_1*x
  return np.dot(X, theta)                     # in matrix form : h(X) = X*theta

def compute_cost(X,y,theta):                  # Defining cost function : J(theta) = (1/2m) * sum((h(x) - y)^2)
  m = len(y)
  J = (1/(2*m)) * np.sum((hypothesis(X,theta)-y)**2)
  return J

def gradient_descent(X, y, theta, learning_rate, num_epochs):
  print(f'Gradient Descent:')
  m = len(y)
  cost_history = []
  for epoch in range(num_epochs):
    predictions = hypothesis(X, theta)
    error = predictions - y
    gradient = (1/m) * np.dot(X.T, error)
    theta -= learning_rate * gradient         # updating batch gradient descent : theta' = theta - lr * d(J(theta))/d(theta)
    cost = compute_cost(X, y, theta)          # Calculate cost for this iteration and store
    cost_history.append(cost)
    if epoch % 100 == 0:
      print(f'Epoch {epoch}, Cost: {cost}')
    if epoch > 0 and abs(cost_history[-2] - cost_history[-1]) < 1e-5:   # Checking convergence
      print(f'Converged at epoch {epoch}')
      break

  print(f'Final Cost: {cost}')
  return theta, cost_history

# running BGD
learning_rate = 0.01
num_epochs = 1000
theta_bgd, cost_history_bgd = gradient_descent(X_train, y_train, theta, learning_rate, num_epochs)
print("Batch Gradient Descent Theta:", theta_bgd)

def stochastic_gradient_descent(X, y, theta, learning_rate, num_epochs, batch_size):
  print(f'Stochastic Gradient Decent:')
  m = len(y)
  num_batches = int(np.ceil(m / batch_size))
  cost_history = []

  for epoch in range(num_epochs):
    indices = np.random.permutation(m)                      # Shuffle the data at the start of each epoch
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    cost_epoch = 0

    for batch_idx in range(num_batches):
      start_idx = batch_idx * batch_size
      end_idx = (batch_idx + 1) * batch_size
      if end_idx > m:
        end_idx = m

      X_batch = X_shuffled[start_idx:end_idx]               # mini-batches
      y_batch = y_shuffled[start_idx:end_idx]

      predictions = hypothesis(X_batch, theta)              # Prediction for the batch and error
      error = predictions - y_batch
      gradient = (1/batch_size) * np.dot(X_batch.T, error)  # Gradient calculation for the batch
      theta -= learning_rate * gradient
      cost = compute_cost(X_batch, y_batch, theta)          # Cost calculation for current mini-batch
      cost_epoch += cost

    cost_epoch /= num_batches
    cost_history.append(cost_epoch)                         # Append average cost of the epoch to cost_history

    if epoch % 100 == 0:
      print(f'Epoch {epoch}, Cost: {cost_epoch}')

    if epoch > 0 and abs(cost_history[-2] - cost_history[-1]) < 1e-5:   # Checking convergence
      print(f'Converged at epoch {epoch}')
      break

  print(f'Final Cost: {cost_epoch}')
  return theta, cost_history

#running SGD
learning_rate = 0.01
num_epochs = 1000
batch_size = 32
theta_sgd, cost_history_sgd = stochastic_gradient_descent(X_train, y_train, theta, learning_rate, num_epochs, batch_size)
print("Stochastic Gradient Descent Theta:", theta_sgd)

# COMPARING COST HISTORIES FOR BATCH AND STOCHASTIC GRADIENT DESCENT
plt.plot(cost_history_bgd, label='Batch Gradient Descent')
plt.plot(cost_history_sgd, label='Stochastic Gradient Descent')
plt.xlabel('Epochs')
#plt.yscale('log')
plt.ylabel('Cost')
plt.legend()
plt.title('Cost Function Convergence')
plt.show()

#performance evaluation
# Function to calculate Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Predictions and MSE for BGD
y_pred_bgd = hypothesis(X_val, theta_bgd)
mse_bgd = mean_squared_error(y_val, y_pred_bgd)
print(f'MSE on validation data (BGD): {mse_bgd}')

# Predictions and MSE for SGD
y_pred_sgd = hypothesis(X_val, theta_sgd)
mse_sgd = mean_squared_error(y_val, y_pred_sgd)
print(f'MSE on validation data (SGD): {mse_sgd}')

# Plot predictions vs actual data
plt.figure(figsize=(10,6))
plt.scatter(X_val[:, 1], y_val, label='Actual data', color='blue')
plt.plot(X_val[:, 1], y_pred_bgd, label='BGD Predictions', color='red')
plt.plot(X_val[:, 1], y_pred_sgd, label='SGD Predictions', color='green')
plt.title('Model Predictions vs Actual Data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Function to run SGD with different batch sizes and learning rates
def test_sgd_with_varied_params(X_train, y_train, theta, learning_rate, num_epochs, batch_size):
    theta_sgd, cost_history_sgd = stochastic_gradient_descent(X_train, y_train, theta, learning_rate, num_epochs, batch_size)
    return theta_sgd, cost_history_sgd

# Parameters for experimentation
batch_sizes = [1, 16, 32, 64, 128]
learning_rates = [0.1, 0.01, 0.001]
num_epochs = 1000

# Visualizing the effect of different batch sizes
plt.figure(figsize=(10, 6))
for batch_size in batch_sizes:
    theta_sgd, cost_history_sgd = test_sgd_with_varied_params(X_train, y_train, np.zeros(X_train.shape[1]), 0.01, num_epochs, batch_size)
    plt.plot(cost_history_sgd, label=f'Batch Size: {batch_size}')

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Effect of Batch Size on SGD Convergence (Learning Rate = 0.01)')
plt.legend()
plt.show()

# Visualizing the effect of different learning rates
plt.figure(figsize=(10, 6))
for lr in learning_rates:
    theta_sgd, cost_history_sgd = test_sgd_with_varied_params(X_train, y_train, np.zeros(X_train.shape[1]), lr, num_epochs, 32)
    plt.plot(cost_history_sgd, label=f'Learning Rate: {lr}')

plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Effect of Learning Rate on SGD Convergence (Batch Size = 32)')
plt.legend()
plt.show()

# Plot predictions vs actual data for BGD and SGD after tuning
plt.figure(figsize=(10, 6))
plt.scatter(X_val[:, 1], y_val, label='Actual data', color='blue', alpha=0.5)
plt.plot(X_val[:, 1], hypothesis(X_val, theta_bgd), label='BGD Predictions', color='red')
plt.plot(X_val[:, 1], hypothesis(X_val, theta_sgd), label='SGD Predictions', color='green')
plt.title('Model Predictions vs Actual Data (After Tuning)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()