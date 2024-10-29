import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n = 10  # number of observations
x = np.linspace(0, 1, n)
y = np.sin(2 * np.pi * x) + np.sqrt(0.1) * np.random.randn(n)

# Prepare data for polynomial regression
degree = 9
X = np.vander(x, degree + 1, increasing=True)  # Vandermonde matrix for polynomial terms

# Ridge regression with different lambda values
lambdas = [10**(-i) for i in range(10, 1, -2)]

# Plotting
plt.figure(figsize=(12, 8))
xx = np.linspace(0, 1, 100)
X_test = np.vander(xx, degree + 1, increasing=True)

for lambda_val in lambdas:
    # Regularization matrix
    ridge_matrix = lambda_val * np.eye(degree + 1)
    
    # Compute the weights
    w = np.linalg.inv(X.T @ X + ridge_matrix) @ X.T @ y

    # Prediction
    y_pred = X_test @ w
    
    # Plot
    plt.plot(xx, y_pred, label=f'λ = {lambda_val:.0e}')
    
# Plot true function and data points
plt.plot(xx, np.sin(2 * np.pi * xx), 'b--', label='True function')
plt.plot(x, y, 'bo', label='Data')
plt.ylim(-1.5, 1.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Polynomial Ridge Regression with varying λ values")
plt.legend()
plt.show()
