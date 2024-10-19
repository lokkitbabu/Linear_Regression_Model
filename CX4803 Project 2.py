import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulation parameters
M = 1000000        # Number of simulations
n = 50          # Number of data points per simulation
w0_true = 2     # True intercept
w1_true = 3     # True slope
sigma = 1       # Standard deviation of noise

# Arrays to store results
w0_hats = np.zeros(M)
var_w0_hats = np.zeros(M)

for i in range(M):
    # Generate random x values
    x = np.random.uniform(0, 10, n)
    X = np.column_stack((np.ones(n), x))  # Add intercept term
    # Generate noise
    epsilon = np.random.normal(0, sigma, n)
    # Generate y values
    y = w0_true + w1_true * x + epsilon
    # Estimate coefficients using least squares
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    w0_hat, w1_hat = beta_hat
    w0_hats[i] = w0_hat
    # Estimate variance of the residuals
    y_hat = X @ beta_hat
    residuals = y - y_hat
    sigma_hat_sq = np.sum(residuals**2) / (n - 2)  # Degrees of freedom: n - p - 1
    # Calculate variance of w0_hat
    XTX_inv = np.linalg.inv(X.T @ X)
    var_w0_hat = sigma_hat_sq * XTX_inv[0, 0]
    var_w0_hats[i] = var_w0_hat

# Compute q as the average variance
q = np.mean(var_w0_hats)

# Standardize the estimates using q
Z = (w0_hats - w0_true) / np.sqrt(q)

# Analyze the distribution
# Plot histogram
plt.figure(figsize=(10,6))
count, bins, ignored = plt.hist(Z, bins=30, density=True, alpha=0.6, color='blue', label='Standardized Estimates')

# Plot standard normal PDF
z_vals = np.linspace(-4, 4, 1000)
pdf_vals = (1/np.sqrt(2 * np.pi)) * np.exp(-z_vals**2 / 2)
plt.plot(z_vals, pdf_vals, 'r-', lw=2, label='Standard Normal PDF')

plt.title('Distribution of Standardized Estimates')
plt.xlabel('Z')
plt.ylabel('Density')
plt.legend()
plt.show()

# Compute sample mean and standard deviation of Z
mean_Z = np.mean(Z)
std_Z = np.std(Z, ddof=1)
print(f'Sample mean of Z: {mean_Z:.4f}')
print(f'Sample standard deviation of Z: {std_Z:.4f}')
