import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set parameters for noise variance and degrees of freedom
noise_variance = 0.1       # Set the noise variance
degrees_of_freedom = 3     # Degrees of freedom: sample size - (number of parameters + 1)
scaling_factor = degrees_of_freedom / noise_variance  # Calculate the scaling factor

# Generate values for plotting the scaled chi-squared distribution
x_values = np.arange(0, 0.5, 0.01)  # Define points at which we will plot the distribution

# Custom chi-squared PDF approximation
def chi2_pdf_approx(x, dof):
    return (x**(dof/2 - 1) * np.exp(-x / 2)) / (2**(dof/2) * np.math.gamma(dof/2))

y_values = scaling_factor * np.array([chi2_pdf_approx(val * scaling_factor, degrees_of_freedom) for val in x_values])

# Plot the distribution
plt.plot(x_values, y_values, label='Scaled Chi-squared Distribution')
plt.xlabel('Noise Variance')
plt.ylabel('Probability Density')
plt.title('Distribution of the Noise Variance')
plt.show()

# Calculate and plot the peak location for this distribution
peak_location = (degrees_of_freedom - 2) / degrees_of_freedom * noise_variance
plt.plot(peak_location, scaling_factor * chi2_pdf_approx(peak_location * scaling_factor, degrees_of_freedom), 'ro', label='Peak')
plt.legend()
plt.show()

# Simulating noise distribution
sample_size = 5               # Sample size
n_samples = 1000000           # Number of simulations
S2 = np.zeros(n_samples)      # Array to store variance estimates

# Define the linear model with true parameters
w0, w1 = 2, 1
linear_model = lambda x: w0 + w1 * x
x_data = np.arange(1, sample_size + 1)
X = np.vstack([np.ones(sample_size), x_data]).T

# Simulate the linear regression and estimate noise variance
for i in range(n_samples):
    noise = np.random.randn(sample_size) * np.sqrt(noise_variance)
    y_data = linear_model(x_data) + noise
    w_est = np.linalg.lstsq(X, y_data, rcond=None)[0]  # Calculate least squares solution
    residuals = y_data - X @ w_est                     # Compute residuals
    S2[i] = (residuals.T @ residuals) / (sample_size - 2)  # Estimate of noise variance

# Plot histogram of noise variance estimates
bin_spacing = 0.01
bin_centers = np.arange(0, 0.5, bin_spacing)
hist, bin_edges = np.histogram(S2, bins=bin_centers, density=True)
plt.plot(bin_centers[:-1], hist, 'k^')
plt.xlabel('Noise Variance Estimate')
plt.ylabel('Frequency')
plt.title('Histogram of Noise Variance Estimates')
plt.show()
