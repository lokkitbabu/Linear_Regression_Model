import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Generate synthetic data
np.random.seed(42) 
x = np.linspace(-1, 1, 100)
u = np.random.normal(0, 0.1, len(x)) 
y = np.sin(2 * np.pi * x) + u 


def create_polynomial_features(x, degree):
    """
    Creates a polynomial feature matrix for input array x up to the specified degree.
    """
    return np.vstack([x**i for i in range(degree + 1)]).T

def k_fold_cross_validation(x, y, degree, k=5):
    """
    Perform k-fold cross-validation and return the mean squared error for the given polynomial degree.
    """
    # Shuffle data for cross-validation
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    x, y = x[indices], y[indices]

    # Split data into k approximately equal-sized folds
    fold_size = len(x) // k
    mse_values = []

    for fold in range(k):
        # Define training and testing indices
        test_indices = np.arange(fold * fold_size, (fold + 1) * fold_size)
        train_indices = np.delete(indices, test_indices)

        # Create training and testing sets
        x_train, y_train = x[train_indices], y[train_indices]
        x_test, y_test = x[test_indices], y[test_indices]

        # Create polynomial features
        X_train = create_polynomial_features(x_train, degree)
        X_test = create_polynomial_features(x_test, degree)

        # Fit polynomial model using normal equation: (X^T X)^{-1} X^T y
        coefficients = np.linalg.inv(X_train.T @ X_train) @ (X_train.T @ y_train)

        # Predict on the test set
        y_pred = X_test @ coefficients

        # Compute MSE for the fold
        mse = np.mean((y_test - y_pred) ** 2)
        mse_values.append(mse)

    # Return the mean MSE across all folds
    return np.mean(mse_values)

# Step 4: Compute MSE for each polynomial degree (1 to 9) for k = 5 and k = 10
degrees = range(1, 10)
mse_k5 = [k_fold_cross_validation(x, y, degree, k=5) for degree in degrees]
mse_k10 = [k_fold_cross_validation(x, y, degree, k=10) for degree in degrees]

# Step 5: Plot the results
plt.figure(figsize=(10, 6))
plt.plot(degrees, mse_k5, label='5-Fold Cross-Validation', marker='o')
plt.plot(degrees, mse_k10, label='10-Fold Cross-Validation', marker='s')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('MSE for Polynomial Models with 5-Fold and 10-Fold Cross-Validation')
plt.legend()
plt.grid(True)
plt.show()
