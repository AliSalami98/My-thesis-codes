import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from Theater1_data import Tw_in, omega1, omega2, omega3, omegab, Theater1, Theater2

# ------------------------
# Prepare data
# ------------------------
# Define current states (X_k) and next states (X_k1)
X1_k = np.column_stack([Theater1[:-1]])
U1_k = np.column_stack([omegab[:-1]])
X1_k1 = np.column_stack([Theater1[1:]])  # X[k+1]

# Generate polynomial features for state parameters (for A matrix dependency)
# poly_X = PolynomialFeatures(degree=1, include_bias=True)  # Linear dependency on states
# state_features = poly_X.fit_transform(X_k)

# Generate polynomial features for input parameters (for B matrix dependency)
poly_U1 = PolynomialFeatures(degree=2, include_bias=True)  # Linear dependency on inputs
input_features1 = poly_U1.fit_transform(U1_k)

# Prepare regression inputs (A depends on states; B depends on inputs; D is independent)
# reg_input = np.hstack([state_features, input_features, D_k])
reg_input1 = np.hstack([X1_k, input_features1])
# reg_input1 = np.hstack([X1_k, U1_k])

reg_output1 = X1_k1  # Target outputs

# Fit regression model
model1 = LinearRegression(fit_intercept=False).fit(reg_input1, reg_output1)
coeffs1 = model1.coef_

# ------------------------
# Extract parameter-dependent and independent matrices
# ------------------------
# Determine dimensions
# n_state_params = state_features.shape[1]  # Number of features for state dependency
n_state_params = X1_k.shape[1]  # Number of features for state dependency
n_input_params = input_features1.shape[1]  # Number of features for input dependency
# n_input_params = U1_k.shape[1]  # Number of features for input dependency

n_X = X1_k.shape[1]  # Number of state variables

# Split coefficients into parameter-dependent and independent matrices
A1_coeffs = coeffs1[:, :n_state_params]  # State-dependent A
B1_coeffs = coeffs1[:, n_state_params:n_state_params + n_input_params]  # Input-dependent B

import joblib
from sklearn.preprocessing import PolynomialFeatures

# # Save A1, B1 coefficients and poly_U1 transformer
# joblib.dump(A1_coeffs, "A1_coeffs.pkl")
# joblib.dump(B1_coeffs, "B1_coeffs.pkl")
# joblib.dump(poly_U1, "poly_U1.pkl")

print("A Coefficients (state-dependent):\n", A1_coeffs)
print("B Coefficients (parameter-dependent):\n", B1_coeffs)

# ------------------------
# Simulation with LPV
# ------------------------
X1_pred = []  # List to store predicted states
X1_current = X1_k[0]  # Initialize the state with the first state

for k in range(len(U1_k)):
    # Compute parameter-dependent A matrix (based on current state)
    # state_vector = poly_X.transform(X_current.reshape(1, -1))  # Transform current state
    # A_k = A_coeffs @ state_vector.T  # State-dependent A

    A1_k = A1_coeffs  # State-dependent A

    # Compute parameter-dependent B matrix (based on input parameters)
    # B1_k = B1_coeffs

    input_vector1 = poly_U1.transform(U1_k[k].reshape(1, -1))  # Transform current input
    B1_k = B1_coeffs @ input_vector1.T  # Input-dependent B
    # Compute next state (D remains independent)
    # X1_next = (A1_k @ X1_current) + (B1_k @ U1_k[k])
    X1_next = (A1_k @ X1_current) + (B1_k.flatten())

    X1_pred.append(X1_next)
    X1_current = X1_next

# Convert predictions to a NumPy array
X1_pred = np.array(X1_pred)

mape = np.mean(np.abs((X1_k1 - X1_pred) / X1_k1)) * 100
print("Mean Absolute Percentage Error (MAPE):", mape, "%")
from sklearn.metrics import r2_score
r2 = r2_score(X1_k1, X1_pred)
print("Coefficient of Determination (R^2):", r2)

# ------------------------
# Compare with observed X[k+1]
# ------------------------
plt.figure(figsize=(10, 6))
plt.plot(X1_k1[:, 0], label="Observed Theater1", color="blue")
plt.plot(X1_pred[:, 0], label="Predicted Theater1", linestyle="dashed", color="orange")
plt.legend()
plt.title("Comparison of Observed vs Predicted States (Theater1)")
plt.xlabel("Time Steps")
plt.ylabel("Theater1")
plt.grid()

plt.show()