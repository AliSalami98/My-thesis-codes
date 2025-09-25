import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from Theater2_data import Tw_in, omega1, omega2, omega3, omegab, Theater1, Theater2


# ------------------------
# Theater2
# ------------------------
# Define current states (X_k) and next states (X_k1)
X2_k = np.column_stack([Theater2[:-1]])
U2_k = np.column_stack([omegab[:-1], omega2[:-1]])
X2_k1 = np.column_stack([Theater2[1:]])  # X[k+1]

# Generate polynomial features for state parameters (for A matrix dependency)
# poly_X = PolynomialFeatures(degree=1, include_bias=True)  # Linear dependency on states
# state_features = poly_X.fit_transform(X_k)

# Generate polynomial features for input parameters (for B matrix dependency)
# poly_U2 = PolynomialFeatures(degree=2, include_bias=True)  # Linear dependency on inputs
# input_features2 = poly_U2.fit_transform(U2_k)

# Prepare regression inputs (A depends on states; B depends on inputs; D is independent)
# reg_input = np.hstack([state_features, input_features, D_k])
# reg_input2 = np.hstack([X2_k, input_features2])
reg_input2 = np.hstack([X2_k, U2_k])

reg_output2 = X2_k1  # Target outputs

# Fit regression model
model2 = LinearRegression(fit_intercept=False).fit(reg_input2, reg_output2)
coeffs2 = model2.coef_

# ------------------------
# Extract parameter-dependent and independent matrices
# ------------------------
# Determine dimensions
# n_state_params = state_features.shape[1]  # Number of features for state dependency
n_state_params = X2_k.shape[1]  # Number of features for state dependency
n_input_params = U2_k.shape[1]  # Number of features for input dependency
n_X = X2_k.shape[1]  # Number of state variables

# Split coefficients into parameter-dependent and independent matrices
A2_coeffs = coeffs2[:, :n_state_params]  # State-dependent A
B2_coeffs = coeffs2[:, n_state_params:n_state_params + n_input_params]  # Input-dependent B


import joblib
from sklearn.preprocessing import PolynomialFeatures

# Save A1, B1 coefficients and poly_U1 transformer
joblib.dump(A2_coeffs, "A2_coeffs.pkl")
joblib.dump(B2_coeffs, "B2_coeffs.pkl")
# joblib.dump(poly_U2, "poly_U2.pkl")
