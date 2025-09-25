import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from Theater1_data import omegab, omega1, Theater1
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error


# ------------------------
# Theater1
# ------------------------
# Define current states (X_k) and next states (X_k1)
X1_k = np.column_stack([Theater1[:-1]])
U1_k = np.column_stack([omegab[:-1]])
d1_k = np.column_stack([omega1[:-1]])
X1_k1 = np.column_stack([Theater1[1:]])  # X[k+1]

# Generate polynomial features for state parameters (for A matrix dependency)
# poly_X = PolynomialFeatures(degree=1, include_bias=True)  # Linear dependency on states
# state_features = poly_X.fit_transform(X_k)

# Generate polynomial features for input parameters (for B matrix dependency)
poly_U1 = PolynomialFeatures(degree=2, include_bias=True)  # Linear dependency on inputs
input_features1 = poly_U1.fit_transform(U1_k)

# Prepare regression inputs (A depends on states; B depends on inputs; D is independent)
reg_input1 = np.hstack([X1_k, input_features1, d1_k])
# reg_input1 = np.hstack([X1_k, input_features1])
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
n_X = X1_k.shape[1]  # Number of state variables

# Split coefficients into parameter-dependent and independent matrices
A1_coeffs = coeffs1[:, :n_state_params]  # State-dependent A
B1_coeffs = coeffs1[:, n_state_params:n_state_params + n_input_params]  # Input-dependent B
D1_coeffs = coeffs1[:, n_state_params + n_input_params:]  # Input-dependent B

print("A Coefficients (state-dependent):\n", A1_coeffs)
print("B Coefficients (parameter-dependent):\n", B1_coeffs)
print("D Coefficients (parameter-dependent):\n", D1_coeffs)

import joblib

joblib.dump(A1_coeffs, "coeffs/A1_coeffs.pkl")
joblib.dump(B1_coeffs, "coeffs/B1_coeffs.pkl")
joblib.dump(D1_coeffs, "coeffs/D1_coeffs.pkl")
joblib.dump(poly_U1, "coeffs/poly_U1.pkl")
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
    input_vector1 = poly_U1.transform(U1_k[k].reshape(1, -1))  # Transform current input
    B1_k = B1_coeffs @ input_vector1.T  # Input-dependent B

    D1_k = D1_coeffs
    # Compute next state (D remains independent)
    X1_next = (A1_k @ X1_current) + (B1_k.flatten()) + (D1_k @ d1_k[k])
    X1_pred.append(X1_next)
    X1_current = X1_next

# Convert predictions to a NumPy array
X1_pred = np.array(X1_pred)

# Calculate MAPE and R²
mae = mean_absolute_error(X1_k1, X1_pred)
print("Mean Absolute Error (MAE):", mae, "K")
from sklearn.metrics import r2_score
r2 = r2_score(X1_k1, X1_pred)
print("Coefficient of Determination (R²):", r2)


# Convert temperatures from Kelvin to Celsius
X1_k1[:, 0] -= 273.15
X1_pred[:, 0] -= 273.15

# Define time in hours
time = np.linspace(0, len(X1_pred[:, 0]) / 3600, len(X1_pred[:, 0]))

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# First subplot: Theater1 temperature
ax1.plot(time, X1_k1[:,0],
         label="Measured",
         lw=1.8,
         color="#8A2BE2",
         linestyle='-',
        #  marker='o',
         markevery=10)
ax1.plot(time, X1_pred[:,0],
         label="Predicted",
         color="#8A2BE2",
         lw=1.3,
         linestyle='--',
        #  marker='x',
         markevery=10)

ax1.set_ylabel(r"Heater 1 temperature $T_\text{h1}$ [°C]", fontsize=16)
ax1.tick_params(axis='both', labelsize=14)  # Increase tick size
ax1.legend(loc="best", fontsize=14)
ax1.grid(True, alpha=0.3)

# Add MAPE and R² to the first subplot
ax1.text(
    0.75, 0.2, 
    f"MAE: {mae:.2f}K\nR²: {r2:.2f}",
    transform=ax1.transAxes, 
    fontsize=14,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)

# Second subplot: Inputs (Burner Fan Speed & Motor Speed)
ax2.plot(time, omegab[:-1], lw=1.8, label="Burner fan", color="#FFD700", alpha=0.7)
ax2.set_ylabel(r"Burner fan speed $\omega_\text{bf}$ [rpm]", fontsize=16)
ax2.tick_params(axis='y', labelsize=14)  # Increase tick size
ax2.legend(loc="upper center", fontsize=14)

# Twin y-axis for Motor Speed
ax3 = ax2.twinx()
ax3.plot(time, omega1[:-1], lw=1.8, label="Motor 1", color="#1E90FF", alpha=0.7)
ax3.set_ylabel(r"Roational speed $\omega_\text{m1}$ [rpm]", fontsize=16)
ax3.tick_params(axis='y', labelsize=14)  # Increase tick size
ax3.legend(loc="upper right", fontsize=14)

ax2.tick_params(axis='x', labelsize=14)  # Increase x-axis tick size
ax2.grid(True, alpha=0.3)
# Set common x-label
ax2.set_xlabel("Time [h]", fontsize=16)
plt.tight_layout()

plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\transient\TCHP_Theater1.eps", format='eps', bbox_inches='tight')

# Show the plot
plt.show()



# def predict_next_Theater1(X1_current, U1_current, D1_current, A1_coeffs, B1_coeffs, D1_coeffs, poly_U1):
#     # Compute A_k (parameter-independent in this case)
#     A1_k = A1_coeffs

#     # Compute parameter-dependent B matrix
#     input_vector = poly_U1.transform(U1_current.reshape(1, -1))  # Transform current input
#     B1_k = B1_coeffs @ input_vector.T  # Input-dependent B


#     D1_k = D1_coeffs
#     # Compute next state (D remains independent)
#     X1_next = (A1_k @ X1_current) + (B1_k.flatten()) + (D1_k @ D1_current)
#     return X1_next

# X_current = np.column_stack([450])
# # U_current = np.column_stack([6000, 100])

# # print(X_current)
# # print(U_current)
# import joblib

# A1_coeffs = joblib.load("coeffs/A1_coeffs.pkl")
# B1_coeffs = joblib.load("coeffs/B1_coeffs.pkl")
# D1_coeffs = joblib.load("coeffs/D1_coeffs.pkl")
# poly_U1 = joblib.load("coeffs/poly_U1.pkl")

# # # Predict the next state
# a_X_current = []
# j = 0
# for i in range(100000):
#     if i % 10000 == 0:
#         j = j + 1        
#     U_current = np.column_stack([2000 + j/10 * 7500])
#     D_current = np.column_stack([100])
#     X_current = predict_next_Theater1(X_current, U_current, D_current, A1_coeffs, B1_coeffs, D1_coeffs, poly_U1)
#     a_X_current.append(X_current[0][0])

# # print(X_next)
# plt.plot(a_X_current)
# plt.show()