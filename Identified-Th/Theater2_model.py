import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from Theater2_data import Tw_in, omega1, omega2, omega3, omegab, Theater1, Theater2
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# 1) Point to your installed file (from your screenshot)
path = r"C:\Users\ali.salame\AppData\Local\Microsoft\Windows\Fonts\CHARTERBT-ROMAN.OTF"
# (add the Bold/Italic too if you use them)
# fm.fontManager.addfont(r"...\CHARTERBT-BOLD.OTF")
# fm.fontManager.addfont(r"...\CHARTERBT-ITALIC.OTF")

# 2) Register and use the exact internal name
fm.fontManager.addfont(path)
prop = fm.FontProperties(fname=path)
mpl.rcParams["font.family"] = prop.get_name()   # e.g., "Bitstream Charter"
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10

# ------------------------
# Theater2
# ------------------------
# Define current states (X_k) and next states (X_k1)
X2_k = np.column_stack([Theater2[:-1]])
U2_k = np.column_stack([omegab[:-1]])
d2_k = np.column_stack([omega2[:-1]])
X2_k1 = np.column_stack([Theater2[1:]])  # X[k+1]

# Generate polynomial features for state parameters (for A matrix dependency)
# poly_X = PolynomialFeatures(degree=1, include_bias=True)  # Linear dependency on states
# state_features = poly_X.fit_transform(X_k)

# Generate polynomial features for input parameters (for B matrix dependency)
poly_U2 = PolynomialFeatures(degree=2, include_bias=True)  # Linear dependency on inputs
input_features2 = poly_U2.fit_transform(U2_k)

# Prepare regression inputs (A depends on states; B depends on inputs; D is independent)
reg_input2 = np.hstack([X2_k, input_features2, d2_k])
# reg_input2 = np.hstack([X2_k, input_features2])
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
n_input_params = input_features2.shape[1]  # Number of features for input dependency
n_X = X2_k.shape[1]  # Number of state variables

# Split coefficients into parameter-dependent and independent matrices
A2_coeffs = coeffs2[:, :n_state_params]  # State-dependent A
B2_coeffs = coeffs2[:, n_state_params:n_state_params + n_input_params]  # Input-dependent B
D2_coeffs = coeffs2[:, n_state_params + n_input_params:]  # Input-dependent B

print("A Coefficients (state-dependent):\n", A2_coeffs)
print("B Coefficients (parameter-dependent):\n", B2_coeffs)
print("D Coefficients (parameter-dependent):\n", D2_coeffs)

import joblib

joblib.dump(A2_coeffs, "coeffs/A2_coeffs.pkl")
joblib.dump(B2_coeffs, "coeffs/B2_coeffs.pkl")
joblib.dump(D2_coeffs, "coeffs/D2_coeffs.pkl")
joblib.dump(poly_U2, "coeffs/poly_U2.pkl")
# ------------------------
# Simulation with LPV
# ------------------------
X2_pred = []  # List to store predicted states
X2_current = X2_k[0]  # Initialize the state with the first state

for k in range(len(U2_k)):
    # Compute parameter-dependent A matrix (based on current state)
    # state_vector = poly_X.transform(X_current.reshape(1, -1))  # Transform current state
    # A_k = A_coeffs @ state_vector.T  # State-dependent A

    A2_k = A2_coeffs  # State-dependent A

    # Compute parameter-dependent B matrix (based on input parameters)
    input_vector2 = poly_U2.transform(U2_k[k].reshape(1, -1))  # Transform current input
    B2_k = B2_coeffs @ input_vector2.T  # Input-dependent B

    D2_k = D2_coeffs
    # Compute next state (D remains independent)
    X2_next = (A2_k @ X2_current) + (B2_k.flatten()) + (D2_k @ d2_k[k])
    X2_pred.append(X2_next)
    X2_current = X2_next

# Convert predictions to a NumPy array
X2_pred = np.array(X2_pred)

# Calculate MAPE and R²
mae = mean_absolute_error(X2_k1, X2_pred)
print("Mean Absolute Error (MAE):", mae, "K")
from sklearn.metrics import r2_score
r2 = r2_score(X2_k1, X2_pred)
print("Coefficient of Determination (R²):", r2)

# Plot Observed vs Predicted States (Theater1)
plt.figure(figsize=(10, 6))
X2_k1[:, 0] = [x - 273.15 for x in X2_k1[:, 0]]
X2_pred[:, 0] = [x - 273.15 for x in X2_pred[:, 0]]
time = np.linspace(0, int(len(X2_pred[:, 0])/3600), len(X2_pred[:, 0]))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# First subplot: Theater1 temperature
ax1.plot(time, X2_k1[:, 0], lw=1.8, label="Measured", color="#DA70D6")
ax1.plot(time, X2_pred[:, 0], lw=1.3, label="Predicted", linestyle="dashed", color="#DA70D6")
ax1.set_ylabel(r"Heater 2 temperature $T_\text{h2}$ [°C]", fontsize=16)
ax1.tick_params(axis='both', labelsize=14)  # Increase tick size
ax1.legend(loc="lower left", fontsize=14)
ax1.grid()

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
ax3.plot(time, omega2[:-1], lw=1.8, label="Motor 2", color="#00008B", alpha=0.7)
ax3.set_ylabel(r"Rotational speed $\omega_\text{m2}$ [rpm]", fontsize=16)
ax3.tick_params(axis='y', labelsize=14)  # Increase tick size
ax3.legend(loc="upper right", fontsize=14)

ax2.tick_params(axis='x', labelsize=14)  # Increase x-axis tick size
ax2.grid()

# Set common x-label
ax2.set_xlabel("Time [h]", fontsize=16)
plt.tight_layout()

plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\transient\TCHP_Theater2.eps", format='eps', bbox_inches='tight')

# Show the plot
plt.show()



# # import numpy as np

# def predict_next_Theater2(X2_current, U2_current, D2_current, A2_coeffs, B2_coeffs, D2_coeffs, poly_U2):
#     # Compute A_k (parameter-independent in this case)
#     A2_k = A2_coeffs

#     # Compute parameter-dependent B matrix
#     input_vector = poly_U2.transform(U2_current.reshape(1, -1))  # Transform current input
#     B2_k = B2_coeffs @ input_vector.T  # Input-dependent B


#     D2_k = D2_coeffs
#     # Compute next state (D remains independent)
#     X2_next = (A2_k @ X2_current) + (B2_k.flatten()) + (D2_k @ D2_current)
#     return X2_next

# X_current = np.column_stack([450])
# # U_current = np.column_stack([6000, 100])

# # print(X_current)
# # print(U_current)
# import joblib

# A2_coeffs = joblib.load("coeffs/A2_coeffs.pkl")
# B2_coeffs = joblib.load("coeffs/B2_coeffs.pkl")
# D2_coeffs = joblib.load("coeffs/D2_coeffs.pkl")
# poly_U2 = joblib.load("coeffs/poly_U2.pkl")

# # # Predict the next state
# a_X_current = []
# j = 0
# for i in range(100000):
#     if i % 10000 == 0:
#         j = j + 1        
#     U_current = np.column_stack([9500])
#     D_current = np.column_stack([100 + j/10 * (140)])
#     X_current = predict_next_Theater2(X_current, U_current, D_current, A2_coeffs, B2_coeffs, D2_coeffs, poly_U2)
#     a_X_current.append(X_current[0][0])

# # print(X_next)
# plt.plot(a_X_current)
# plt.show()