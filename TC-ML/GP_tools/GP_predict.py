# -------------------------
# 1. IMPORTS
# -------------------------
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error

# -------------------------
# 2. SETUP CO2 REFERENCE STATE
# -------------------------
T0 = 298.15
p0 = 101325
state = AbstractState("HEOS", "CO2")
state.update(CP.PT_INPUTS, p0, T0)
h0 = state.hmass()
s0 = state.smass()
CP.set_reference_state('CO2', T0, p0, h0, s0)

# -------------------------
# 3. READ THE DATA
# -------------------------
data = {
    "Tin [K]": [], "pin [pa]": [], "pout [pa]": [],
    "Th_wall [K]": [], "Tw_in [K]": [], "omega [rpm]": [],
    "mdot [kg/s]": [], "Pcomb [W]": [], "Pheating [W]": [],
    "Pcooling [W]": [], "Pmotor [W]": [], "Tout [°C]": [],
    "Pout [W]": [], "eff [%]": [], "Ploss [W]": [],
}

with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\I-O data4_filtered.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        data['Tin [K]'].append(float(row['Tin [°C]']) + 273.15)
        data['pin [pa]'].append(float(row['pin [bar]']) * 1e5)
        data['pout [pa]'].append(float(row['pout [bar]']) * 1e5)
        data['Th_wall [K]'].append(float(row['Th_wall[°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        data['mdot [kg/s]'].append(float(row['mdot [g/s]']) * 1e-3)
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Tout [°C]'].append(float(row['Tout [°C]']))
        data['Pout [W]'].append(float(row['Pout [W]']))
        data['eff [%]'].append(100 * float(row['eff [%]']))
        data['Ploss [W]'].append(data['Pheating [W]'][-1] + data['Pmotor [W]'][-1] - data['Pcooling [W]'][-1] - data['Pout [W]'][-1])


# -------------------------
# 4. DEFINE INPUTS X AND OUTPUTS Y
# -------------------------
Pr = [s/r for s, r in zip(data['pout [pa]'], data['pin [pa]'])]
Tr = [s/r for s, r in zip(data['Th_wall [K]'], data['Tw_in [K]'])]
p_charged = [np.sqrt(s * r) for s, r in zip(data['pout [pa]'], data['pin [pa]'])]

# Build X dynamically
X = np.column_stack([data['pin [pa]'], data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]']])
# X = np.column_stack([Pr, Tr, p_charged, data['omega [rpm]'], data['Tin [K]']])  # 4 inputs now
y_names = ['mdot [kg/s]', 'Pheating [W]', 'Pcooling [W]', 'Pmotor [W]', 'Tout [°C]', 'Ploss [W]']

y = [np.array(data[name]) for name in y_names]

# -------------------------
# 5. NORMALIZATION
# -------------------------
scaler_X = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler_X.fit_transform(X)

scaler_y = []
y_scaled = []
for yi in y:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_y.append(scaler)
    y_scaled.append(scaler.fit_transform(yi.reshape(-1, 1)).flatten())

# -------------------------
# 6. SPLIT DATA
# -------------------------
X_train, X_test, idx_train, idx_test = train_test_split(X_scaled, np.arange(len(X_scaled)), test_size=0.2, random_state=0)

y_train = [y[idx_train] for y in y_scaled]
y_test = [y[idx_test] for y in y_scaled]

# -------------------------
# 7. DEFINE GP REGRESSORS (ADAPTED TO X SHAPE!)
# -------------------------
n_features = X.shape[1]

kernels = [RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
           for _ in range(len(y_train))]

gp_models = [
    GaussianProcessRegressor(kernel=k, n_restarts_optimizer=10, alpha=0)
    for k in kernels
]

# -------------------------
# 8. TRAIN
# -------------------------
for model, yt in zip(gp_models, y_train):
    model.fit(X_train, yt)

# -------------------------
# 9. PREDICT
# -------------------------
predictions = []
for model in gp_models:
    y_pred, _ = model.predict(X_test, return_std=True)
    predictions.append(y_pred)

# -------------------------
# 10. PLOT RESULTS
# -------------------------
labels = ['$\dot{m}_{f}$ [kg/s]', '$\dot{Q}_{heater}$ [W]', '$\dot{Q}_{cooler}$ [W]', '$P_{motor}$ [W]', '$T_{out}$ [°C]', '$\dot{Q}_{loss}$ [W]']

metrics_summary = []

for i in range(len(labels)):
    y_true_real = scaler_y[i].inverse_transform(y_test[i].reshape(-1,1)).flatten()
    y_pred_real = scaler_y[i].inverse_transform(predictions[i].reshape(-1,1)).flatten()

    rmse = root_mean_squared_error(y_true_real, y_pred_real)
    mape = mean_absolute_percentage_error(y_true_real, y_pred_real) * 100
    r2 = r2_score(y_true_real, y_pred_real) * 100

    metrics_summary.append((labels[i], mape, r2))

    # Sorting for plotting
    y_true_sorted = np.sort(y_true_real)

    plt.figure(figsize=(7,6))
    plt.plot(y_true_real, y_true_real, 'k-', label='45° line')

    if '$T_{out}$ [°C]' in labels[i]:
        plt.fill_between(y_true_sorted, y_true_sorted-2, y_true_sorted+2, color='lightgray', label='±2 K band')
        plt.scatter(y_true_real, y_pred_real, edgecolors='red', facecolors='none', label=f'GPR (RMSE={rmse:.2f} K, $R^2$={r2:.2f}%)')

    else:
        plt.fill_between(y_true_sorted, 0.95*y_true_sorted, 1.05*y_true_sorted, color='lightgray', label='±5% band')
        plt.scatter(y_true_real, y_pred_real, edgecolors='red', facecolors='none', label=f'GPR (MAPE={mape:.2f}%, $R^2$={r2:.2f}%)')
    plt.xlabel(f'Measured {labels[i]}', fontsize=14)
    plt.ylabel(f'Predicted {labels[i]}', fontsize=14)
    # plt.title(f'Prediction for {labels[i]}', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# plt.show()
# -------------------------
# 11. PRINT METRICS
# -------------------------
print("\n--- GPR Evaluation Metrics ---")
for label, mape, r2 in metrics_summary:
    print(f"{label}: MAPE = {mape:.2f}%, R² = {r2:.2f}%")