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
import joblib
import os

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
    "Tsuc [K]": [], "psuc [pa]": [], "pdis [pa]": [],
    "Theater [K]": [], "Tw_in [K]": [], "omega [rpm]": [],
    "mdot [g/s]": [], "Pcomb [kW]": [], "Pheating [kW]": [],
    "Pcooling [kW]": [], "Pmotor [kW]": [], "Tdis [°C]": [],
    "Pcomp [kW]": [], "eff [%]": [], "Ploss [kW]": [], "Pmech [kW]": []
}

with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\Tc experiments\all_filtered_2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        data['Tsuc [K]'].append(float(row['Tsuc [°C]']) + 273.15)
        data['psuc [pa]'].append(float(row['psuc [bar]']) * 1e5)
        data['pdis [pa]'].append(float(row['pdis [bar]']) * 1e5)
        data['Theater [K]'].append(float(row['Theater [°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        data['mdot [g/s]'].append(float(row['mdot [g/s]']))
        data['Pcomb [kW]'].append(float(row['Pcomb [W]']) * 1e-3)
        data['Pheating [kW]'].append(float(row['Pheating [W]']) * 1e-3)
        data['Pcooling [kW]'].append(float(row['Pcooling [W]']) * 1e-3)
        data['Pmotor [kW]'].append(float(row['Pmotor [W]']) * 1e-3)
        data['Pmech [kW]'].append(float(row['Pmotor [W]']) * 0.9 * 1e-3)
        data['Tdis [°C]'].append(float(row['Tdis [°C]']))
        data['Pcomp [kW]'].append(float(row['Pcomp [W]']) * 1e-3)
        # data['eff [%]'].append(100 * float(row['eff [%]']))
        data['Ploss [kW]'].append(float(row['Ploss [W]']) * 1e-3)

# -------------------------
# 4. DEFINE INPUTS X AND OUTPUTS Y
# -------------------------
Pr = [s/r for s, r in zip(data['pdis [pa]'], data['psuc [pa]'])]
Tr = [s/r for s, r in zip(data['Theater [K]'], data['Tw_in [K]'])]
p_charged = [np.sqrt(s * r) for s, r in zip(data['pdis [pa]'], data['psuc [pa]'])]

# X = np.column_stack([data['psuc [pa]'], data['pdis [pa]'], data['omega [rpm]'], data['Theater [K]'], data['Tw_in [K]'], data['Tsuc [K]']])
X = np.column_stack([Pr, Tr, p_charged, data['omega [rpm]'], data['Tsuc [K]']])
y_names = ['mdot [g/s]', 'Pheating [kW]', 'Pcooling [kW]', 'Pmech [kW]', 'Tdis [°C]', 'Ploss [kW]']
y = [np.array(data[name]) for name in y_names]

# -------------------------
# 5. NORMALIZATION
# -------------------------
scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
X_scaled = scaler_X.fit_transform(X)
scaler_y, y_scaled = [], []

for yi in y:
    s = MinMaxScaler(feature_range=(0.1, 0.9))
    scaler_y.append(s)
    y_scaled.append(s.fit_transform(yi.reshape(-1, 1)).flatten())

# -------------------------
# 6. TRAIN/TEST SPLIT
# -------------------------
X_train, X_test, idx_train, idx_test = train_test_split(X_scaled, np.arange(len(X_scaled)), test_size=0.2, random_state=0)
n_features = X.shape[1]
y_train = [y[idx_train] for y in y_scaled]
y_test = [y[idx_test] for y in y_scaled]

# -------------------------
# 7. GPR TRAINING & OUTLIER DETECTION ON FULL DATA
# -------------------------
model_dir = "saved_gpr_models"
os.makedirs(model_dir, exist_ok=True)
z_threshold = 2.1
# z_threshold = 2.326  # for 90% CI → more restrictive

model_filenames = [f"{model_dir}/gpr_model_{name.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '')}.pkl" for name in y_names]
scaler_filenames = [f"{model_dir}/scaler_y_{name.replace(' ', '_').replace('[', '').replace(']', '').replace('/', '')}.pkl" for name in y_names]
scaler_X_filename = f"{model_dir}/scaler_X.pkl"
joblib.dump(scaler_X, scaler_X_filename)

gp_models, predictions, STD, outliers_all = [], [], [], []

for i, name in enumerate(y_names):
    print(f"\nProcessing: {name}")
    model_path = model_filenames[i]
    scaler_path = scaler_filenames[i]
    y_full = y_scaled[i]

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler_y[i] = joblib.load(scaler_path)
    else:
        kernel = RBF(length_scale=[1.0]*n_features, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0)
        model.fit(X_scaled, y_full)
        joblib.dump(model, model_path)
        joblib.dump(scaler_y[i], scaler_path)

    gp_models.append(model)
    y_pred_full, y_std_full = model.predict(X_scaled, return_std=True)
    predictions.append(y_pred_full)
    STD.append(y_std_full)

    outliers = np.abs(y_full - y_pred_full) > z_threshold * y_std_full
    outliers_all.append(outliers)

    outlier_indices = np.where(outliers)[0]  # <-- THIS LINE ADDED
    print(f"Outliers in {name}: {len(outlier_indices)} out of {len(y_full)}")
    print(f"Outlier indices for {name}: {outlier_indices.tolist()}")  # <-- THIS LINE ADDED

# -------------------------
# 7b. REMOVE UNION OF OUTLIERS
# -------------------------
# Step 1: Get union of all outlier indices
all_outlier_indices = set()
for outliers in outliers_all:
    all_outlier_indices.update(np.where(outliers)[0])

all_outlier_indices = sorted(list(all_outlier_indices))
print(f"\nTotal unique outlier samples to remove: {len(all_outlier_indices)}")
print(f"Indices to remove: {all_outlier_indices}")

# Step 2: Create mask to keep non-outliers
mask = np.ones(X_scaled.shape[0], dtype=bool)
mask[all_outlier_indices] = False

# Step 3: Filter all data arrays
X_filtered = X_scaled[mask]
y_filtered = [yi[mask] for yi in y_scaled]

# -------------------------
# 8. TRAIN/TEST SPLIT ON FILTERED DATA
# -------------------------
X_train_filt, X_test_filt, idx_train_filt, idx_test_filt = train_test_split(
    X_filtered, np.arange(len(X_filtered)), test_size=0.2, random_state=0
)
y_train_filt = [y[idx_train_filt] for y in y_filtered]
y_test_filt = [y[idx_test_filt] for y in y_filtered]
# -------------------------
# 9. RETRAIN GPR MODELS ON FILTERED DATA
# -------------------------
gp_models_filtered = []
predictions_test_filt = []
print("\n--- Retraining on filtered data ---")

for i, name in enumerate(y_names):
    kernel = RBF(length_scale=[1.0]*n_features, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0)
    model.fit(X_train_filt, y_train_filt[i])
    gp_models_filtered.append(model)

    # Predict on test set
    y_pred_test_filt, _ = model.predict(X_test_filt, return_std=True)
    predictions_test_filt.append(y_pred_test_filt)
print("\n--- Filtered Model Performance on Test Set ---")
for i, name in enumerate(y_names):
    y_pred_real = scaler_y[i].inverse_transform(predictions_test_filt[i].reshape(-1, 1)).flatten()
    y_true_real = scaler_y[i].inverse_transform(y_test_filt[i].reshape(-1, 1)).flatten()

    rmse = root_mean_squared_error(y_true_real, y_pred_real)
    mape = mean_absolute_percentage_error(y_true_real, y_pred_real) * 100
    r2 = r2_score(y_true_real, y_pred_real) * 100

    print(f"{name}: RMSE = {rmse:.2f}, MAPE = {mape:.2f}%, R² = {r2:.2f}%")

# -------------------------
# 11. SAVE FILTERED DATASET TO CSV
# -------------------------
# Path to save
output_path = r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\Tc experiments\all_filtered_3.csv'

# Reconstruct the filtered DataFrame
filtered_data = {
    'Tsuc [°C]': np.array(data['Tsuc [K]'])[mask] - 273.15,
    'psuc [bar]': np.array(data['psuc [pa]'])[mask] * 1e-5,
    'pdis [bar]': np.array(data['pdis [pa]'])[mask] * 1e-5,
    'Theater [°C]': np.array(data['Theater [K]'])[mask] - 273.15,
    'Tw_in [°C]': np.array(data['Tw_in [K]'])[mask] - 273.15,
    'omega [rpm]': np.array(data['omega [rpm]'])[mask],
    'mdot [g/s]': np.array(data['mdot [g/s]'])[mask],
    'Pcomb [W]': np.array(data['Pcomb [kW]'])[mask] * 1e3,
    'Pheating [W]': np.array(data['Pheating [kW]'])[mask] * 1e3,
    'Pcooling [W]': np.array(data['Pcooling [kW]'])[mask] * 1e3,
    'Pmotor [W]': np.array(data['Pmotor [kW]'])[mask] * 1e3,
    'Pmech [W]': np.array(data['Pmech [kW]'])[mask] * 1e3,
    'Tdis [°C]': np.array(data['Tdis [°C]'])[mask],
    'Pcomp [W]': np.array(data['Pcomp [kW]'])[mask] * 1e3,
    # 'eff [%]': np.array(data['eff [%]'])[mask] / 100,
    'Ploss [W]': np.array(data['Ploss [kW]'])[mask] * 1e3,
}

# Convert to DataFrame
df_filtered = pd.DataFrame(filtered_data)

# Save to CSV with same format
df_filtered.to_csv(output_path, sep=';', index=False)
print(f"\n✅ Filtered dataset saved to: {output_path}")

# -------------------------
# 8. PLOTTING FUNCTION
# -------------------------
from scipy.stats import norm
labels = [
    r'Mass flow rate $\dot{m}_{\text{f}}$',
    r'Heater heat transfer rate $\dot{Q}_{\text{h}}$',
    r'Cooler heat transfer rate $\dot{Q}_{\text{k}}$',
    r'Mechanical Power $\dot{W}_{\text{mech}}$',
    r'Discharge Temperature $T_{\text{dis}}$',
    r'Heat Loss $\dot{Q}_{\text{loss}}$'
]

filenames_for_save = [
    "mdot_outliers",
    "Qheater_outliers",
    "Qcooler_outliers",
    "Wmech_outliers",
    "Tdis_outliers",
    "Qloss_outliers"
]

units = [
    r'[g/s]',     # for $\dot{m}_{f}$
    r'[kW]',        # for $\dot{Q}_{heater}$
    r'[kW]',        # for $\dot{Q}_{cooler}$
    r'[kW]',        # for $P_{elec}$
    r'[°C]',       # for $T_{dis}$
    r'[kW]'         # for $\dot{Q}_{loss}$
]
def plot_unscaled_outliers(y_true_real, y_pred_real, y_std_real, outlier_mask, label, unit, filename, z_threshold):
    from scipy.stats import norm
    x = np.arange(len(y_true_real))
    plt.figure()  # Slightly larger figure for better resolution

    ci_percent = int((norm.cdf(z_threshold) - norm.cdf(-z_threshold)) * 100)
    plt.fill_between(x, y_pred_real - z_threshold * y_std_real, y_pred_real + z_threshold * y_std_real,
                     color='lightgray', label=f'{ci_percent}% Confidence Interval')

    plt.plot(x, y_pred_real, 'k-', label='GP fit', linewidth=2)
    plt.scatter(x[~outlier_mask], y_true_real[~outlier_mask], color='blue', s=40, label='Normal')
    plt.scatter(x[outlier_mask], y_true_real[outlier_mask], color='red', s=60, label='Outliers')

    for xi, yt, yp, is_out in zip(x, y_true_real, y_pred_real, outlier_mask):
        if is_out:
            plt.annotate('', xy=(xi, yt), xytext=(xi, yp),
                         arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel(f"{label} {unit}", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    if filename == 'Qheater_outliers':
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "Qheater_outliers.eps"), format='pdf')
    else:
        plt.tight_layout()
        full_path = os.path.join(save_dir, f"{filename}.eps")
        plt.savefig(full_path, format='eps')
        print(f"✅ Saved pdf plot to: {full_path}")


# Define your target folder for saving plots
save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_data\outliers'
os.makedirs(save_dir, exist_ok=True)

# -------------------------
# 9. PLOT ALL OUTLIERS
# -------------------------
for i in range(len(y_names)):
    y_scaled_i = np.array(y_scaled[i]).reshape(-1, 1)
    pred_scaled = np.array(predictions[i]).reshape(-1, 1)
    std_scaled = np.array(STD[i]).reshape(-1, 1)

    y_real = scaler_y[i].inverse_transform(y_scaled_i).flatten()
    y_pred_real = scaler_y[i].inverse_transform(pred_scaled).flatten()
    y_std_real = scaler_y[i].inverse_transform(pred_scaled + std_scaled).flatten() - y_pred_real

    plot_unscaled_outliers(
        y_true_real=y_real,
        y_pred_real=y_pred_real,
        y_std_real=y_std_real,
        outlier_mask=outliers_all[i],
        label=labels[i],
        unit=units[i],
        filename=filenames_for_save[i],
        z_threshold = z_threshold
    )

plt.show()




# -------------------------
# 10. PERFORMANCE METRICS ON TEST SET
# -------------------------
print("\n--- Model Performance on Test Set ---")
for i, name in enumerate(y_names):
    model = gp_models[i]
    y_pred_test, _ = model.predict(X_test, return_std=True)
    y_true_real = scaler_y[i].inverse_transform(y_test[i].reshape(-1, 1)).flatten()
    y_pred_real = scaler_y[i].inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

    rmse = root_mean_squared_error(y_true_real, y_pred_real)
    mape = mean_absolute_percentage_error(y_true_real, y_pred_real) * 100
    r2 = r2_score(y_true_real, y_pred_real) * 100

    print(f"{name}: RMSE = {rmse:.2f}, MAPE = {mape:.2f}%, R² = {r2:.2f}%")
