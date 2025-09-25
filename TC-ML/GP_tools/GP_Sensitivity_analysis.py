# -------------------------
# 1. IMPORTS
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import MinMaxScaler
import csv
# -------------------------
# 2. LOAD DATA
# -------------------------
data = {
    "Tin [K]": [], "pin [pa]": [], "pout [pa]": [],
    "Th_wall [K]": [], "Tw_in [K]": [], "omega [rpm]": [],
    "mdot [kg/s]": [], "Pcomb [W]": [], "Pheating [W]": [],
    "Pcooling [W]": [], "Pmotor [W]": [], "Tout [°C]": [],
    "Pout [W]": [], "Ploss [W]": [], "eff [%]": []
}

with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\I-O data4_filtered2.csv') as csv_file:    
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

features = ['pin [pa]', 'pout [pa]', 'Th_wall [K]', 'Tw_in [K]', 'omega [rpm]', 'Tin [K]']
X = np.column_stack([data['pin [pa]'], data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]'], data['Tw_in [K]'], data['Tin [K]']])

# features = ['Pr', 'Tr', 'p_charged', 'omega [rpm]', 'Tin [K]']
# X = np.column_stack([Pr, Tr, p_charged, data['omega [rpm]'], data['Tin [K]']])  # 4 inputs now

outputs = {
    '$\dot{m}_f$ [kg/s]': np.array(data['mdot [kg/s]']),
    '$\dot{Q}_{heater}$ [W]': np.array(data['Pheating [W]']),
    '$\dot{Q}_{cooler}$ [W]': np.array(data['Pcooling [W]']),
    '$P_{motor}$ [W]': np.array(data['Pmotor [W]']),
    '$T_{dis}$ [°C]': np.array(data['Tout [°C]'])
}

# -------------------------
# 4. NORMALIZATION
# -------------------------
scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
X_scaled = scaler_X.fit_transform(X)

output_scalers = {}
outputs_scaled = {}

for name, y in outputs.items():
    scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))
    outputs_scaled[name] = scaler_y.fit_transform(y.reshape(-1,1)).flatten()
    output_scalers[name] = scaler_y

# -------------------------
# 5. TRAIN GP MODELS
# -------------------------
gp_models = {}
length_scales = {}
n_features = X.shape[1]

for name, y_scaled in outputs_scaled.items():
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=0)

    kernel = RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0)
    gpr.fit(X_train, y_train)

    gp_models[name] = gpr
    length_scales[name] = gpr.kernel_.k1.length_scale  # Only RBF part (not WhiteKernel)

# -------------------------
# 6. SENSITIVITY ANALYSIS
# -------------------------
# Sensitivity = 1 / (length_scale^2)
sensitivity_results = {}

for output_name, ls in length_scales.items():
    sensitivity = 1.0 / (np.array(ls)**2)
    sensitivity_results[output_name] = sensitivity

# -------------------------
# 7. PLOT RESULTS
# -------------------------
for output_name, sensitivity in sensitivity_results.items():
    sensitivity_series = pd.Series(sensitivity, index=features)

    plt.figure(figsize=(10, 6))
    sensitivity_series.sort_values(ascending=False).plot(kind='bar', color='steelblue')
    plt.title(f'Sensitivity Analysis for {output_name}', fontsize=18)
    plt.ylabel('Sensitivity Index ($1/\\ell^2$)', fontsize=14)
    plt.xlabel('Input Variables', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y')
    plt.tight_layout()

plt.show()

# -------------------------
# 8. OPTIONAL: SUMMARY TABLE
# -------------------------
summary_table = pd.DataFrame({output: pd.Series(1/(ls**2), index=features) for output, ls in length_scales.items()})
summary_table = summary_table.round(3)
print(summary_table)
