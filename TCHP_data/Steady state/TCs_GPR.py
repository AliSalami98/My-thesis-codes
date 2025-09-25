import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Data loading and preprocessing remain unchanged
data = {
    "omegab [rpm]": [], "Tw_in [K]": [], "T1_in [K]": [], "p1_in [pa]": [], "p1_out [pa]": [],
    "Theater1 [K]": [], "omega1 [rpm]": [], "mdot1 [kg/s]": [], "Pcooling1 [W]": [], "T1_out [K]": [],
    "T2_in [K]": [], "p2_in [pa]": [], "p2_out [pa]": [], "Theater2 [K]": [], "omega2 [rpm]": [],
    "mdot [kg/s]": [], "Pcooling2 [W]": [], "T2_out [K]": [], "T3_in [K]": [], "p3_in [pa]": [],
    "p3_out [pa]": [], "omega3 [rpm]": [], "Pcooling3 [W]": [], "T3_out [K]": [], "PRt": [],
    "PR23": [], "Pcooling23 [W]": [], "Pcomb [W]": [], "Pmotor [W]": [], "Pheating [W]": [],
    "Prec_total [W]": [], "mw_dot [kg/s]": []
}

counter = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\comps 2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        counter += 1
        data['omegab [rpm]'].append(float(row['omegab [rpm]']))
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['T1_in [K]'].append(float(row['T1_in [°C]']) + 273.15)
        data['p1_in [pa]'].append(float(row['p1_in [bar]'])*10**5)
        data['p1_out [pa]'].append(float(row['p1_out [bar]'])*10**5)
        data['Theater1 [K]'].append(float(row['Theater1 [°C]']) + 273.15)
        data['omega1 [rpm]'].append(float(row['omega1 [rpm]']))
        data['mdot1 [kg/s]'].append(float(row['mdot1 [g/s]']) * 1e-3)
        data['Pcooling1 [W]'].append(float(row['Pcooling1 [W]']))
        data['T1_out [K]'].append(float(row['T1_out [°C]']) + 273.15)
        data['T2_in [K]'].append(float(row['T2_in [°C]']) + 273.15)
        data['p2_in [pa]'].append(float(row['p2_in [bar]'])*10**5)
        data['p2_out [pa]'].append(float(row['p2_out [bar]'])*10**5)
        data['Theater2 [K]'].append(float(row['Theater2 [°C]']) + 273.15)
        data['omega2 [rpm]'].append(float(row['omega2 [rpm]']))
        data['mdot [kg/s]'].append(float(row['mdot [g/s]']) * 1e-3)
        data['Pcooling2 [W]'].append(float(row['Pcooling2 [W]']))
        data['T2_out [K]'].append(float(row['T2_out [°C]']) + 273.15)
        data['T3_in [K]'].append(float(row['T3_in [°C]']) + 273.15)
        data['p3_in [pa]'].append(float(row['p3_in [bar]'])*10**5)
        data['p3_out [pa]'].append(float(row['p3_out [bar]'])*10**5)
        data['omega3 [rpm]'].append(float(row['omega3 [rpm]']))
        data['Pcooling3 [W]'].append(float(row['Pcooling3 [W]']))
        data['T3_out [K]'].append(float(row['T3_out2 [°C]']) + 273.15)
        data['PRt'].append(float(row['PRt']))
        data['PR23'].append(data['p3_out [pa]'][-1]/data['p2_in [pa]'][-1])
        data['Pcooling23 [W]'].append(data['Pcooling2 [W]'][-1] + data['Pcooling3 [W]'][-1])
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Prec_total [W]'].append(float(row['Prec_total [W]']))
        data['mw_dot [kg/s]'].append(float(row['mw_dot [l/min]']))

# Feature engineering
Pr1 = np.array([s/r for s,r in zip(data['p1_out [pa]'], data['p1_in [pa]'])])
Pr2 = np.array([s/r for s,r in zip(data['p2_out [pa]'], data['p2_in [pa]'])])
Pr3 = np.array([s/r for s,r in zip(data['p3_out [pa]'], data['p3_in [pa]'])])
Tr1 = np.array([s/r for s,r in zip(data['Theater1 [K]'], data['Tw_in [K]'])])
Tr2 = np.array([s/r for s,r in zip(data['Theater2 [K]'], data['Tw_in [K]'])])
w1 = np.array(data['omega1 [rpm]'])
w2 = np.array(data['omega2 [rpm]'])
w3 = np.array(data['omega3 [rpm]'])
omegab = np.array(data['omegab [rpm]'])
Theater1 = np.array(data['Theater1 [K]'])
Theater2 = np.array(data['Theater2 [K]'])

X1_array = np.array([Pr1, data['Theater1 [K]'], data['Tw_in [K]'], w1])
X2_array = np.array([Pr2, data['Theater2 [K]'], data['Tw_in [K]'], w2])
X3_array = np.array([data['PR23'], data['Theater2 [K]'], data['Tw_in [K]'], w2])
X4_array = np.array([w1, w2, w3])

X1 = X1_array.T
X2 = X2_array.T
X3 = X3_array.T
X4 = X4_array.T

y1_1 = np.array(data['mdot1 [kg/s]'])
y1_2 = np.array(data['T1_out [K]'])
y1_3 = np.array(data['Pcooling1 [W]'])
y2_1 = np.array(data['mdot [kg/s]'])
y2_2 = np.array(data['T2_out [K]'])
y2_3 = np.array(data['Pcooling2 [W]'])
y3_1 = np.array(data['mdot [kg/s]'])
y3_2 = np.array(data['T3_out [K]'])
y3_3 = np.array(data['Pcooling23 [W]'])
y4 = np.array(data['Pmotor [W]'])

# Train-test split
X1_1_train, X1_1_test, y1_1_train, y1_1_test = train_test_split(X1, y1_1, test_size=0.01, random_state=0)
X1_2_train, X1_2_test, y1_2_train, y1_2_test = train_test_split(X1, y1_2, test_size=0.01, random_state=0)
X1_3_train, X1_3_test, y1_3_train, y1_3_test = train_test_split(X1, y1_3, test_size=0.01, random_state=0)
X2_1_train, X2_1_test, y2_1_train, y2_1_test = train_test_split(X2, y2_1, test_size=0.01, random_state=0)
X2_2_train, X2_2_test, y2_2_train, y2_2_test = train_test_split(X2, y2_2, test_size=0.01, random_state=0)
X2_3_train, X2_3_test, y2_3_train, y2_3_test = train_test_split(X2, y2_3, test_size=0.01, random_state=0)
X3_1_train, X3_1_test, y3_1_train, y3_1_test = train_test_split(X3, y3_1, test_size=0.01, random_state=0)
X3_2_train, X3_2_test, y3_2_train, y3_2_test = train_test_split(X3, y3_2, test_size=0.01, random_state=0)
X3_3_train, X3_3_test, y3_3_train, y3_3_test = train_test_split(X3, y3_3, test_size=0.01, random_state=0)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.01, random_state=0)

# Define GPR kernel (Constant kernel * RBF kernel)
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

# Initialize and train GPR models
model1_1_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model1_1_GPR.fit(X1_1_train, y1_1_train)

model1_2_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model1_2_GPR.fit(X1_2_train, y1_2_train)

model1_3_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model1_3_GPR.fit(X1_3_train, y1_3_train)

model2_1_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model2_1_GPR.fit(X2_1_train, y2_1_train)

model2_2_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model2_2_GPR.fit(X2_2_train, y2_2_train)

model2_3_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model2_3_GPR.fit(X2_3_train, y2_3_train)

model3_1_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model3_1_GPR.fit(X3_1_train, y3_1_train)

model3_2_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model3_2_GPR.fit(X3_2_train, y3_2_train)

model3_3_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model3_3_GPR.fit(X3_3_train, y3_3_train)

model4_GPR = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0)
model4_GPR.fit(X4_train, y4_train)

# Predictions
y1_1_pred = model1_1_GPR.predict(X1)
y1_2_pred = model1_2_GPR.predict(X1)
y1_3_pred = model1_3_GPR.predict(X1)
y2_1_pred = model2_1_GPR.predict(X2)
y2_2_pred = model2_2_GPR.predict(X2)
y2_3_pred = model2_3_GPR.predict(X2)
y3_1_pred = model3_1_GPR.predict(X3)
y3_2_pred = model3_2_GPR.predict(X3)
y3_3_pred = model3_3_GPR.predict(X3)
y4_pred = model4_GPR.predict(X4)

# Parity plot function (unchanged)
def parity_plot_with_metrics(y_true, y_pred, variable_name):
    if "Temperature" in variable_name or "[K]" in variable_name:
        y_true = np.array(y_true) - 273.15
        y_pred = np.array(y_pred) - 273.15
        unit = "°C"
        show_rmse = True
    else:
        unit = variable_name.split("[")[-1].replace("]", "")
        show_rmse = False
    
    if show_rmse:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue', label='Predicted vs Measured')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Ideal')

    if "Temperature" in variable_name or "°C" in unit:
        deviation = 2
        plt.plot(y_true, y_true + deviation, color='gray', linestyle='--', label='±2°C Deviation')
        plt.plot(y_true, y_true - deviation, color='gray', linestyle='--')
    else:
        plt.plot(y_true, 1.1 * y_true, color='gray', linestyle='--', label='±10% Deviation')
        plt.plot(y_true, 0.9 * y_true, color='gray', linestyle='--')

    if show_rmse:
        plt.text(0.05, 0.95, f"RMSE: {rmse:.2f} °C", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color="darkblue")
    else:
        plt.text(0.05, 0.95, f"MAPE: {mape:.2f}%", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color="darkblue")
    plt.text(0.05, 0.90, f"R²: {r2:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color="darkblue")

    plt.xlabel(f"Measured {variable_name.split('[')[0]} [{unit}]", fontsize=14)
    plt.ylabel(f"Predicted {variable_name.split('[')[0]} [{unit}]", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(fontsize=12)
    plt.tight_layout()

# Generate parity plots
parity_plot_with_metrics(y1_1, y1_1_pred, "TC1 Mass Flow Rate [kg/s]")
parity_plot_with_metrics(y1_2, y1_2_pred, "TC1 Outlet Temperature [K]")
parity_plot_with_metrics(y1_3, y1_3_pred, "TC1 Cooler Power [W]")
parity_plot_with_metrics(y3_1, y3_1_pred, "TC23 Mass Flow Rate [kg/s]")
parity_plot_with_metrics(y3_2, y3_2_pred, "TC23 Outlet Temperature [K]")
parity_plot_with_metrics(y3_3, y3_3_pred, "TC23 Cooler Power [W]")
parity_plot_with_metrics(y4, y4_pred, "TC Motor Power [W]")

plt.show()

# Create a synthetic test grid
Pr1_grid = np.linspace(min(Pr1), max(Pr1), 20)
T_heater_grid = np.linspace(min(Theater1), max(Theater1), 20)
Tw_in_grid = np.linspace(min(data['Tw_in [K]']), max(data['Tw_in [K]']), 20)
omega1_grid = np.linspace(min(w1), max(w1), 20)

# Generate full test matrix (brute-force all combinations)
import itertools
X_synthetic = np.array(list(itertools.product(Pr1_grid, T_heater_grid, Tw_in_grid, omega1_grid)))

# Optional: reduce size if needed
X_synthetic = X_synthetic[np.random.choice(len(X_synthetic), size=200, replace=False)]

# Predict using trained model
y1_1_synthetic_pred, y1_1_synthetic_std = model1_1_GPR.predict(X_synthetic, return_std=True)

# Visualize one projection: e.g., omega1 vs predicted mdot1
plt.figure(figsize=(8,6))
plt.scatter(X_synthetic[:, 1], y1_1_synthetic_pred, c=y1_1_synthetic_std, cmap='coolwarm', s=50)
plt.colorbar(label="Prediction Uncertainty (±σ)")
plt.xlabel("omega1 [rpm]")
plt.ylabel("Predicted mdot1 [kg/s]")
plt.title("Predicted Mass Flow Rate vs omega1 (GPR model)")
plt.grid(True)
plt.tight_layout()
plt.show()
