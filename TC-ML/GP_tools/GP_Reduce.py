import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
# Create an AbstractState object using the HEOS backend and CO2
T0 = 298.15
p0 = 101325
state = AbstractState("HEOS", "CO2")
state.update(CP.PT_INPUTS, p0, T0)
h0 = state.hmass()
s0 = state.smass()
CP.set_reference_state('CO2',T0, p0, h0, s0)

data = {
    "Tin [K]": [],
    "pin [pa]": [],
    "pout [pa]": [],
    "Th_wall [K]": [],
    "Tw_in [K]": [],
    "omega [rpm]": [],
    "mdot [kg/s]": [],
    "Pcomb [W]": [],
    "Pheating [W]": [],
    "Pcooling [W]": [],
    "Pmotor [W]": [],
    "Tout [°C]": [],
    "Pout [W]": [],
    "eff [%]": [],
}
counter = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\I-O data4_filtered.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        counter += 1
        data['Tin [K]'].append(float(row['Tin [°C]']) + 273.15)
        data['pin [pa]'].append(float(row['pin [bar]'])*10**5)
        data['pout [pa]'].append(float(row['pout [bar]'])*10**5)
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
        data['eff [%]'].append(100*float(row['eff [%]']))
        # if counter > 52:
        #     break

Pr = [s/r for s,r in zip(data['pout [pa]'][:], data['pin [pa]'][:])]
Tr = [s/r for s,r in zip(data['Th_wall [K]'][:], data['Tw_in [K]'][:])]

X1_array = np.array([data['pin [pa]']])
X2_array = np.array([data['pin [pa]'],data['pout [pa]']])
X3_array = np.array([data['pin [pa]'],data['pout [pa]'],data['omega [rpm]']])
X4_array = np.array([data['pin [pa]'],data['pout [pa]'],data['omega [rpm]'], data['Th_wall [K]']])
X5_array = np.array([data['pin [pa]'],data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]'], data['Tw_in [K]']])
X6_array = np.array([data['pin [pa]'],data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]'], data['Tw_in [K]'],data['Tin [K]']])

X1 = X1_array.T
X2 = X2_array.T
X3 = X3_array.T
X4 = X4_array.T
X5 = X5_array.T
X6 = X6_array.T

y1 = np.array(data['mdot [kg/s]'])
y2 = np.array(data['Pheating [W]'])
y3 = np.array(data['Pcooling [W]'])
y4 = np.array(data['Pmotor [W]'])
y5 = np.array(data['Tout [°C]'])

from sklearn.preprocessing import StandardScaler,  MinMaxScaler
scaler = MinMaxScaler(feature_range=(0.1, 0.9))

# # Normalize X
X1 = scaler.fit_transform(X1)
X2 = scaler.fit_transform(X2)
X3 = scaler.fit_transform(X3)
X4 = scaler.fit_transform(X4)
X5 = scaler.fit_transform(X5)
X6 = scaler.fit_transform(X6)

# # Normalize y1, y2, y3, y4, y5
y1 = scaler.fit_transform(y1.reshape(-1, 1)).flatten()
y2 = scaler.fit_transform(y2.reshape(-1, 1)).flatten()
y3= scaler.fit_transform(y3.reshape(-1, 1)).flatten()
y4 = scaler.fit_transform(y4.reshape(-1, 1)).flatten()
y5 = scaler.fit_transform(y5.reshape(-1, 1)).flatten()

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
X2_train, X2_test, y1_train, y1_test = train_test_split(X2, y1, test_size = 0.2, random_state = 0)
X3_train, X3_test, y1_train, y1_test = train_test_split(X3, y1, test_size = 0.2, random_state = 0)
X4_train, X4_test, y1_train, y1_test = train_test_split(X4, y1, test_size = 0.2, random_state = 0)
X5_train, X5_test, y1_train, y1_test = train_test_split(X5, y1, test_size = 0.2, random_state = 0)
X6_train, X6_test, y1_train, y1_test = train_test_split(X6, y1, test_size = 0.2, random_state = 0)

X1_train, X1_test, y2_train, y2_test = train_test_split(X1, y2, test_size = 0.2, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)
X3_train, X3_test, y2_train, y2_test = train_test_split(X3, y2, test_size = 0.2, random_state = 0)
X4_train, X4_test, y2_train, y2_test = train_test_split(X4, y2, test_size = 0.2, random_state = 0)
X5_train, X5_test, y2_train, y2_test = train_test_split(X5, y2, test_size = 0.2, random_state = 0)
X6_train, X6_test, y2_train, y2_test = train_test_split(X6, y2, test_size = 0.2, random_state = 0)

X1_train, X1_test, y3_train, y3_test = train_test_split(X1, y3, test_size = 0.2, random_state = 0)
X2_train, X2_test, y3_train, y3_test = train_test_split(X2, y3, test_size = 0.2, random_state = 0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.2, random_state = 0)
X4_train, X4_test, y3_train, y3_test = train_test_split(X4, y3, test_size = 0.2, random_state = 0)
X5_train, X5_test, y3_train, y3_test = train_test_split(X5, y3, test_size = 0.2, random_state = 0)
X6_train, X6_test, y3_train, y3_test = train_test_split(X6, y3, test_size = 0.2, random_state = 0)

X1_train, X1_test, y4_train, y4_test = train_test_split(X1, y4, test_size = 0.2, random_state = 0)
X2_train, X2_test, y4_train, y4_test = train_test_split(X2, y4, test_size = 0.2, random_state = 0)
X3_train, X3_test, y4_train, y4_test = train_test_split(X3, y4, test_size = 0.2, random_state = 0)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.2, random_state = 0)
X5_train, X5_test, y4_train, y4_test = train_test_split(X5, y4, test_size = 0.2, random_state = 0)
X6_train, X6_test, y4_train, y4_test = train_test_split(X6, y4, test_size = 0.2, random_state = 0)

X1_train, X1_test, y5_train, y5_test = train_test_split(X1, y5, test_size = 0.2, random_state = 0)
X2_train, X2_test, y5_train, y5_test = train_test_split(X2, y5, test_size = 0.2, random_state = 0)
X3_train, X3_test, y5_train, y5_test = train_test_split(X3, y5, test_size = 0.2, random_state = 0)
X4_train, X4_test, y5_train, y5_test = train_test_split(X4, y5, test_size = 0.2, random_state = 0)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size = 0.2, random_state = 0)
X6_train, X6_test, y5_train, y5_test = train_test_split(X6, y5, test_size = 0.2, random_state = 0)
# 1. Prepare GP models for all outputs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Define a helper function to create GP models
def create_gp_models(X_trains):
    return [GaussianProcessRegressor(
                kernel=RBF(length_scale=[1.0] * X.shape[1], length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5),
                n_restarts_optimizer=10, alpha=0
            ) for X in X_trains]

# List of your training feature sets
X_trains = [X1_train, X2_train, X3_train, X4_train, X5_train, X6_train]
X_tests = [X1_test, X2_test, X3_test, X4_test, X5_test, X6_test]

# Create models for each output
gp_models_y1 = create_gp_models(X_trains)  # mdot
gp_models_y2 = create_gp_models(X_trains)  # heating
gp_models_y3 = create_gp_models(X_trains)  # cooling
gp_models_y4 = create_gp_models(X_trains)  # motor power
gp_models_y5 = create_gp_models(X_trains)  # outlet temperature

# 2. Fit the models
for i in range(6):
    gp_models_y1[i].fit(X_trains[i], y1_train)
    gp_models_y2[i].fit(X_trains[i], y2_train)
    gp_models_y3[i].fit(X_trains[i], y3_train)
    gp_models_y4[i].fit(X_trains[i], y4_train)
    gp_models_y5[i].fit(X_trains[i], y5_train)

# 3. Predict
y1_preds = [gp.predict(X_tests[i]) for i, gp in enumerate(gp_models_y1)]
y2_preds = [gp.predict(X_tests[i]) for i, gp in enumerate(gp_models_y2)]
y3_preds = [gp.predict(X_tests[i]) for i, gp in enumerate(gp_models_y3)]
y4_preds = [gp.predict(X_tests[i]) for i, gp in enumerate(gp_models_y4)]
y5_preds = [gp.predict(X_tests[i]) for i, gp in enumerate(gp_models_y5)]

# 4. Calculate Metrics
from sklearn.metrics import mean_absolute_percentage_error, r2_score

def calc_metrics(y_true, y_preds):
    mape = [mean_absolute_percentage_error(y_true, yp) * 100 for yp in y_preds]
    mape_acc = [100 - m for m in mape]  # MAPE accuracy: higher is better
    r2 = [r2_score(y_true, yp) * 100 for yp in y_preds]
    return mape_acc, r2


mape_y1, r2_y1 = calc_metrics(y1_test, y1_preds)
mape_y2, r2_y2 = calc_metrics(y2_test, y2_preds)
mape_y3, r2_y3 = calc_metrics(y3_test, y3_preds)
mape_y4, r2_y4 = calc_metrics(y4_test, y4_preds)
mape_y5, r2_y5 = calc_metrics(y5_test, y5_preds)

# 5. Plot Results
import matplotlib.pyplot as plt
import numpy as np

import os
import matplotlib.pyplot as plt
import numpy as np

save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_ML'

def plot_metrics(mape, r2, output_name, file_name):
    features = [
        "$p_{suc}$",
        "$p_{suc}, p_{dis}$",
        "$p_{suc}, p_{dis},$\n$\\omega$",
        "$p_{suc}, p_{dis},$\n$\\omega, T_{heater}$",
        "$p_{suc}, p_{dis},$\n$\\omega, T_{heater}$,\n$T_{w, in}$",
        "$p_{suc}, p_{dis},$\n$\\omega, T_{heater}$,\n$T_{w, in}, T_{suc}$"
    ]

    n_groups = len(features)
    index = np.arange(n_groups)
    bar_width = 0.35

    plt.figure(figsize=(12, 8))
    plt.bar(index, mape, bar_width, label='100 - MAPE (%)')
    plt.bar(index + bar_width, r2, bar_width, label='$R^2$ (%)')

    plt.ylabel('Values (%)', fontsize=20)
    plt.title(f'{output_name}', fontsize=20)
    plt.xticks(index + bar_width / 2, features, rotation=0, ha="left", fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    for i in range(len(index)):
        plt.text(i, mape[i] + 1, f'{mape[i]:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        plt.text(i + bar_width, r2[i] + 1, f'{r2[i]:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')


    plt.tight_layout()

    # Construct full path
    full_path = os.path.join(save_dir, f"{file_name}.eps")
    plt.savefig(full_path, format='eps')
    plt.close()

plot_metrics(mape_y1, r2_y1, "$\dot{m}_f$", "GP_mdot_reduced")
plot_metrics(mape_y2, r2_y2, "$\dot{Q}_{heater}$", "GP_Pheating_reduced")
plot_metrics(mape_y3, r2_y3, "$\dot{Q}_{cooler}$", "GP_Pcooling_reduced")
plot_metrics(mape_y4, r2_y4, "$P_{motor}$", "GP_Pmotor_reduced")
plot_metrics(mape_y5, r2_y5, "$T_{dis}$", "GP_Tout_reduced")
