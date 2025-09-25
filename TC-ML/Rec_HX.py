import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from scipy.optimize import curve_fit# Create an AbstractState object using the HEOS backend and CO2
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
    "mdot [g/s]": [],
    "Pcomb [W]": [],
    "Pheating [W]": [],
    "Pcooling [W]": [],
    "Pmotor [W]": [],
    "Tout [K]": [],
    "Pout [W]": [],
    "eff [%]": [],
    "Tf_in [K]": [],
    "Tf_out [K]": [],
    "Tw_in [K]": [],
    "Tw_out [K]": [],
    "Pfume [W]": [],
    "mw_dot [kg/s]": [],
    "mCH4_dot [kg/s]": [],
    "mg_dot [kg/s]": []
}

AUrec = []
counter = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\TC experiments\all.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        counter += 1
        data['Tin [K]'].append(float(row['Tin [°C]']) + 273.15)
        data['pin [pa]'].append(float(row['pin [bar]'])*10**5)
        data['pout [pa]'].append(float(row['pout [bar]'])*10**5)
        data['Th_wall [K]'].append(float(row['Th_wall[°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        data['mdot [g/s]'].append(float(row['mdot [g/s]']))
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Tout [K]'].append(float(row['Tout [°C]']) + 273.15)
        data['Pout [W]'].append(float(row['Pout [W]']))
        data['eff [%]'].append(100*float(row['eff [%]']))
        data['Tf_in [K]'].append(float(row['Tf_in [°C]']) + 273.15)
        data['Tf_out [K]'].append(float(row['Tf_out [°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Twf_in [°C]']) + 273.15)
        data['Tw_out [K]'].append(float(row['Twf_out [°C]']) + 273.15)
        data['Pfume [W]'].append(float(row['Pfume [W]']))
        data['mw_dot [kg/s]'].append(float(row['mwf_dot [l/min]'])/60)
        data['mCH4_dot [kg/s]'].append(float(row['mCH4_dot [m^3/h]']) * 0.675/3600)
        data['mg_dot [kg/s]'].append(data['mCH4_dot [kg/s]'][-1] * 18.125)

        AUrec.append(data['Pfume [W]'][-1]/(data['Tf_in [K]'][-1] - data['Tw_in [K]'][-1]))
        # if counter > 52:
        #     break


Pr = [s/r for s,r in zip(data['pout [pa]'][:], data['pin [pa]'][:])]
Tr = [s/r for s,r in zip(data['Th_wall [K]'][:], data['Tw_in [K]'][:])]

# Define constants and properties
cp_fume = 1.005  # Specific heat capacity of fume (air) in kJ/(kg*K) - adjust as needed
cp_water = 4.18  # Specific heat capacity of water in kJ/(kg*K) - adjust as needed

# Updated AUrec calculation
AUrec = []
for i in range(len(data['Tf_in [K]'])):
    # Calculate log-mean temperature difference
    delta_T1 = data['Tf_in [K]'][i] - data['Tw_out [K]'][i]
    delta_T2 = data['Tf_out [K]'][i] - data['Tw_in [K]'][i]
    if delta_T1 > 0 and delta_T2 > 0:  # Avoid log of negative or zero
        delta_T_lm = (data['Tf_in [K]'][i] - data['Tw_in [K]'][i]) #(delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
    else:
        delta_T_lm = 0  # Handle cases with invalid temperature differences

    # Heat transfer rate (Q)
    Q = data['Pfume [W]'][i]

    # Calculate AUrec
    if delta_T_lm > 0:
        AU_value = Q / delta_T_lm
        AUrec.append(AU_value)
    else:
        AUrec.append(0)

# Fit and plot AUrec as before
x = np.array(data['mg_dot [kg/s]'])  # Example: Use water flow rate as the independent variable
AUrec = np.array(AUrec)

# Define linear fit model
def linear_model(x, a, b):
    return a * x + b

# Fit the linear model
params, _ = curve_fit(linear_model, x, AUrec)

# Generate fitted AUrec
AUrec_fitted = linear_model(x, *params)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, AUrec, label="Original AUrec Data", color="blue")
plt.plot(x, AUrec_fitted, label="Linear Fit", color="red")
plt.xlabel("Mass Flow Rate of Water (kg/s)")
plt.ylabel("AUrec (W/K)")
plt.title("AUrec vs. Water Mass Flow Rate")
plt.legend()
plt.grid()
plt.show()

# Print linear fit parameters
print("Linear Fit Parameters: a =", params[0], ", b =", params[1])
