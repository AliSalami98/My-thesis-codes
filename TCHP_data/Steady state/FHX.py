import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from scipy.optimize import curve_fit# Create an AbstractState object using the HEOS backend and CO2
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

T0 = 298.15
p0 = 101325
state = AbstractState("HEOS", "CO2")
state.update(CP.PT_INPUTS, p0, T0)
h0 = state.hmass()
s0 = state.smass()
CP.set_reference_state('CO2',T0, p0, h0, s0)

def Delta_T_lm(Tf_in, Tf_out, Tfw_in, Tfw_out):
    Delta_T1 = Tf_in - Tfw_out
    Delta_T2 = Tf_out - Tfw_in

    delta_T_lm = (Delta_T1 - Delta_T2)/np.log(Delta_T1/Delta_T2)  #(data['Tf_in [K]'][i] - data['Tfw_in [K]'][i])
    return delta_T_lm

Dw = 1000
Dmpg = 1100
cpw = 4168
cpg = 1000

data = {
    'Pfhx [W]': [],
    'Tf_in [K]': [],
    'Tf_out [K]': [],
    'Tfw_in [K]': [],
    'Tfw_out [K]': [],
    'mw_dot [kg/s]': [],
    'mg_dot [kg/s]': [],
    'omegab [rpm]': []
}

with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\cycle measurements\all 2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        T13 = float(row['T13'])+ 273.15
        T19 = float(row['T19'])+ 273.15
        T28 = max(float(row['T28'])+ 273.15, T19 + 0.2)
        mw_dot = float(row['mw_dot'])/60
        mCH4_dot = float(row['mCH4_dot [kg/s]'])
        data['mg_dot [kg/s]'].append(mCH4_dot * 18.125)
        # -------------------------------
        # Recovery heat exchanger
        # -------------------------------
        
        Prec = mw_dot*cpw*(T28 - T19)
        data['Pfhx [W]'].append(Prec)
        data['Tf_in [K]'].append(T13)

        data['Tfw_in [K]'].append(T19)
        data['Tfw_out [K]'].append(T28)

        data['mw_dot [kg/s]'].append(mw_dot)
        data['omegab [rpm]'].append(float(row['omegab']))

# # --- data ---
# omega = np.array(data['omegab [rpm]'])
# Tf_in_C = np.array(data['Tf_in [K]']) - 273.15   # fit in °C

# # --- linear fit: T_f,in = a * omegab + b ---
# lin = LinearRegression()
# lin.fit(omega.reshape(-1, 1), Tf_in_C)
# a = float(lin.coef_[0])
# b = float(lin.intercept_)
# print(f"T_f,in = {a:.6g} * omegab + {b:.6g}")

# # predictions & metrics
# y_pred = lin.predict(omega.reshape(-1, 1))
# r2 = r2_score(Tf_in_C, y_pred)
# mae = mean_absolute_error(Tf_in_C, y_pred)
# print(f"R^2 = {r2:.4f}, MAE = {mae:.2f}%")

# # --- smooth line for plotting ---
# x_fit = np.linspace(omega.min(), omega.max(), 300)
# y_fit = lin.predict(x_fit.reshape(-1, 1))

# # --- plot (style matched to plot_fit) ---
# plt.figure(figsize=(8, 5))
# plt.scatter(omega, Tf_in_C, label='Data', color='blue', s=60, edgecolors='k', zorder=3)
# plt.plot(x_fit, y_fit, label='Linear fit', color='orange', lw=2, zorder=2)
# plt.xlabel(r'$\omega_{\mathrm{bf}}$ [rpm]', fontsize=14)
# plt.ylabel(r'$T_{\mathrm{fhx,fume,in}}$ [°C]', fontsize=14)
# # plt.title(r'$T_{\mathrm{fhx,fume,in}}$ vs $\omega_{\mathrm{bf}}$ — '
# #           f'MAPE: {mape:.2f}%', fontsize=16, pad=12)
# plt.legend(fontsize=12, loc='best')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# # plt.savefig('Tf_in_vs_omegab.eps', format='eps', bbox_inches='tight')
# plt.show()

# Get min and max mg_dot for scaling
mg_min = min(data['mg_dot [kg/s]'])
mg_max = max(data['mg_dot [kg/s]'])

for j in range(len(data['mw_dot [kg/s]'])):
    mg_dot = data['mg_dot [kg/s]'][j]
    Tfw_in = data['Tfw_in [K]'][j]

    # Linear scaling: 0 → 4 K, 1 → 15 K
    scaled_deltaT = 4 + (mg_dot - mg_min) / (mg_max - mg_min) * (15 - 4)
    Tf_out = Tfw_in + scaled_deltaT

    data['Tf_out [K]'].append(Tf_out)

print(data['mw_dot [kg/s]'])

# Updated AUrec calculation
AUrec = []

for i in range(len(data['Tf_in [K]'])):

    Delta_T1 = data['Tf_in [K]'][i] - data['Tfw_out [K]'][i]
    Delta_T2 = data['Tf_out [K]'][i] - data['Tfw_in [K]'][i]

    delta_T_lm = (Delta_T1 - Delta_T2)/np.log(Delta_T1/Delta_T2)  #(data['Tf_in [K]'][i] - data['Tfw_in [K]'][i])

    # Heat transfer rate (Q)
    Q = data['Pfhx [W]'][i]

    # Calculate AUrec
    if delta_T_lm > 0:
        AU_value = Q / delta_T_lm
        AUrec.append(AU_value)
    else:
        AUrec.append(0)

# Prepare the input (independent variables) and output (dependent variable)
X = np.array([
    data['mw_dot [kg/s]'],   # Water mass flow rate
    data['omegab [rpm]'],   # Gas mass flow rate
    # [x - 273.15 for x in data['Tfw_in [K]']]       # Water inlet temperature
]).T  # Transpose to make it of shape (n_samples, n_features)

y = np.array(AUrec)  # AUrec as the dependent variable

# Fit the multivariate regression model
model = LinearRegression()
model.fit(X, y)

# Get regression coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

print("Regression Coefficients:")
print(f"coefficients: {coefficients}")
print(f"Intercept: {intercept}")

# Predict AUrec values using the regression model
AUrec_pred = model.predict(X)

# Calculate RMSE for model performance
Pfhx_pred = [AUrec_pred[i] * Delta_T_lm(data['Tf_in [K]'][i], data['Tf_out [K]'][i], data['Tfw_in [K]'][i], data['Tfw_out [K]'][i]) for i in range(len(AUrec_pred))] 

rmse = np.sqrt(mean_squared_error(Pfhx_pred, data["Pfhx [W]"]))
print(f"Root Mean Squared Error (RMSE): {rmse}")
mape_Qfhx = mean_absolute_percentage_error(Pfhx_pred, data["Pfhx [W]"]) * 100
r2_Qfhx = r2_score(data["Pfhx [W]"], Pfhx_pred)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

Pfhx_real = np.array(data['Pfhx [W]'])*1e-3
Pfhx_pred = np.array(Pfhx_pred) * 1e-3
# Plot 1: Gas Cooler Outlet Temperature
fig1 = plt.figure()
plt.plot(Pfhx_real, Pfhx_real, c='k', label='Ideal line')
Pfhx_sorted = np.sort(Pfhx_real)
plt.plot(Pfhx_sorted, 0.9 * Pfhx_sorted, linestyle='--', color='blue', label='10% error')
plt.plot(Pfhx_sorted, 1.1 * Pfhx_sorted, linestyle='--', color='blue')
plt.scatter(Pfhx_real, Pfhx_pred, s=50, edgecolor="red", facecolor="lightgrey", label="Simulation results", linewidths=2.0)
plt.xlabel(r'Measured $\dot{Q}_\text{fhx}$ [kW]', fontsize=16)
plt.ylabel(r'Predicted $\dot{Q}_\text{fhx}$ [kW]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAPE: {mape_Qfhx:.1f}%\nR²: {r2_Qfhx:.3f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Qfhx.eps", format='eps', bbox_inches='tight')

plt.show()