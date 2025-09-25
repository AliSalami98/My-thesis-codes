# -------------------------
# IMPORTS
# -------------------------
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import csv
import matplotlib.pyplot as plt
import numpy as np
from utils import get_state, CP, h0, T0, s0, p0
data = {
    "Tsuc [K]": [], "psuc [pa]": [], "pdis [pa]": [],
    "Theater [K]": [], "Tw_in [K]": [], "omega [rpm]": [], "omegab [rpm]": [],
    "mdot [kg/s]": [], "Pcomb [W]": [], "Pheating [W]": [],
    "Pcooling [W]": [], "Pmotor [W]": [], "Tdis [K]": [],
    "Pcomp [W]": [], "Ex_eff [%]": [], "Ploss [W]": [], "Pmech [W]": []
}


a_Pr = []
a_Theater = []
a_pcharged = []
a_omega = []
eff_Ex = []
i = 0

with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\Tc experiments\all_ultimate.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        data['Tsuc [K]'].append(float(row['Tsuc [°C]']) + 273.15)
        data['psuc [pa]'].append(float(row['psuc [bar]']) * 1e5)
        data['pdis [pa]'].append(float(row['pdis [bar]']) * 1e5)
        data['Theater [K]'].append(float(row['Theater [°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        # data['omegab [rpm]'].append(float(row['omegab [rpm]']))
        data['mdot [kg/s]'].append(float(row['mdot [g/s]']) * 1e-3)
        data['Tdis [K]'].append(float(row['Tdis [°C]']) + 273.15)
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))

        # data['Pheating [W]'].append(float(row['Pheating [W]']))
        # data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        # data['Pmech [W]'].append(float(row['Pmech [W]']))
        # data['Pcomp [W]'].append(float(row['Pcomp [W]']))
        # # data['eff [%]'].append(100 * float(row['eff [%]']))
        # data['Ploss [W]'].append(float(row['Ploss [W]']))


        # mw_dot = 10/60
        # Theater = data['Theater [K]'][-1]
        # omega = data['omega [rpm]'][-1]
        # Twi = data['Tw_in [K]'][-1]
        # T2 = data['Tdis [K]'][-1]
        # p2 = data['pdis [pa]'][-1]
        # mdot = data['mdot [kg/s]'][-1]
        # state = get_state(CP.PT_INPUTS, p2, T2)
        # h2, s2 = state.hmass(), state.smass()

        # T1 = data['Tsuc [K]'][-1]
        # p1 = data['psuc [pa]'][-1]
        # state = get_state(CP.PT_INPUTS, p1, T1)
        # h1, s1 = state.hmass(), state.smass()
        # # --- Compression power and exergy output ---
        # Pcomp = data['Pcomp [W]'][-1]
        # Pcool  = data['Pcooling [W]'][-1]
        # Pheat  = data['Pheating [W]'][-1]
        # Pmech  = data['Pmech [W]'][-1]
        # psi1 = (h1 - h0) - T0 * (s1 - s0)
        # psi2 = (h2 - h0) - T0 * (s2 - s0)
        # Ex_comp = mdot * (psi2 - psi1)

        # # --- Water side (cooling) ---
        # hwi = CP.PropsSI('H', 'P', p0, 'T', Twi, 'Water')
        # swi = CP.PropsSI('S', 'P', p0, 'T', Twi, 'Water')
        # hwo = hwi + Pcool / mw_dot
        # swo = CP.PropsSI('S', 'P', p0, 'H', hwo, 'Water')

        # psi_wi = (hwi - h0) - T0 * (swi - s0)
        # psi_wo = (hwo - h0) - T0 * (swo - s0)
        # Ex_cooler = mw_dot * (psi_wo - psi_wi)

        # # --- Exergy terms ---
        # Ex_heater = (1 - T0 / Theater) * Pheat
        # X_flow   = mdot * T0 * (s2 - s1)
        # X_transfer  = -T0 / Theater * Pheat + T0 / Twi * Pcool
        # X_heater    = T0 / Theater * Pheat
        # X_cooler    = T0 / Twi * Pcool
        # X_total     = Ex_heater + Pmech - Ex_comp - Ex_cooler

        # # --- Efficiencies ---
        # data['Ex_eff [%]'].append(100 * (1 - X_total / (Ex_heater + Pmech)))
        # a_Pr.append(p2/p1)
        # a_Theater.append(Theater)
        # a_omega.append(omega)
        # a_pcharged.append(np.sqrt(p1 * p2))

# -------------------------
# 4. DEFINE INPUTS X AND OUTPUTS Y
# -------------------------
a_Pr = [s/r for s, r in zip(data['pdis [pa]'], data['psuc [pa]'])]
a_Tr = [s/r for s, r in zip(data['Theater [K]'], data['Tw_in [K]'])]
a_pcharged = [np.sqrt(s * r) for s, r in zip(data['pdis [pa]'], data['psuc [pa]'])]

# Build X dynamically
X = np.column_stack([data['psuc [pa]'], data['pdis [pa]'], data['omega [rpm]'], data['Theater [K]'], data['Tw_in [K]'], data['Tsuc [K]']])
# X = np.column_stack([a_Pr, a_pcharged, data['Theater [K]'], data['Tw_in [K]'], data['omega [rpm]'], data['Tsuc [K]']])  # 4 inputs now
# y_names = ['mdot [kg/s]', 'Pheating [W]', 'Pcooling [W]', 'Pmech [W]', 'Tdis [K]', 'Ploss [W]', 'Ex_eff [%]']
y_names = ['mdot [kg/s]', 'Pcooling [W]', 'Tdis [K]']

y = [np.array(data[name]) for name in y_names]

# ------------------------------------
# 5. Scaling
# ------------------------------------
scaler_X = MinMaxScaler((0.1, 0.9))
X_scaled = scaler_X.fit_transform(X)

scaler_y = [MinMaxScaler((0.1, 0.9)) for _ in y]
y_scaled = [scaler.fit_transform(target.reshape(-1, 1)).flatten() for scaler, target in zip(scaler_y, y)]

# Directory to save scalers
scaler_dir = 'scalers'
os.makedirs(scaler_dir, exist_ok=True)

# Save input scaler
joblib.dump(scaler_X, os.path.join(scaler_dir, 'scaler_X.pkl'))

# Save output scalers
for i, s in enumerate(scaler_y):
    joblib.dump(s, os.path.join(scaler_dir, f'scaler_y_{i+1}.pkl'))

# List saved files
scaler_files = os.listdir(scaler_dir)