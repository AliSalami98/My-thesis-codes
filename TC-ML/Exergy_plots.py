import csv
import tensorflow as tf
import numpy as np
from utils import get_state, CP, T0, p0, h0, s0
import matplotlib.pyplot as plt
import os
import joblib
from data_filling import(data, a_pcharged, a_Tr, a_Pr)

GP_models_dir = 'GP_models'
models_GP = [joblib.load(os.path.join(GP_models_dir, f'GP_model_{i+1}.pkl')) for i in range(6)]

NN_models_dir = 'NN_models'
models_NN = [tf.keras.models.load_model(os.path.join(NN_models_dir, f'NN_model_{i+1}.h5')) for i in range(6)]

scalers_dir = 'scalers'
scaler_X = joblib.load(os.path.join(scalers_dir, 'scaler_X.pkl'))
scalers_y = [joblib.load(os.path.join(scalers_dir, f'scaler_y_{i+1}.pkl')) for i in range(6)]

n = 10

# Rounded arrays for each parameter
a_omega = np.round(np.linspace(np.min(data['omega [rpm]']), np.max(data['omega [rpm]']), n)).astype(int)
a_pcharged = np.round(np.linspace(35e5, np.max(a_pcharged), n)).astype(int)
a_Tr = np.round(np.linspace(2.1, np.max(a_Tr), n), 2)
a_Pr = np.round(np.linspace(np.min(a_Pr), np.max(a_Pr), n), 2)
a_Theater = np.round(np.linspace(np.min(773), np.max(data['Theater [K]']), n)).astype(int)


a_mdot = np.zeros(n)
a_Tout = np.zeros(n)
a_Pmech = np.zeros(n)
a_Pheat = np.zeros(n)
a_Pcool = np.zeros(n)
a_Pcp = np.zeros(n)
eff = np.zeros(n)
Ex_out = np.zeros(n)
Ex_heater = np.zeros(n)
Ex_cooler = np.zeros(n)
X_total = np.zeros(n)
X_flow = np.zeros(n)
X_transfer = np.zeros(n)
X_heater = np.zeros(n)
X_cooler = np.zeros(n)

eff_Ex = np.zeros(n)

mw_dot = 20/60
cpw = 4186
Dw = 997

# key = 'omega'
key = 'pcharged'
# key = 'Tr'
# key = 'Pr'

Pr = 1.35
p1 = 40e5
p2 = p1 * Pr
pcharged = np.sqrt(p1 * p2)
Theater = 973.15
Twi = 293.15
Tr = Theater/Twi
omega = 150
for k in range(n):
    if key == 'omega':
        omega = a_omega[k]
        pcharged = np.sqrt(p1 **2 * Pr)
        x_vals = a_omega
        xlabel = r'Rotational speed $\omega_\text{m}$ [rpm]'
    elif key == 'pcharged':
        pcharged = a_pcharged[k]
        p1 = pcharged/np.sqrt(Pr)
        p2 = p1 * Pr
        x_vals = a_pcharged * 1e-5
        xlabel = r'Charged pressure $p_\text{charged}$ [bar]'
    elif key == 'Tr':
        # Tr = a_Tr[k]
        Theater = a_Theater[k] #Twi * Tr
        x_vals = [int(x - 273.15) for x in a_Theater]
        xlabel = r'Heater temperature $T_\text{h}$ [°C]'
    elif key == 'Pr':
        Pr = a_Pr[k]
        p1 = pcharged/np.sqrt(Pr)
        p2 = p1 * Pr
        x_vals = a_Pr
        xlabel = r'Pressure ratio $r_\text{p}$ [-]'

    state = get_state(CP.PQ_INPUTS, p1, 1)
    T1 = state.T() + 3
    state = get_state(CP.PT_INPUTS, p1, T1)
    h1 = state.hmass()
    s1 = state.smass()
    # Construct input sample
    test1 = np.array([[p1, p2, omega, Theater, Twi, T1]])  # shape (1, 6)
    # test1 = np.array([[Pr, Tr, pcharged, omega, T1]])
    # Scale input
    test1_scaled = scaler_X.transform(test1)

    # # NN predictions for all 6 outputs
    # predictions_NN = [model.predict(test1_scaled) for model in models_NN]
    # y_pred_real_NN = [scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
    #                   for pred, scaler in zip(predictions_NN, scalers_y)]

    # # # Assign predictions to each target array
    # a_mdot[k]   = y_pred_real_NN[0]
    # a_Pheat[k]  = y_pred_real_NN[1]
    # a_Pcool[k]  = y_pred_real_NN[2]
    # a_Pmech[k] = y_pred_real_NN[3]
    # a_Tout[k]   = y_pred_real_NN[4]
    # # a_Ploss[k]  = y_pred_real_NN[5]

    # GP predictions for all 6 outputs
    predictions_GP = [model.predict(test1_scaled) for model in models_GP]
    y_pred_real_GP = [scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
                      for pred, scaler in zip(predictions_GP, scalers_y)]

    # Assign predictions to each target array
    a_mdot[k]   = y_pred_real_GP[0]
    a_Pheat[k]  = y_pred_real_GP[1]
    # a_Pcool[k]  = y_pred_real_GP[2]
    a_Pmech[k] = y_pred_real_GP[3]
    a_Tout[k]   = y_pred_real_GP[4]
    # a_Ploss[k]  = y_pred_real_GP[5]

    # --- Thermodynamic state at outlet ---
    T2 = a_Tout[k]
    mdot = a_mdot[k]
    state = get_state(CP.PT_INPUTS, p2, T2)
    h2, s2 = state.hmass(), state.smass()

    # --- Compression power and exergy output ---
    a_Pcp[k] = mdot * (h2 - h1)
    a_Pcool[k]  = a_Pheat[k] + a_Pmech[k] - a_Pcp[k]
    psi1 = (h1 - h0) - T0 * (s1 - s0)
    psi2 = (h2 - h0) - T0 * (s2 - s0)
    Ex_out[k] = mdot * (psi2 - psi1)

    # --- Water side (cooling) ---
    hwi = CP.PropsSI('H', 'P', p0, 'T', Twi, 'Water')
    swi = CP.PropsSI('S', 'P', p0, 'T', Twi, 'Water')
    hwo = hwi + a_Pcool[k] / mw_dot
    swo = CP.PropsSI('S', 'P', p0, 'H', hwo, 'Water')

    psi_wi = (hwi - h0) - T0 * (swi - s0)
    psi_wo = (hwo - h0) - T0 * (swo - s0)
    Ex_cooler[k] = mw_dot * (psi_wo - psi_wi)

    # --- Exergy terms ---
    Ex_heater[k] = (1 - T0 / Theater) * a_Pheat[k]
    X_flow[k]   = mdot * T0 * (s2 - s1)
    X_transfer[k]  = -T0 / Theater * a_Pheat[k] + T0 / Twi * a_Pcool[k]
    X_heater[k]    = T0 / Theater * a_Pheat[k]
    X_cooler[k]    = T0 / Twi * a_Pcool[k]
    X_total[k]     = Ex_heater[k] + a_Pmech[k] - Ex_out[k] - Ex_cooler[k]

    # --- Efficiencies ---
    eff_Ex[k] = 100 * (1 - X_total[k] / (Ex_heater[k] + a_Pmech[k]))
    eff[k]    = 100 * a_Pcp[k] / (a_Pheat[k] + a_Pmech[k])

    # --- Unit conversion to kW ---
    for arr in [a_Pcool, a_Pmech, a_Pcp, Ex_out, Ex_cooler, X_flow, X_transfer, X_heater, X_cooler]:
        arr[k] *= 1e-3

# fig, ax1 = plt.subplots(figsize=(10, 6))

# Combined stacked bar chart for Ex_heater, Ex_cooler, and Ex_out on the same bar
width = 0.5  # Adjusted bar width for clarity
x = np.arange(len(x_vals))

fig2, ax3 = plt.subplots(figsize=(10, 6))


ax3.bar(x, a_Pmech, width, label='Motor', color='orange', edgecolor='black', alpha=0.8)
ax3.bar(x, Ex_cooler, width, bottom=a_Pmech, label='Cooler', color='red', edgecolor='black', alpha=0.8)
ax3.bar(x, Ex_out, width, bottom= a_Pmech + Ex_cooler, label='Compression', color='green', edgecolor='black', alpha=0.8)

# Set labels and ticks for the left y-axis
ax3.set_ylabel('Exergy [kW]', fontsize=16)
ax3.set_xlabel(xlabel, fontsize=16)
ax3.tick_params(axis='y', labelcolor='black', labelsize=16)
ax3.set_xticks(x)
ax3.set_xticklabels([('{:g}'.format(x)) for x in x_vals], fontsize=16)
ax3.legend(loc='upper left', fontsize=12)

# Create a Twiin y-axis for the efficiency
ax4 = ax3.twinx()
ax4.plot(x, eff_Ex, 'k--o', label='Exergy Efficiency', linewidth=2, markersize=6)
ax4.set_ylabel('Exergy Efficiency [%]', fontsize=16)
ax4.tick_params(axis='y', labelsize=16)

# Add a legend for the efficiency
ax4.legend(loc='upper right', fontsize=12)

# Add grid
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.tight_layout()

save_path = "C:/Users/ali.salame/Desktop/plots/Thesis figs/TC_ML/Exergy_plots/Pr.eps"
plt.savefig(save_path, format='eps', dpi=300)
plt.show()