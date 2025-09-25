import csv
import tensorflow as tf
import numpy as np
from utils import get_state, CP, T0, p0, h0, s0
import matplotlib.pyplot as plt
import os
import joblib
from data_filling import(data, a_pcharged, a_Tr, a_Pr)

GP_models_dir = 'GP_models'
models_GP = [joblib.load(os.path.join(GP_models_dir, f'GP_model_{i+1}.pkl')) for i in range(7)]

NN_models_dir = 'NN_models'
models_NN = [tf.keras.models.load_model(os.path.join(NN_models_dir, f'NN_model_{i+1}.h5')) for i in range(7)]

scalers_dir = 'scalers'
scaler_X = joblib.load(os.path.join(scalers_dir, 'scaler_X.pkl'))
scalers_y = [joblib.load(os.path.join(scalers_dir, f'scaler_y_{i+1}.pkl')) for i in range(7)]

nj = 5
nk = 10

# Rounded arrays for each parameter
# a_omega = np.array([60, 120, 180, 240])
# a_pcharged = np.array([30, 40, 50, 60]) #np.round(np.linspace(np.min(a_pcharged), np.max(a_pcharged), n) * 1e-5).astype(int)
# a_Tr = np.round(np.linspace(np.min(a_Tr), np.max(a_Tr), n), 2)
# a_Pr = np.round(np.linspace(np.min(a_Pr), np.max(a_Pr), n), 2)

a_omega = np.round(np.linspace(np.min(data['omega [rpm]']), np.max(data['omega [rpm]']), nj)).astype(int)
a_Theater = np.round(np.linspace(773.15, np.max(data['Theater [K]']), nj)).astype(int)
a_Tr = np.round(np.linspace(np.min(a_Tr), np.max(a_Tr), nj), 2)
a_pcharged = np.round(np.linspace(np.min(a_pcharged), np.max(a_pcharged), nk)).astype(int)
a_Pr = np.round(np.linspace(np.min(a_Pr), np.max(a_Pr), nj), 2)


a_mdot = np.zeros(nk)
a_Tout = np.zeros(nk)
a_Pmech = np.zeros(nk)
a_Pheat = np.zeros(nk)
a_Pcool = np.zeros(nk)
a_Pcp = np.zeros(nk)
eff = np.zeros(nk)
Ex_out = np.zeros(nk)
Ex_heater = np.zeros(nk)
Ex_cooler = np.zeros(nk)
X_total = np.zeros(nk)
X_flow = np.zeros(nk)
X_transfer = np.zeros(nk)
X_heater = np.zeros(nk)
X_cooler = np.zeros(nk)

eff_Ex = np.zeros((nj, nk))
Ex_out_matrix = np.zeros((nj, nk))
mdot_matrix = np.zeros((nj, nk))

mw_dot = 20/60
cpw = 4186
Dw = 997

# key = 'omega'
# key = 'Tr'
key = 'pcharged'
# key = 'Pr'

Pr = 1.35
p1 = 40e5
p2 = p1 * Pr
pcharged = np.sqrt(p1 * p2)
Theater = 973.15
Twi = 293.15
Tr = Theater/Twi
omega = 150
for j in range(nj):
    # pcharged = a_pcharged[j]
    Theater = a_Theater[j]
    # omega = a_omega[j]
    # Pr = a_Pr[j]
    p1 = pcharged/np.sqrt(Pr)
    p2 = p1 * Pr
    for k in range(nk):
        if key == 'omega':
            omega = a_omega[k]
            pcharged = np.sqrt(p1 **2 * Pr)
            x_vals = a_omega
            xlabel = 'Rotation Speed [rpm]'
        elif key == 'pcharged':
            pcharged = a_pcharged[k]
            p1 = pcharged/np.sqrt(Pr)
            p2 = p1 * Pr
            x_vals = a_pcharged * 1e-5
            xlabel = 'Charged Pressure [bar]'
        elif key == 'Tr':
            # Tr = a_Tr[k]
            Theater = a_Theater[k] #Twi * Tr
            x_vals = a_Theater - 273.15
            xlabel = 'Heater temperature [°C]'
        elif key == 'Pr':
            Pr = a_Pr[k]
            p1 = pcharged/np.sqrt(Pr)
            p2 = p1 * Pr
            x_vals = a_Pr
            xlabel = 'Pressure Ratio $P_r$'

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
        # y_pred_NN = [scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
        #                   for pred, scaler in zip(predictions_NN, scalers_y)]

        # # Assign predictions to each target array
        # a_mdot[k]   = y_pred_NN[0]
        # a_Pheat[k]  = y_pred_NN[1]
        # a_Pcool[k]  = y_pred_NN[2]
        # a_Pmech[k] = y_pred_NN[3]
        # a_Tout[k]   = y_pred_NN[4]
        # # a_Ploss[k]  = y_pred_NN[5]

        # GP predictions for all 6 outputs
        predictions_GP = [model.predict(test1_scaled) for model in models_GP]
        y_pred_GP = [scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
                        for pred, scaler in zip(predictions_GP, scalers_y)]

        # Assign predictions to each target array
        a_mdot[k]   = y_pred_GP[0]
        a_Pheat[k]  = y_pred_GP[1]
        a_Pcool[k]  = y_pred_GP[2]
        a_Pmech[k] = y_pred_GP[3]
        a_Tout[k]   = y_pred_GP[4]
        # a_Ploss[k]  = y_pred_GP[5]

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
        Ex_heater[k]    = (1 - T0 / Theater) * a_Pheat[k]
        X_flow[k]      = mdot * T0 * (s2 - s1)
        X_transfer[k]  = -T0 / Theater * a_Pheat[k] + T0 / Twi * a_Pcool[k]
        X_heater[k]    = T0 / Theater * a_Pheat[k]
        X_cooler[k]    = T0 / Twi * a_Pcool[k]
        X_total[k]     = Ex_heater[k] + a_Pmech[k] - Ex_out[k] - Ex_cooler[k]

        # --- Efficiencies ---
        eff_Ex[j, k] = max(100 * (1 - X_total[k] / (Ex_heater[k] + a_Pmech[k])), 0)
        Ex_out_matrix[j, k] = Ex_cooler[k] + Ex_out[k]
        mdot_matrix[j,k] = mdot
        eff[k]    = 100 * a_Pcp[k] / (a_Pheat[k] + a_Pmech[k])

        # --- Unit conversion to kW ---
        for arr in [a_Pcool, a_Pmech, a_Pcp, Ex_out, Ex_cooler, X_flow, X_transfer, X_heater, X_cooler]:
            arr[k] *= 1e-3

a_pcharged = a_pcharged * 1e-5
a_Theater = a_Theater - 273.15
# Example: Replace with your desired output path
save_path = "C:/Users/ali.salame/Desktop/plots/Thesis figs/TC_ML/zabri2/pcharged_Theater.eps"

plt.figure()
for i, p in enumerate(a_Theater):
    plt.plot(x_vals, eff_Ex[i], marker='o', label=f'{int(p)} [°C]')

# plt.xlabel(r'Rotational speed $\omega_{\text{m}}$ [rpm]', fontsize=14)
plt.xlabel(r'Charged pressure $p_{\text{charged}}$ [bar]', fontsize=14)
# plt.xlabel(r'Heater temperature $T_{\text{h}}$ [°C]', fontsize=14)
# plt.xlabel(r'Pressure ratio $r_{\text{p}}$ [-]', fontsize=14)

plt.ylabel('Exergy efficiency [%]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(save_path, format='eps', dpi=300)
plt.show()

