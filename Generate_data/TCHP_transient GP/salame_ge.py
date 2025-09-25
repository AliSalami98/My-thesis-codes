from scipy.integrate import solve_ivp
from model import cycle
from post_processing import post_process
import numpy as np
import config
import time
import csv
import matplotlib.pyplot as plt
from algebraic_model import compute_cycle_inputs
from config import CP
from utils import (
	get_state,
)
from funcs import(
    predict_next_Theater1,
    predict_next_Theater2
)
import joblib
import os 

GP_models_dir = 'ML_Models/GP_models'
models_GP = [joblib.load(os.path.join(GP_models_dir, f'GP_model_{i+1}.pkl')) for i in range(3)]

scalers_dir = 'ML_Models/scalers'
scaler_X = joblib.load(os.path.join(scalers_dir, 'scaler_X.pkl'))
scalers_y = [joblib.load(os.path.join(scalers_dir, f'scaler_y_{i+1}.pkl')) for i in range(3)]

from config import CP, total_time_steps
from utils import (
	get_state,
)

from ML_Models.comps import (
    model1_1_LR,
    model1_2_LR,
    model1_3_LR,
    model2_1_LR,
    model2_2_LR,
    model2_3_LR,
    model3_1_LR,
    model3_2_LR,
    model3_3_LR,
)
from data_filling_ge import (
    d_omegab,
    d_omega2,
    d_mw_dot,
    d_Tw_in,
    d_Hpev,
)

omega1 = 200
omega2 = 150
omega3 = 100
mmpg_dot = 16.5/60
Tmpg_in = 3 + 273.15
Lpev = 50
Hpev= 50
hpev = Hpev
lpev = Lpev 
omegab = 4000
Theater1 = 800.15
Theater2 = 773.15

pe_in = 30e5 #pe[0]
state = get_state(CP.PQ_INPUTS, pe_in, 0)
he_in = state.hmass()
state = get_state(CP.HmassP_INPUTS, he_in, pe_in)
De_in = state.rhomass()
Te_in = state.T()
cp_mpg = 3600

config.pe =  pe_in
for i in range(config.Ne):
    config.Tmpg[i] = Tmpg_in
    config.Te_wall[i] = (Tmpg_in)
    config.he[i] = he_in
    config.De[i] = De_in

pc_in = 72e5
# a3 = np.array([[pc_in/pe_in], [Theater3/Tw_in[0]], [omega3[0]]])
# a3_T = a3.T
Tc_in = 320
state = get_state(CP.PT_INPUTS, pc_in, Tc_in)
hc_in = state.hmass()

config.pc =  pc_in
for i in range(config.Nc):
    config.Tc_w[i] = d_Tw_in[0]
    config.Tc_wall[i] = (d_Tw_in[0])
    config.hc[i] = hc_in

pihx1_in = pc_in
hihx1_in = hc_in
state = get_state(CP.HmassP_INPUTS, hihx1_in, pihx1_in)
Dihx1_in = state.rhomass()
Tihx1_in = state.T()

pihx2_in = pe_in
hihx2_in = he_in
state = get_state(CP.HmassP_INPUTS, hihx2_in, pihx2_in)
Tihx2_in = state.T()
Dihx2_in = state.rhomass()

config.pihx1 =  pihx1_in
config.hihx1 = hihx1_in
config.Dihx1 = Dihx1_in
config.Tihx_wall = (Tihx2_in + Tihx1_in)/2
config.pihx2 =  pihx2_in
config.hihx2 = hihx2_in
config.Dihx2 = Dihx2_in

Tr1 = Theater1/d_Tw_in[0]
Pr1 = 1.3 #/config.pe
a1 = np.array([[Pr1], [Tr1], [omega1]])
# a1 = np.array([[1.3 * pe_in], [pe_in], [omega1], [Theater1], [d_Tw_in[0]]])
a1_T = a1.T

pbuff1_in = 1.3 * pe_in
# hbuff1_in = he_in
Tbuff1_in = 320 #model2_PR.predict(poly2_reg.transform(a1_T))[0]
state = get_state(CP.PT_INPUTS, pbuff1_in, Tbuff1_in)
hbuff1_in = state.hmass()
Dbuff1_in = state.rhomass()
Tbuff1_w_in = d_Tw_in[0]

config.Tbuff1_w = d_Tw_in[0]
config.Tbuff1_wall = (d_Tw_in[0])
config.pbuff1 =  pbuff1_in
config.hbuff1 = hbuff1_in
config.Dbuff1 = Dbuff1_in

pbuff2_in = pc_in/1.2
# hbuff2_in = he_in
Tbuff2_in = 320 # model2_PR.predict(poly2_reg.transform(a1_T))[0]
state = get_state(CP.PT_INPUTS, pbuff2_in, Tbuff2_in)
hbuff2_in = state.hmass()
Dbuff2_in = state.rhomass()
Tbuff2_w_in = d_Tw_in[0]

config.Tbuff2_w = d_Tw_in[0]
config.Tbuff2_wall = (d_Tw_in[0])
config.pbuff2 =  pbuff2_in
config.hbuff2 = hbuff2_in
config.Dbuff2 = Dbuff2_in


pft_in = pbuff1_in
hft_in = (he_in + hc_in)/2 #hc_in
state = get_state(CP.HmassP_INPUTS, hft_in, pft_in)
Tft_in = state.T()
Dft_in = state.rhomass()

config.pft = pft_in
state = get_state(CP.PQ_INPUTS, pft_in, 0)
config.hft_l = state.hmass()
config.Dft_l = state.rhomass()
state = get_state(CP.PQ_INPUTS, pft_in, 1)
config.hft_v = state.hmass()
config.Dft_v = state.rhomass()
config.hft = hft_in
config.Dft = Dft_in


tau_HPV = 20
tau_LPV = 10

X0 = ([config.pe] + config.he + config.Te_wall + config.Tmpg
    + [config.pc] + config.hc+ config.Tc_wall + config.Tc_w
    + [config.pihx1, config.hihx1, config.Tihx_wall, config.pihx2, config.hihx2,
    config.pbuff1, config.hbuff1, config.Tbuff1_wall, config.pbuff2, config.hbuff2, config.Tbuff2_wall,
    config.pft, config.hft])


# import csv

# # Define file path
# X0_csv_path = "X0.csv"

# # Load X0 from CSV
# def load_X0_from_csv(csv_path):
#     with open(csv_path, 'r') as f:
#         reader = csv.reader(f, delimiter=';')  # Use correct delimiter
#         next(reader)  # Skip header
#         X0_values = next(reader)  # Read the first row of data

#     X0 = [float(value) for value in X0_values]
#     return X0
# # Call function to get X0
# X0 = load_X0_from_csv(X0_csv_path)
# # Assign values to config
# index = 0  # Keep track of position in X0

# # Assign scalars
# config.pe = X0[index]
# index += 1

# # Assign lists dynamically
# config.he = X0[index : index + len(config.he)]
# index += len(config.he)

# config.Te_wall = X0[index : index + len(config.Te_wall)]
# index += len(config.Te_wall)

# config.pc = X0[index]
# index += 1

# config.hc = X0[index : index + len(config.hc)]
# index += len(config.hc)

# config.Tc_wall = X0[index : index + len(config.Tc_wall)]
# index += len(config.Tc_wall)

# config.pihx1 = X0[index]
# index += 1

# config.hihx1 = X0[index]
# index += 1

# config.Tihx_wall = X0[index]
# index += 1

# config.pihx2 = X0[index]
# index += 1

# config.hihx2 = X0[index]
# index += 1

# config.pbuff1 = X0[index]
# index += 1

# config.hbuff1 = X0[index]
# index += 1

# config.Tbuff1_wall = X0[index]
# index += 1

# config.pft = X0[index]
# index += 1

# config.hft = X0[index]
# index += 1

import joblib

A1_coeffs = joblib.load("coeffs/Theater/A1_coeffs.pkl")
B1_coeffs = joblib.load("coeffs/Theater/B1_coeffs.pkl")
D1_coeffs = joblib.load("coeffs/Theater/D1_coeffs.pkl")
poly_U1 = joblib.load("coeffs/Theater/poly_U1.pkl")

A2_coeffs = joblib.load("coeffs/Theater/A2_coeffs.pkl")
B2_coeffs = joblib.load("coeffs/Theater/B2_coeffs.pkl")
D2_coeffs = joblib.load("coeffs/Theater/D2_coeffs.pkl")
poly_U2 = joblib.load("coeffs/Theater/poly_U2.pkl")

Theater1_current = np.column_stack([Theater1])
U1_current = np.column_stack([d_omegab[0]])
D1_current = np.column_stack([omega1])

Theater2_current = np.column_stack([Theater2])
U2_current = np.column_stack([d_omegab[0]])
D2_current = np.column_stack([omega2])

a_Theater1 = []
a_Theater2 = []
a_t = []

total_time_steps = len(d_omegab)
dt = 1
# Loop over time steps
for current_time_step in range(total_time_steps):
    current_time = current_time_step * dt

    omegab = d_omegab[current_time_step]

    # Predict next Theater1 and Theater2
    Theater1_current = predict_next_Theater2(Theater1_current, U1_current, D1_current, A1_coeffs, B1_coeffs, D1_coeffs, poly_U1)
    Theater2_current = predict_next_Theater2(Theater2_current, U2_current, D2_current, A2_coeffs, B2_coeffs, D2_coeffs, poly_U2)

    Theater1 = Theater1_current[0][0]
    Theater2 = Theater2_current[0][0]

    U1_current = np.column_stack([omegab])
    D1_current = np.column_stack([omega1])

    U2_current = np.column_stack([omegab])
    D2_current = np.column_stack([omega2])

    a_Theater1.append(Theater1)
    a_Theater2.append(Theater2)
    a_t.append(current_time)

plt.plot(a_t, a_Theater1)
plt.plot(a_t, a_Theater2)
plt.show()

Tc_out_ss = []
hc_out_ss = []
pe_ss = []
Te_out_ss = []
he_out_ss = []

me_dot_ss = []
mihx2_dot_ss = []
mbuff1_dot_ss = []
mbuff2_dot_ss = []
mc_dot_ss = []
mihx1_dot_ss = []
Pgc_ss = []
Pevap_ss = []
Pihx1_ss = []
Pihx2_ss = []
Pbuff1_ss = []
Pbuff2_ss = []
COP_ss = []
Pheat_out_ss = []
Pcoolers_ss = []
Pbuffers_ss = []
mc_in_dot_ss = []
mbuff1_in_dot_ss = []
pft_ss = []
hft_ss = []
mc_out_dot_ss = []
mft_in_dot_ss = []
mft_out_dot_ss = []
me_in_dot_ss = []
me_out_dot_ss = []
mbuff1_out_dot_ss = []
pbuff1_ss = []
pbuff2_ss = []
mft_dot_ss = []
hihx1_out_ss = []
hihx2_out_ss = []
Tmpg_out_ss = []
Theater1_ss = []
Theater2_ss = []
pc_ss = []
Tw_out_ss = []
t_ss = []
a_Hpev = []
import os
import pandas as pd

# Specify the path to save the CSV file
csv_path = r'outputs_omegab.csv'

# Initialize the CSV file with headers
if not os.path.exists(csv_path):
    with open(csv_path, 'w') as f:
        f.write("pc [bar],Tw_out [°C],Theater1 [°C],Theater2 [°C],Pheat_out [W]\n")
print(total_time_steps)

hpev = d_Hpev[0]
Hpev_prev = hpev
zabri_hpv = 0
for current_time_step in range(total_time_steps):
    # try:
    current_time = current_time_step * dt

    print(current_time)
    omegab = d_omegab[current_time_step]
    mw_dot = d_mw_dot[current_time_step]
    Tw_in = d_Tw_in[current_time_step]
    Hpev = d_Hpev[current_time_step]
    hpev = Hpev 

    Theater1 = a_Theater1[current_time_step] #Theater1_current[0][0]
    Theater2 = a_Theater2[current_time_step] #Theater2_current[0][0]
    # tau_LPV = 10
    # tau_HPV = 20

    # if Hpev > Hpev_prev + 1e-5 or 1 <= zabri_hpv <= 60:
    #     tau_HPV = 60
    #     zabri_hpv += 1
    # else:
    #     zabri_hpv = 0

    hpev = Hpev #hpev + dt/tau_HPV * (Hpev - hpev)
    
    Hpev_prev = Hpev
    algebraic_out = compute_cycle_inputs(
        dt=dt,
        hpev=hpev, lpev=lpev,
        Tw_in=Tw_in, Theater1=Theater1, Theater2=Theater2,
        omega1=omega1, omega2=omega2, omega3=omega3, omegab=omegab,
        config=config, CP=CP, get_state=get_state,
        scaler_X=scaler_X, models_GP=models_GP, scalers_y=scalers_y,
        model1_1_LR=model1_1_LR, model1_2_LR=model1_2_LR, model1_3_LR=model1_3_LR,
        model2_1_LR=model2_1_LR, model2_2_LR=model2_2_LR, model2_3_LR=model2_3_LR,
        model3_1_LR=model3_1_LR, model3_2_LR=model3_2_LR, model3_3_LR=model3_3_LR,
    )
    (mc_in_dot,
        mc_out_dot,
        me_in_dot,
        me_out_dot,
        mft_l_dot,
        mft_v_dot,
        mtc2_dot,
        Ttc2_out,
        hc_in,
        Dc_in,
        hbuff1_in,
        Dbuff1_in,
        hTC1_out,
        hTC2_in,
        Pcooler1_pred,
        Pcooler2_pred,
        Pcooler3_pred,
    ) = algebraic_out

    sol = solve_ivp(cycle, config.t0, X0, method = 'RK23', max_step=0.5, args = (
        mw_dot,
        Tw_in,
        Tmpg_in,
        mmpg_dot,
        mc_in_dot,
        mc_out_dot,
        me_in_dot,
        me_out_dot,
        mft_l_dot,
        mft_v_dot,
        mtc2_dot,
        Ttc2_out,
        hc_in,
        Dc_in,
        hbuff1_in,
        Dbuff1_in
        )
    )

    y = sol.y
    t = sol.t
    X0 = sol.y[:, -1]

    ##########################################################################################################

    (
        a_t,
        a_Pgc,
        a_Pevap,
        a_Pihx1,
        a_Pihx2,
        a_Pbuff1,
        a_Pbuff2,
        Pfhx,
        a_me_dot,
        a_mihx2_dot,
        a_mbuff1_dot,
        a_mbuff2_dot,
        a_mc_dot,
        a_mihx1_dot,
        a_COP,
        a_Pheat_out,
        a_Pcoolers,
        a_Pbuffers,
        a_mc_in_dot,
        a_mbuff1_in_dot,
        a_pft,
        a_pc,
        a_pe,
        a_Tc_out,
        a_Te_out,
        a_hc_out,
        a_he_out,
        a_mc_out_dot,
        a_mft_in_dot,
        a_mft_out_dot,
        a_me_in_dot,
        a_me_out_dot,
        a_hft,
        a_mbuff1_out_dot,
        a_pbuff1,
        a_pbuff2,
        a_mft_dot,
        a_hihx1_out,
        a_hihx2_out,
        a_Tw_out,
        a_Tmpg_out,
        a_SH,
        a_Pcomb,
        a_Prec_total,
    ) = post_process(
        y,
        t,
        mw_dot,
        Tw_in,
        Tmpg_in,
        mmpg_dot,
        mc_in_dot,
        mc_out_dot,
        me_in_dot,
        me_out_dot,
        mft_l_dot,
        mft_v_dot,
        mtc2_dot,
        Ttc2_out,
        hc_in,
        Dc_in,
        hbuff1_in,
        Dbuff1_in,
        hTC1_out,
        hTC2_in,
        Pcooler1_pred,
        Pcooler2_pred,
        Pcooler3_pred,
        omegab
        )
    # print(np.mean(a_Pbuff2))
    # Store steady-state values for the current time step
    t_ss.append(current_time)
    Tw_out_ss.append(np.mean(a_Tw_out))
    pc_ss.append(np.mean(a_pc))
    hc_out_ss.append(np.mean(a_hc_out))
    hihx1_out_ss.append(np.mean(a_hihx1_out))
    hft_ss.append(np.mean(a_hft))
    hihx2_out_ss.append(np.mean(a_hihx2_out))
    he_out_ss.append(np.mean(a_he_out))
    pft_ss.append(np.mean(a_pft))
    pbuff1_ss.append(np.mean(a_pbuff1))
    pe_ss.append(np.mean(a_pe))
    Tc_out_ss.append(np.mean(a_Tc_out))
    Te_out_ss.append(np.mean(a_Te_out))
    mc_in_dot_ss.append(np.mean(a_mc_in_dot))
    mc_out_dot_ss.append(np.mean(a_mc_out_dot))
    mft_in_dot_ss.append(np.mean(a_mft_in_dot))
    mft_out_dot_ss.append(np.mean(a_mft_out_dot))
    mbuff1_in_dot_ss.append(np.mean(a_mbuff1_in_dot))
    mbuff1_out_dot_ss.append(np.mean(a_mbuff1_out_dot))
    me_in_dot_ss.append(np.mean(a_me_in_dot))
    me_out_dot_ss.append(np.mean(a_me_out_dot))
    Pheat_out_ss.append(np.mean(a_Pheat_out))
    COP_ss.append(np.mean(a_COP))

    # if current_time >= 200:
    with open(csv_path, 'a') as f:
        f.write(f"{np.mean(a_pc)},{np.mean(a_Tw_out)},{Theater1 - 273.15},{Theater2 - 273.15}, {np.mean(a_Pheat_out)} \n")
        
    # Define the file path for X0 storage
    X0_csv_path = "X0.csv"

    # Convert X0 to a list format for CSV saving
    X0_values = [config.pe] + config.he + config.Te_wall + \
                [config.pc] + config.hc + config.Tc_wall + \
                [config.pihx1, config.hihx1, config.Tihx_wall, 
                config.pihx2, config.hihx2, config.pbuff1, 
                config.hbuff1, config.Tbuff1_wall, config.pft, config.hft]

    # Define column headers
    X0_headers = ["pe"] + [f"he_{i}" for i in range(len(config.he))] + \
                [f"Te_wall_{i}" for i in range(len(config.Te_wall))] + \
                ["pc"] + [f"hc_{i}" for i in range(len(config.hc))] + \
                [f"Tc_wall_{i}" for i in range(len(config.Tc_wall))] + \
                ["pihx1", "hihx1", "Tihx_wall", "pihx2", "hihx2",
                "pbuff1", "hbuff1", "Tbuff1_wall", "pft", "hft"]

    # Write data to CSV file
    with open(X0_csv_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(X0_headers)  # Write headers
        writer.writerow(X0_values)   # Write values
    # except Exception as e:
    #     # Log the error and continue
    #     print(f"Error at time step {current_time_step}: {e}")
    #     break  # Optionally, remove this line to skip the error and proceed with the next iteration

# import matplotlib.pyplot as plt

# t1_ss = np.linspace(t_ss[0], t_ss[-1], int(total_time_steps))
# # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 10))

# # # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
# # # Plot pgc on the first subplot
# # ax1.plot(t_ss, pc_ss, linestyle = '--', label='Sim high pressure', color='r')
# # ax1.set_ylabel("Pressure [bar]", fontsize=14)
# # ax1.legend(loc='best', fontsize=12)
# # ax1.grid(True)
# # ax1.tick_params(axis='both', which='major', labelsize=14)
# # # ax1.set_title("Output Change According to Input Change")

# # ax2.plot(t_ss, Tw_out_ss, linestyle = '--', label='Sim Water out', color='r')
# # # ax2.set_xlabel("t [s]", fontsize=12)
# # ax2.set_ylabel("Temperature [°C]", fontsize=12)
# # ax2.legend()
# # ax2.grid(True)
# # ax2.tick_params(axis='both', which='major', labelsize=14)


# # ax3.plot(t1_ss, a_Theater1[:int(total_time_steps)], label='Heater 1', color='b')
# # ax3.set_xlabel("time [s]", fontsize=14)
# # ax3.set_ylabel("Heater Temperature [°C]", fontsize=14)
# # ax3.legend(loc='best', fontsize=12)
# # ax3.grid(True)
# # ax3.tick_params(axis='both', which='major', labelsize=14)

# # ax4.plot(t1_ss, a_Hpev[:int(total_time_steps)], label='HPV', color='r')
# # ax4.plot(t1_ss, a_Lpev[:int(total_time_steps)], label='LPV', color='b')    
# # ax4.set_xlabel("time [s]", fontsize=14)
# # ax4.set_ylabel("Valve opening [%]", fontsize=14)
# # ax4.legend(loc='best', fontsize=12)
# # ax4.grid(True)
# # ax4.tick_params(axis='both', which='major', labelsize=14)

# # plt.tight_layout()
# # plt.show()


# plt.figure(1)
# plt.plot(t_ss, hc_out_ss, label='GC', color='r')
# plt.plot(t_ss, hihx1_out_ss, label='IHX1', color='orange')
# plt.plot(t_ss, hft_ss, label='FT', color='green')
# plt.plot(t_ss, hihx2_out_ss, label='IHX2', color='k')
# plt.plot(t_ss, he_out_ss, label='EVAP', color='b')
# plt.ylabel("Enthalpy [J/kg]", fontsize=14)
# plt.legend(loc='best', fontsize=14)
# plt.grid(True)

# plt.figure(2)
# plt.plot(t_ss, pc_ss, linestyle = '--', label='Sim high pressure', color='r')
# # plt.plot(t1_ss, a_pc[:int(config.t0[-1])], label='Exp high pressure', color='r')
# plt.plot(t_ss, pft_ss, label='pft sim', color='gray')
# plt.plot(t_ss, pbuff1_ss, linestyle = '--', label='pbuff1 sim', color='green')
# plt.plot(t_ss, pe_ss, linestyle = '--', label='Sim low pressure', color='b')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylabel("Pressure [bar]", fontsize=14)
# plt.legend(loc='best', fontsize=12)
# plt.grid(True)


# plt.figure(4)
# plt.plot(t_ss, Tc_out_ss, linestyle = '--', label='Sim GC out', color='r')
# plt.plot(t_ss, Te_out_ss, linestyle = '--', label='Sim EVAP out', color='b')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("t [s]", fontsize=12)
# plt.ylabel("Temperature [°C]", fontsize=12)
# plt.legend()
# plt.grid()

# plt.figure(5)
# plt.plot(t_ss, mc_in_dot_ss, label='GC in', linestyle='-', color='blue', linewidth=2, marker='o', markersize=5)
# plt.plot(t_ss, mc_out_dot_ss, label='GC out', linestyle='--', color='orange', linewidth=2, marker='x', markersize=5)
# plt.plot(t_ss, mft_in_dot_ss, label='FT in', linestyle='-.', color='green', linewidth=2, marker='s', markersize=5)
# plt.plot(t_ss, mft_out_dot_ss, label='FT out', linestyle=':', color='red', linewidth=2, marker='^', markersize=5)
# plt.plot(t_ss, mbuff1_in_dot_ss, label='buff1 in', linestyle='-', color='purple', linewidth=2, marker='D', markersize=5)
# plt.plot(t_ss, mbuff1_out_dot_ss, label='buff1 out', linestyle='--', color='brown', linewidth=2, marker='v', markersize=5)
# plt.plot(t_ss, me_in_dot_ss, label='EVAP in', linestyle='-.', color='pink', linewidth=2, marker='>', markersize=5)
# plt.plot(t_ss, me_out_dot_ss, label='EVAP out', linestyle=':', color='cyan', linewidth=2, marker='<', markersize=5)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("t [s]", fontsize=12)
# plt.ylabel("mass flow rate [kg/s]", fontsize=12)
# plt.legend()
# plt.grid()

# plt.figure(6)
# plt.plot(t_ss, COP_ss, linestyle = '--', label = 'Sim', color = 'green')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("t [s]", fontsize=12)
# plt.ylabel("COP [-]", fontsize=12)
# plt.legend(loc='best', fontsize=12)
# plt.grid()

# plt.show()
