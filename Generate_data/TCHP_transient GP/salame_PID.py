from scipy.integrate import solve_ivp
from model import cycle
import matplotlib.pyplot as plt
import numpy as np
from post_processing import post_process
import config
import time
import csv
from config import (
    CP,
)
from algebraic_model import compute_cycle_inputs
from funcs import (
    predict_next_Theater1,
    predict_next_Theater2,
    predict_next_state
)
from utils import (
	get_state,
)

import os
import joblib

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
from utils import(
    interpolate_single,
    calculate_supply_temp,
    calculate_T_diff_sp,
    fuzzy_control,
    fuzzy_control_mw,
    run_pid_omegab,
    # run_pid_Pheat,
    run_pid_omega2,
    run_pid_Theater,
    fuzzy_control_Theater,

    run_pid_Hpev,
    run_pid_Lpev

)
import joblib
A1_coeffs = joblib.load("coeffs/Theater/A1_coeffs.pkl")
B1_coeffs = joblib.load("coeffs/Theater/B1_coeffs.pkl")
D1_coeffs = joblib.load("coeffs/Theater/D1_coeffs.pkl")
poly_U1 = joblib.load("coeffs/Theater/poly_U1.pkl")

A2_coeffs = joblib.load("coeffs/Theater/A2_coeffs.pkl")
B2_coeffs = joblib.load("coeffs/Theater/B2_coeffs.pkl")
D2_coeffs = joblib.load("coeffs/Theater/D2_coeffs.pkl")
poly_U2 = joblib.load("coeffs/Theater/poly_U2.pkl")


Tmpg_in = 275 
p25 = 45e5
Tw_in = 303
Theater1 = 773.15
Theater2 = Theater1
omega1 = 150
omega2 = 150
omegab = 4000
Hpev = 50
Lpev = 50
hpev = Hpev
lpev = Lpev
pc_in = 70e5
pi = 62e5
pe_in = 30e5 #a_pe[j]
state = get_state(CP.PQ_INPUTS, p25, 0)
he_in = state.hmass()
state = get_state(CP.HmassP_INPUTS, he_in, pe_in)
De_in = state.rhomass()
Te_in = state.T()

config.pe = pe_in
for i in range(config.Ne):
    config.Tmpg[i] = Tmpg_in
    config.Te_wall[i] = (Tmpg_in)
    config.he[i] = he_in
    config.De[i] = De_in

pc_in = pc_in
# a3 = np.array([[pc_in/pi[j]], [Theater3/Tw_in[j]], [omega3[j]]])
# a3_T = a3.T
Tc_in = 330
state = get_state(CP.PT_INPUTS, pc_in, Tc_in)
hc_in = state.hmass()

config.pc = pc_in
for i in range(config.Nc):
    config.Tc_w[i] = Tw_in
    config.Tc_wall[i] = (Tw_in)
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

Tr1 = Theater1/Tw_in
Pr1 = p25/config.pe
a1 = np.array([[Pr1], [Tr1], [omega1]])
# a1 = np.array([[config.pe], [p25], [omega1], [Theater1], [Tw_in]])
a1_T = a1.T

pbuff1_in = p25
# hbuff1_in = he_in
Tbuff1_in = 320 #model2_PR.predict(poly2_reg.transform(a1_T))[0]
state = get_state(CP.PT_INPUTS, pbuff1_in, Tbuff1_in)
hbuff1_in = state.hmass()
Dbuff1_in = state.rhomass()
Tbuff1_w_in = Tw_in

config.Tbuff1_w = Tw_in
config.Tbuff1_wall = (Tw_in)
config.pbuff1 =  pbuff1_in
config.hbuff1 = hbuff1_in
config.Dbuff1 = Dbuff1_in

pbuff2_in = pi
# hbuff2_in = he_in
Tbuff2_in = 320 # model2_PR.predict(poly2_reg.transform(a1_T))[0]
state = get_state(CP.PT_INPUTS, pbuff2_in, Tbuff2_in)
hbuff2_in = state.hmass()
Dbuff2_in = state.rhomass()
Tbuff2_w_in = Tw_in

config.Tbuff2_w = Tw_in
config.Tbuff2_wall = (Tw_in)
config.pbuff2 =  pbuff2_in
config.hbuff2 = hbuff2_in
config.Dbuff2 = Dbuff2_in

Tr2 = Theater2/Tw_in
Pr2 = pi/p25
a2 = np.array([[Pr2], [Tr2], [omega2]])
# a2 = np.array([[p25], [pi], [omega2], [Theater2], [Tw_in]])
a2_T = a2.T

pbuff2_in = pi
Tbuff2_in = 320 #model2_PR.predict(poly2_reg.transform(a2_T))[0]
state = get_state(CP.PT_INPUTS, pbuff2_in, Tbuff2_in)
hbuff2_in = state.hmass()
Dbuff2_in = state.rhomass()
Tbuff2_w_in = Tw_in


pft_in = p25
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
# Define simulation parameters
total_time = 3 # Total simulation time in seconds
dt = 1       # Small simulation step size (1 second for actual simulation)
# control_update_interval = 25  # Interval for holding the control inputs constant
total_time_steps = int(total_time / dt)

SH_sp = 5

Tw_in_ss = []
Tw_out_ss = []
Tw_out_sp_ss = []
t_ss = []
Hpev_opt_ss = []
Lpev_opt_ss = []
omegab_opt_ss = []
Pheat_opt_ss = []
pc_ss = []
pc_opt_ss = []
SH_ss = []
SH_sp_ss = []
COP_ss = []
Theater_opt_ss = []
Theater1_ss = []
Theater2_ss = []

Pgc_ss = []
Prec_ss = []
Pcomb_ss = []
Pheat_ss = []
pc_rnn_ss = []
Tw_out_rnn_ss = []
J_ss = []

Theater_opt = Theater2
Hpev_opt = Hpev
Lpev_opt = Lpev
omegab_opt = omegab
old_omegab = omegab_opt
Pheat_opt = 0
Tw_in = Tw_in
Tw_out = Tw_in + 2
omega1 = omega1
omega2 = omega2
omega2_opt = omega2
# PID initialization
I_Hpev = 0
I_Lpev = 0
I_mw = 0
I_omegab = 0
I_omega2 = 0
I_Theater = 0
I_Qheat = 0

J_Hpev = 1

# K_Hp = 50000
# tau_Hp = 20

kp_Hpev = 0.4
ki_Hpev = 0.2

K_mw = 50000
tau_mw = 20

kp_mw = 2e-4
ki_mw = 5.8e-5

J_Lpev = 1
kp_Lpev = 0.2
ki_Lpev = 0.1

J_omegab = 1
kp_omegab = 3000 #159
ki_omegab = 300 #2.65

J_Theater = 1
kp_Theater = 600
ki_Theater = 10

J_omega2 = 1
kp_omega2 = 2 * 9.64
ki_omega2 = 1 * kp_omega2/10

J_Qheat = 1
kp_Qheat = 100* 9.64
ki_Qheat = 100 * kp_Qheat/1500

prev_error_pc = 2e5
prev_error_SH = 3
prev_error_Tdiff = 0
prev_error_Tw_out = 5

prev_Hpev = Hpev_opt
prev_Lpev = Lpev_opt
mw_dot =  0.23
prev_mw_dot = 0.23

Prec_pred = 0

import pandas as pd
csv_file_path = r"C:\Users\ali.salame\Desktop\Python\side codes\day2.csv"
weather_data = pd.read_csv(csv_file_path, encoding="ISO-8859-1", delimiter=';')

T_outdoor_values = []
for hour in range(len(weather_data['T_outdoor [°C]'])):
    T_outdoor = weather_data['T_outdoor [°C]'][hour]
    T_outdoor_values.append(T_outdoor)

# Configuration for heating level
heating_levels = {
    "Low": {"T_supply_ref": 308.15, "Slope": 1.4},
    "Medium": {"T_supply_ref": 318.15, "Slope": 1.5},
    "High": {"T_supply_ref": 328.15, "Slope": 1.3},
    "Very High": {"T_supply_ref": 338.15, "Slope": 1.3},
}

# Parameters
level = "Medium"  # Select heating level
T_room_sp = 293.15  # Room temperature setpoint (20°C in Kelvin)
T_supply_ref = heating_levels[level]["T_supply_ref"]
slope = heating_levels[level]["Slope"]
offset = 0  # System-specific adjustment parameter


k_step = 0
Theater1_current = np.column_stack([Theater1])
U1_current = np.column_stack([omegab_opt])
D1_current = np.column_stack([omega1])

Theater2_current = np.column_stack([Theater2])
U2_current = np.column_stack([omegab_opt])
D2_current = np.column_stack([omega2])

X_current = np.column_stack([pc_in*1e-5, Tw_out])
U_current = np.column_stack([Hpev, Theater2])
D_current = np.column_stack([mw_dot, Tw_in])

Theater_opt = 573.15
import os

# Define CSV file path
csv_file_path = "files/outputs_PID.csv"

# Initialize the file: Write headers if it doesn’t exist
if not os.path.exists(csv_file_path):
    pd.DataFrame(columns=[
        'current_time', 'Tw_in', 'Tw_out', 'Tw_out_sp', 'Theater_opt', 'Theater1', 'Theater2', 'omegab_opt', 'pc_reduced', 'pc', 'pc_opt',
        'Hpev', 'Hpev_opt', 'SH', 'SH_sp', 'Lpev', 'Lpev_opt', 'COP', 'Pgc', 'Prec', 'Pcomb', 'omega1', 'omega2','omega3', 'k_step',
        'T_outdoor'
    ]).to_csv(csv_file_path, index=False, sep = ';')

current_time = 0
Tw_out_reduced = Tw_out
Tw_out_real = Tw_out
error_Tw_out = 5
a_Tw_out_sp = []
a_T_outdoor = []
total_time = 2500
for temporal_time  in range(total_time):
    if temporal_time == 0:
        T_outdoor = 6
    elif temporal_time == 500:
        T_outdoor = 4
    elif temporal_time == 1000:
        T_outdoor = 0
    elif temporal_time == 1500:
        T_outdoor = 2
    elif temporal_time == 2000:
        T_outdoor = 5
    a_T_outdoor.append(T_outdoor)
    Tw_out_sp = calculate_supply_temp(Tw_out, T_outdoor + 273.15, T_room_sp, T_supply_ref, slope=slope, offset=offset)
    Tdiff_sp = calculate_T_diff_sp(Tw_out, T_supply_ref, offset=0)
    Tdiff = (Tw_out - Tw_in)
    error_Tdiff = (Tdiff_sp - Tdiff)
    a_Tw_out_sp.append(Tw_out_sp)

start_time = time.time()
Tc_out_real = 31
for kt in range(total_time):

    Tw_out_sp = a_Tw_out_sp[kt]
    T_outdoor = a_T_outdoor[kt]

    mw_dot = 0.23
    Pheat_opt = interpolate_single([2, 12], [7500, 2200], T_outdoor)

    Tw_in_sp = Tw_out_sp - Pheat_opt/(mw_dot * config.cpw)  #interpolate_single([-10, 10], [40, 26], T_outdoor) + 273.15 #Tw_in #
    Tw_in = Tw_in + dt/100 * (Tw_in_sp - Tw_in)

    Tmpg_in = T_outdoor + 273.15 #interpolate_single([2200, 8500], [8, -10], Pheat_opt) + 273.15 #Tmpg_in
    mmpg_dot = interpolate_single([2200, 7500], [17.4, 7.4], Pheat_opt)/60 #a_mmpg_dot[j]

    omega1 = interpolate_single([2200, 7500], [100, 240], Pheat_opt) #omega1
    # omega2 = interpolate_single([2200, 7500], [120, 220], Pheat_opt) #omega2
    omega3 =interpolate_single([2200, 7500], [100, 160], Pheat_opt) # a_omega3[j]

    # print(Theater_opt)
    Theater_opt = interpolate_single([2200, 7500], [774, 1023], Pheat_opt) + 50 #min((300/11.7) * (Tw_out_sp - 273.15 - 30) + 500 + 273.15, 1073.15)
    pc_opt = (1.47 * (Tw_in - 273.15) + 31.53) #(1.47 * Tc_out_real + 31.53) #(3.3 * (Tw_in - 273.15) - 0.03 * (Tw_in - 273.15)**2 + 6.7) * 1e5

    Theater1_current = predict_next_Theater1(Theater1_current, U1_current, D1_current, A1_coeffs, B1_coeffs, D1_coeffs, poly_U1)
    Theater2_current = predict_next_Theater2(Theater2_current, U2_current, D2_current, A2_coeffs, B2_coeffs, D2_coeffs, poly_U2)

    # print(Theater1_current)
    Theater1 = Theater1_current[0][0]
    Theater2 = Theater2_current[0][0]

    t_span = (0, dt)
    lpev = lpev + dt/tau_LPV * (Lpev - lpev)
    hpev = hpev + dt/tau_HPV * (Hpev - hpev)

    algebraic_out = compute_cycle_inputs(
        dt=dt,
        hpev=hpev, lpev=lpev,
        Tw_in=Tw_in, Theater1=Theater1, Theater2=Theater2,
        omega1=omega1, omega2=omega2, omega3=omega3, omegab=omegab,
        config=config, CP=CP, get_state=get_state,
        scaler_X=scaler_X, models_GP=models_GP, scalers_y=scalers_y,
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
        omegab,
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
    Tw_out_real = np.mean(a_Tw_out) + 273.15
    # Tw_out = Tw_out + 1/10 * (Tw_out_target - Tw_out)
    pc_real = np.mean(a_pc)
    SH_real = np.mean(a_SH)
    COP_real = np.mean(a_COP)
    Pgc_real = np.mean(a_Pgc)
    Pcomb_real = np.mean(a_Pcomb)
    Prec_real = np.mean(a_Prec_total)
    Pheat_out = np.mean(a_Pheat_out)
    Tc_out_real = np.mean(a_Tc_out)

    # Tw_out = Tw_out + dt/200 * (Tw_out_target - Tw_out)
    # Calculate error for PID control
    error_pc = (pc_opt - pc_real) #(pc_opt - np.mean(sol.y[1 + 2 * config.Ne, :]))
    # print(error_pc)
    error_SH = (SH_sp - SH_real)
    error_Theater1 = (Theater_opt - Theater1)
    error_Theater2 = (Theater1 - Theater2)
    error_Tw_out = (Tw_out_sp - Tw_out_real)

    # Theater_opt, I_Theater = run_pid_Theater(error_Tw_out, I_Theater, J_Theater, kp_Theater, ki_Theater) # = interpolate_single([2200, 7500], [774, 1073], Pheat_opt) #min((300/11.7) * (Tw_out_sp - 273.15 - 30) + 500 + 273.15, 1073.15)
    # Theater_opt = min(max(Theater_opt, 773.15), 1073.15)
    # Theater_opt = fuzzy_control_Theater(error_Tw_out, prev_error_Tw_out, Tw_out_real, Theater_opt)

    # prev_error_Tw_out = error_Tw_out

    # Hpev_opt = fuzzy_control(error_pc, prev_error_pc, pc_real + 1, Hpev_opt)
    # Hpev = Hpev_opt
    # prev_error_pc = error_pc

    t0 = time.perf_counter()
    Hpev_opt, I_Hpev = run_pid_Hpev(error_pc, I_Hpev, kp_Hpev, ki_Hpev, 1, ki_Hpev/kp_Hpev)
    Hpev = Hpev_opt
    omegab_opt, I_omegab = run_pid_omegab(error_Theater1, I_omegab, kp_omegab, ki_omegab, 1, ki_omegab/kp_omegab)
    elapsed = time.perf_counter() - t0
    print(f"[Iteration {kt}] PID Hpev time: {elapsed*1000:.4f} ms")

    Lpev_opt, I_Lpev = run_pid_Lpev(error_SH, I_Lpev, kp_Lpev, ki_Lpev, 1, ki_Lpev/kp_Lpev)
    Lpev = Lpev_opt

    # Lpev_opt = fuzzy_control(error_SH, prev_error_SH, SH_real + 1, Lpev_opt)
    # Lpev = Lpev_opt
    # prev_error_SH = error_SH
    # prev_Lpev = Lpev_opt
    omega2_opt, I_omega2 = run_pid_omega2(error_Theater2, I_omega2, kp_omega2, ki_omega2, 1, ki_omega2/kp_omega2)
    omega2 = omega2_opt
    # Pheat_opt, I_Pheat = run_pid_Pheat(error_Tw_out, I_Pheat, J_Pheat, kp_Pheat, ki_Pheat)
    # Pheat_opt = min(max(Pheat_opt, 2000), 8500

    # print(Tw_out - Tw_in)
    print(f"Optimal Lpev: {Lpev_opt}, Optimal Hpev: {Hpev_opt}, Optimal omegab: {omegab_opt}, Pheat_opt: {Pheat_opt}")
    X0 = sol.y[:, -1]

    U1_current = np.column_stack([omegab_opt])
    D1_current = np.column_stack([omega1])
    U2_current = np.column_stack([omegab_opt])
    D2_current = np.column_stack([omega2])
    
    U_current = np.column_stack([Hpev, Theater2])
    D_current = np.column_stack([mw_dot, Tw_in])
    # Create a dictionary for current time step data
    data_dict = {
        'current_time': [current_time],
        'Tw_in': [Tw_in],
        'Tw_out': [Tw_out_real],
        'Tw_out_sp': [Tw_out_sp],
        'Theater_opt': [Theater_opt],
        'Theater1': [Theater1],
        'Theater2': [Theater2],
        'omegab_opt': [omegab_opt],
        'pc_reduced': [pc_real],
        'pc': [pc_real],
        'pc_opt': [pc_opt],
        'Hpev': [Hpev],
        'Hpev_opt': [Hpev_opt],
        'SH': [SH_real],
        'SH_sp': [SH_sp],
        'Lpev': [Lpev],
        'Lpev_opt': [Lpev_opt],
        'COP': [COP_real],
        'Pgc': [Pgc_real],
        'Prec': [Prec_real],
        'Pcomb': [Pcomb_real],
        'omega1': [omega1],
        'omega2': [omega2],
        'omega3': [omega3],
        'k_step': [k_step],
        'T_outdoor': [T_outdoor]


    }

    # Convert to DataFrame and append to CSV
    pd.DataFrame(data_dict).to_csv(csv_file_path, mode='a', header=False, index=False, sep = ';')

    # Define the file path for X0 storage
    X0_csv_path = "files/X0.csv"

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
    # Print every 100 time steps for debugging
    if current_time % 100 == 0:
        print(f"Step {current_time}: Data saved to CSV.")

    Tw_in_ss.append(Tw_in)
    Tw_out_ss.append(Tw_out)  # Record the last output of the current step
    Tw_out_sp_ss.append(Tw_out_sp)
    Hpev_opt_ss.append(Hpev_opt)
    Lpev_opt_ss.append(Lpev_opt)
    Theater_opt_ss.append(Theater_opt)
    Pheat_opt_ss.append(Pheat_opt)
    Theater1_ss.append(Theater1)
    Theater2_ss.append(Theater2)
    omegab_opt_ss.append(omegab_opt)
    t_ss.append(current_time)
    pc_ss.append(pc_real)

    SH_ss.append(SH_real)
    SH_sp_ss.append(SH_sp)

    Pgc_ss.append(Pgc_real)
    Prec_ss.append(Prec_real)
    COP_ss.append(COP_real)
    Pcomb_ss.append(Pcomb_real)
    Pheat_ss.append(Pheat_out)
    current_time += dt

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTotal simulation time: {elapsed_time:.2f} seconds")
