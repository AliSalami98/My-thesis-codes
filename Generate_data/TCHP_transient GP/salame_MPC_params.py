from scipy.integrate import solve_ivp
from model import cycle
import time
import matplotlib.pyplot as plt
import numpy as np
from post_processing import post_process
import config
import csv
import multiprocessing
from config import (
    CP,
)
from funcs import (
    predict_next_Theater1,
    predict_next_Theater2,
    predict_next_state_cell,
    load_RNN,
    load_LSTM,
    LSTMModel,
    RNNModel
)
from scipy.optimize import Bounds, minimize, differential_evolution
from utils import (
	get_state,
    fuzzy_control,
    fuzzy_control_mw,
    run_pid_omegab,
    run_pid_omega2,

    run_pid_Lpev
)
# from ML_Models.compressor_PR import (
#     poly1_reg,
#     model1_PR,
#     poly2_reg,
#     model2_PR,
#     poly3_reg,
#     model3_PR,
# )


from utils import(
    interpolate_single,
    calculate_supply_temp,
    calculate_T_diff_sp,

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


def objective(control_inputs, params):
    Nc, Np, U_current, a_Tw_out_sp_MPC, mw_dot, Tw_in, a_J, sequence_length, prediction_length, rnn_model, scaler_input, scaler_output, alpha = params

    # Split control moves
    omegab_opt_moves = control_inputs[:Nc]
    Hpev_opt_moves = control_inputs[Nc:2*Nc]

    Hpev_plan = np.pad(Hpev_opt_moves, (0, prediction_length - Nc), mode="edge")
    omegab_plan = np.pad(omegab_opt_moves, (0, prediction_length - Nc), mode="edge")

    # print(omegab_plan)
    # Build future control sequence (shape: [20, 4])
    future_controls = np.column_stack([
        omegab_plan,
        Hpev_plan,
        np.full(prediction_length, mw_dot),
        np.full(prediction_length, Tw_in)
    ])

    # Predict next 20 steps
    X_next = predict_next_state_cell(rnn_model, scaler_input, scaler_output, U_current, future_controls, sequence_length, prediction_length)  # shape: (20, 2)
    Tw_out_rnn = X_next[:, 0]    # In Kelvin

    # RNN
    # alpha = 1     # Tracking Tw_out
    beta = (1 - alpha)     # Penalize omegab usage
    gamma = 1e-1  # Penalize Δpc
    delta = 1e-1   # Penalize Δomegab
    sanka = 5e-4

    # LSTM
    # alpha = 1     # Tracking Tw_out
    # beta = 0    # Penalize omegab usage
    # gamma = 2e-2  # Penalize Δpc
    # delta = 2e-2   # Penalize Δomegab
    # sanka = 7e-5

    # Temperature tracking error (normalize)
    T_norm = (Tw_out_rnn - np.array(a_Tw_out_sp_MPC)) / (293 - np.array(a_Tw_out_sp_MPC))

    # print(np.array(a_Tw_out_sp_MPC))
    # Normalize control usage
    omegab_norm = (omegab_plan - 1950) / (9500 - 1950)
    Hpev_norm = (Hpev_plan - 11) / 89

    # Compute deltas (include previous control step)
    prev_omegab = U_current[-1, 1]
    prev_Hpev = U_current[-1, 2]
    Delta_Hpev = np.diff(np.insert(Hpev_plan, 0, prev_Hpev))
    Delta_omegab = np.diff(np.insert(omegab_plan, 0, prev_omegab))
    normalized_Delta_Hpev = Delta_Hpev / 89
    normalized_Delta_omegab = Delta_omegab / 7500

    # Optional saturation penalties (not activated)
    penalty_Hpev_sat = Hpev_norm * (1 - Hpev_norm)
    penalty_omegab_sat = omegab_norm * (1 - omegab_norm)

    # print(normalized_Delta_Hpev )
    # Final cost calculation
    cost = (
        alpha * np.sum(T_norm ** 2) +
        beta * np.sum(omegab_norm ** 2) +
        gamma * np.sum(normalized_Delta_Hpev ** 2) +
        delta * np.sum(normalized_Delta_omegab ** 2)+
        sanka * (1 - penalty_Hpev_sat).sum() + 0 * (1 - penalty_omegab_sat).sum()
    )
    X_prev = X_next
    total_cost = cost / prediction_length
    a_J.append(total_cost)
    return total_cost


def run_mpc(bounds, initial_guess, params):
    Nc, Np, U_current, a_Tw_out_sp_MPC, mw_dot, Tw_in, a_J, sequence_length, prediction_length, rnn_model, scaler_input, scaler_output, alpha = params
    
    result = minimize(
        objective,
        x0=initial_guess,
        bounds= bounds,
        method='Nelder-Mead',
        args=(params,),                # <<< pass params to objective
        options={'maxiter': 100}
    )

    # result = differential_evolution(
    #     func=objective,
    #     bounds=bounds,
    #     strategy='best1bin',
    #     maxiter=100,
    #     disp=True
    # )
    omegab_opt_moves = result.x[:Nc]
    Hpev_opt_moves = result.x[Nc:2*Nc]
    
    # Return the first control move to apply now
    return omegab_opt_moves[0], Hpev_opt_moves[0]
import matplotlib.pyplot as plt

def simulate_alpha(alpha, k):    # Define simulation parameters

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
    config.hft = hft_in
    config.Dft = Dft_in

    tau_HPV = 20
    tau_LPV = 10

    X0 = ([config.pe] + config.he + config.Te_wall + config.Tmpg
        + [config.pc] + config.hc+ config.Tc_wall + config.Tc_w
        + [config.pihx1, config.hihx1, config.Tihx_wall, config.pihx2, config.hihx2,
        config.pbuff1, config.hbuff1, config.Tbuff1_wall, config.pbuff2, config.hbuff2, config.Tbuff2_wall,
        config.pft, config.hft,
        Lpev, Hpev])


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

    # Example usage setup
    if __name__ == "__main__":
        model_path = "coeffs/RNN/rnn_model.pth"
        input_size = 5    # Number of input features (state + inputs)
        output_size = 1   # Number of output features per time step (pc, Tw_out)
        hidden_size = 32
        num_layers = 1
        dropout = 0.1
        sequence_length = 10 # Define the sequence length used during training
        prediction_length = 10
        Np = prediction_length
        Nc = 1
        rnn_model = load_RNN(model_path, input_size, hidden_size, output_size, num_layers)

        scaler_input = joblib.load("coeffs/RNN/scaler_input.pkl")
        scaler_output = joblib.load("coeffs/RNN/scaler_output.pkl")


    import os

    # Define CSV file path
    csv_file_path = "files/outputs_rnn.csv"

    # # Example usage setup
    # # if __name__ == "__main__":
    # model_path = "coeffs/LSTM/lstm_model.pth"
    # input_size = 5    # Number of input features (state + inputs)
    # output_size = 1   # Number of output features per time step (pc, Tw_out)
    # hidden_size = 32
    # num_layers = 1
    # dropout = 0.1
    # sequence_length = 10 # Define the sequence length used during training
    # prediction_length = 10
    # Np = prediction_length
    # Nc = 1
    # rnn_model = load_LSTM(model_path, input_size, hidden_size, output_size, num_layers)

    # scaler_input = joblib.load("coeffs/LSTM/scaler_input.pkl")
    # scaler_output = joblib.load("coeffs/LSTM/scaler_output.pkl")


    # import os

    # # Define CSV file path
    # csv_file_path = "files/outputs_lstm.csv"

    # # Initialize the file: Write headers if it doesn’t exist
    # if not os.path.exists(csv_file_path):
    #     pd.DataFrame(columns=[
    #         'current_time', 'Tw_in', 'Tw_out_rnn', 'Tw_out', 'Tw_out_sp', 'omegab_opt', 'Theater1', 'Theater2', 'Theater_opt', 'pc_rnn', 'pc', 
    #         'Hpev', 'Hpev_opt', 'SH', 'SH_sp', 'Lpev', 'Lpev_opt', 'COP', 'Pgc', 'Prec', 'Pcomb', 'omega1', 'omega2','omega3', 'k_step',
    #         'T_outdoor'
    #     ]).to_csv(csv_file_path, index=False, sep = ';')

    # === Prepare output file (once) ===
    out_path = f"files/file{k}.csv"
    os.makedirs("files", exist_ok=True)

    header = [
        'current_time','Tw_in','Tw_out_rnn','Tw_out','Tw_out_sp','omegab_opt',
        'Theater1','Theater2','Theater_opt','pc_rnn','pc','Hpev','Hpev_opt',
        'SH','SH_sp','Lpev','Lpev_opt','COP','Pgc','Prec','Pcomb',
        'omega1','omega2','omega3','k_step','T_outdoor'
    ]
    write_header = not os.path.exists(out_path)

    # open once; append every loop iteration
    f = open(out_path, mode='a', newline='')
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)

    Tw_out_rnn = Tw_out
    pc_rnn = config.pc * 1e-5
    pc_opt = pc_rnn

    initial_guess = np.concatenate([np.full(Nc, omegab_opt), np.full(Nc, Hpev_opt)])
    bounds = [(1950, 9500)] * Nc + [(11, 100)] * Nc

    U = np.array([Tw_out_rnn, omegab_opt, Hpev_opt, mw_dot, Tw_in])
    U_current = np.tile(U, (sequence_length, 1))
    X_current = np.array([Tw_out_rnn])
    X_prev = [Tw_out]

    a_Tw_out_sp = []
    a_T_outdoor = []
    total_time = 2000
    for temporal_time  in range(total_time):
        if temporal_time == 0:
            T_outdoor = 6
        elif temporal_time == 1000:
            T_outdoor = 2
        # elif temporal_time == 800:
        #     T_outdoor = 2
        # elif temporal_time == 1200:
        #     T_outdoor = 5
        # elif temporal_time == 1600:
        #     T_outdoor = 7
        a_T_outdoor.append(T_outdoor)
        Tw_out_sp = calculate_supply_temp(Tw_out, T_outdoor + 273.15, T_room_sp, T_supply_ref, slope=slope, offset=offset)
        Tdiff_sp = calculate_T_diff_sp(Tw_out, T_supply_ref, offset=0)
        Tdiff = (Tw_out - Tw_in)
        error_Tdiff = (Tdiff_sp - Tdiff)
        a_Tw_out_sp.append(Tw_out_sp)

    future_controls = np.column_stack([
        np.full(prediction_length, omegab_opt),
        np.full(prediction_length, Hpev_opt),
        np.full(prediction_length, mw_dot),
        np.full(prediction_length, Tw_in)
    ])
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

        Theater1_current = predict_next_Theater1(Theater1_current, U1_current, D1_current, A1_coeffs, B1_coeffs, D1_coeffs, poly_U1)
        Theater2_current = predict_next_Theater2(Theater2_current, U2_current, D2_current, A2_coeffs, B2_coeffs, D2_coeffs, poly_U2)


        Theater1 = Theater1_current[0][0]
        Theater2 = Theater2_current[0][0]

        X_next = predict_next_state_cell(rnn_model, scaler_input, scaler_output, U_current, future_controls, sequence_length, prediction_length)  # shape: (20, 2)

        Tw_out_rnn = X_next[0, 0]    # In Kelvin

        a_J = []
        if kt < total_time - Np:
            a_Tw_out_sp_MPC = a_Tw_out_sp[kt:kt + Np]
        else:
            a_Tw_out_sp_MPC = a_Tw_out_sp[-Np:]
        params = (Nc, Np, U_current, a_Tw_out_sp_MPC, mw_dot, Tw_in, a_J, sequence_length, prediction_length, rnn_model, scaler_input, scaler_output, alpha)

        # if kt > sequence_length:
        omegab_opt, Hpev_opt = run_mpc(bounds, initial_guess, params)

        # print(pc_opt)
        # Build future control sequence (shape: [20, 4])
        future_controls = np.column_stack([
            np.full(prediction_length, omegab_opt),
            np.full(prediction_length, Hpev_opt),
            np.full(prediction_length, mw_dot),
            np.full(prediction_length, Tw_in)
        ])


        t_span = (0, dt)

        sol = solve_ivp(cycle, t_span, X0, method='RK23', args=(
            mw_dot,
            Tw_in,
            Tmpg_in,
            mmpg_dot,
            Theater1,
            Theater2,
            omega1,
            omega2_opt,
            omega3,
            Lpev_opt,
            Hpev_opt,
            omegab_opt,
            tau_LPV,
            tau_HPV
        ))

        # Update initial conditions for the next iteration
        y = sol.y
        t = sol.t
        X0 = sol.y[:, -1]

        # Post-process the results
        (
            a_t,
            a_Pgc,
            a_Pevap,
            a_Pihx1,
            a_Pihx2,
            a_Pbuff1,
            a_Pbuff2,
            Pcooler1_pred,
            Pcooler23_pred,
            Prec_pred,
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
            a_Prec_tot,
            a_Hpev
        ) = post_process(
            y,
            t,
            mw_dot,
            Tw_in,
            Tmpg_in,
            mmpg_dot,
            Theater1,
            Theater2,
            omega1,
            omega2_opt,
            omega3,
            Lpev_opt,
            Hpev_opt,
            omegab_opt,
        )
        Tw_out_real = np.mean(a_Tw_out) + 273.15
        # Tw_out = Tw_out + 1/10 * (Tw_out_target - Tw_out)
        pc_real = np.mean(a_pc)
        SH_real = np.mean(a_SH)
        COP_real = np.mean(a_COP)
        Pgc_real = np.mean(a_Pgc)
        Pcomb_real = np.mean(a_Pcomb)
        Prec_real = np.mean(a_Prec_tot)
        Pheat_out = np.mean(a_Pheat_out)
        Hpev_real = np.mean(a_Hpev)

        # Tw_out = Tw_out + dt/200 * (Tw_out_target - Tw_out)
        # print(error_pc)
        error_SH = (SH_sp - SH_real)
        error_Theater1 = (Theater_opt - Theater1)
        error_Theater2 = (Theater1 - Theater2)
        error_Tw_out = (Tw_out_sp - Tw_out_real)

        error_Tsupply = (Tw_out_sp - Tw_out_real)

        Lpev_opt, I_Lpev = run_pid_Lpev(error_SH, I_Lpev, kp_Lpev, ki_Lpev, 1, ki_Lpev/kp_Lpev)
        Lpev = Lpev_opt

        omega2_opt, I_omega2 = run_pid_omega2(error_Theater2, I_omega2, kp_omega2, ki_omega2, 1, ki_omega2/kp_omega2)
        omega2 = omega2_opt

        # Qheat_opt, I_Qheat = run_pid_omegab(error_Tsupply, I_Qheat, J_Qheat, kp_Qheat, ki_Qheat)
        # Qheat_opt = min(max(Qheat_opt, 2000), 8500)

        # print(f"Optimal Lpev: {Lpev_opt}, Optimal Hpev: {Hpev_opt}, Optimal omegab: {omegab_opt}")
        initial_guess = np.concatenate([np.full(Nc, omegab_opt), np.full(Nc, Hpev_real)])

        U1_current = np.column_stack([omegab_opt])
        D1_current = np.column_stack([omega1])
        U2_current = np.column_stack([omegab_opt])
        D2_current = np.column_stack([omega2])
        
        U_new = np.array([Tw_out_real, omegab_opt, Hpev_real, mw_dot, Tw_in])
        U_current = np.vstack([U_current, U_new])
        # Create a dictionary for current time step data
        df = pd.DataFrame([{
            'current_time': kt,
            'Tw_in': Tw_in,
            'Tw_out_rnn': Tw_out_rnn,
            'Tw_out': Tw_out_real,
            'Tw_out_sp': Tw_out_sp,
            'omegab_opt': omegab_opt,
            'Theater1': Theater1,
            'Theater2': Theater2,
            'Theater_opt': Theater2,
            'pc_rnn': pc_rnn,
            'pc': pc_real,
            'Hpev': Hpev,
            'Hpev_opt': Hpev_real,
            'SH': SH_real,
            'SH_sp': SH_sp,
            'Lpev': Lpev,
            'Lpev_opt': Lpev_opt,
            'COP': COP_real,
            'Pgc': Pgc_real,
            'Prec': Prec_real,
            'Pcomb': Pcomb_real,
            'omega1': omega1,
            'omega2': omega2,
            'omega3': omega3,
            'k_step': k_step,
            'T_outdoor': T_outdoor
        }])

        df.to_csv(
            out_path,
            mode='a',                 # <-- append
            header=write_header,      # <-- write header only once
            index=False
        )
        write_header = False          # <-- subsequent rows won't write header


# Main multiprocessing function
if __name__ == "__main__":
    a_alpha = [0.98, 0.985, 0.99, 0.995, 1]
    a_k = [0, 1, 2, 3, 4, 5]
    processes = []

    for alpha, k in zip(a_alpha, a_k):
        process = multiprocessing.Process(target=simulate_alpha, args=(alpha, k))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()