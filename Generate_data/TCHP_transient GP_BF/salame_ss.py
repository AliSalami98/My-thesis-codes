from scipy.integrate import solve_ivp
from model import cycle
from post_processing import post_process
# from plot import plot_and_print
import numpy as np
import config
import time

from config import CP
from utils import (
	get_state,
)
from algebraic_model import compute_cycle_inputs
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
# from ML_Models.compressor_PR import (
#     poly1_reg,
#     model1_PR,
#     poly2_reg,
#     model2_PR,
#     poly3_reg,
#     model3_PR,
# )
#
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
from data_filling_ss import data
from data_filling_ss import (
    d_omegab,
    d_omega1,
    d_omega2,
    d_omega3,
    d_Theater1,
    d_Theater2,
    d_mw_dot,
    d_mmpg_dot,
    d_Tw_in,
    d_Tw_out,
    d_Tmpg_out,
    d_Text,
    d_Tmpg_in,
    d_Lpev,
    d_Hpev,
    d_pc,
    d_p25,
    d_pi,
    d_pe,
    d_Pcomb,
    d_Pbuffer1,
    d_Pbuffer2,
    d_Pcooler1,
    d_Pcooler2,
    d_Pcooler3,
    d_Pfhx,
    d_Tc_in,
    d_Tc_out,
    d_Te_out,
    d_Pheat_out,
    d_Pc,
    d_Pe,
    d_mc_dot,
    d_me_dot,
    d_COP,
    d_SH
)

Pgc_av = []
Pevap_av = []
Pihx1_av = []
Pihx2_av = []
Pbuff1_av = []
Pbuff2_av = []
Pcooler1_av = []
Pcooler2_av = []
Pcooler3_av = []
Pfhx_av = []
Pheat_out_av = []
me_out_dot_av = []
mc_in_dot_av = []
SH_av = []
pe_av = []
pc_av = []
hc_out_av = []
he_out_av = []
Tc_in_av = []
Tc_out_av = []
Te_out_av = []
Tmpg_out_av =  []
Tw_out_av = []
Tbuff1_out_av = []
Tbuff2_out_av = []
Tihx1_out_av = []
Tihx2_out_av = []
COP_av = []

sample_index = 0
while sample_index < len(d_Hpev):
    print(sample_index)
    pe_in = d_pe[sample_index] - 2e5
    state = get_state(CP.PQ_INPUTS, d_p25[sample_index], 0)
    he_in = state.hmass()
    state = get_state(CP.HmassP_INPUTS, he_in, pe_in)
    De_in = state.rhomass()
    Te_in = state.T()


    config.pe =  pe_in
    for i in range(config.Ne):
        config.Tmpg[i] = d_Tmpg_in[sample_index]
        config.Te_wall[i] = (d_Tmpg_in[sample_index])
        config.he[i] = he_in
        config.De[i] = De_in


    mw2_dot = 0.7 * d_mw_dot[sample_index]
    mw1_dot_2 = 0.5 * 0.3 * d_mw_dot[sample_index]
    mw1_dot_1 = 0.5 * 0.3 * d_mw_dot[sample_index]

    pc_in = d_pc[sample_index] + 2e5
    # a3 = np.array([[pc_in/pi[sample_index]], [Theater3/Tw_in[sample_index]], [omega3[sample_index]]])
    # a3_T = a3.T
    Tc_in = 320
    state = get_state(CP.PT_INPUTS, pc_in, Tc_in)
    hc_in = state.hmass()

    config.pc =  pc_in
    for i in range(config.Nc):
        config.Tc_w[i] = d_Tw_in[sample_index]
        config.Tc_wall[i] = (d_Tw_in[sample_index])
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

    Tr1 = d_Theater1[sample_index]/d_Tw_in[sample_index]
    Pr1 = d_p25[sample_index]/config.pe
    # a1 = np.array([[Pr1], [Tr1], [omega1[sample_index]]])
    a1 = np.array([[config.pe], [d_p25[sample_index]], [d_omega1[sample_index]], [d_Theater1[sample_index]], [d_Tw_in[sample_index]]])
    a1_T = a1.T

    pbuff1_in = d_p25[sample_index]
    # hbuff1_in = he_in
    Tbuff1_in = 320 # model2_PR.predict(poly2_reg.transform(a1_T))[sample_index]
    state = get_state(CP.PT_INPUTS, pbuff1_in, Tbuff1_in)
    hbuff1_in = state.hmass()
    Dbuff1_in = state.rhomass()
    Tbuff1_w_in = d_Tw_in[sample_index]

    config.Tbuff1_w = d_Tw_in[sample_index]
    config.Tbuff1_wall = (d_Tw_in[sample_index])
    config.pbuff1 =  pbuff1_in
    config.hbuff1 = hbuff1_in
    config.Dbuff1 = Dbuff1_in

    pbuff2_in = d_pi[sample_index]
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

    pft_in = d_p25[sample_index]
    hft_in = (he_in + hc_in)/2 #hc_in
    state = get_state(CP.HmassP_INPUTS, hft_in, pft_in)
    Tft_in = state.T()
    Dft_in = state.rhomass()

    config.pft =  pft_in
    state = get_state(CP.PQ_INPUTS, pft_in, 0)
    config.hft_l = state.hmass()
    config.Dft_l = state.rhomass()
    state = get_state(CP.PQ_INPUTS, pft_in, 1)
    config.hft_v = state.hmass()
    config.Dft_v = state.rhomass()
    config.hft = hft_in
    config.Dft = Dft_in


    mc_in_dot = 10**(-3) * (5.14e-3 * d_omegab[sample_index] - 2.14e1 * pc_in/d_p25[sample_index] - 1.11e-3 * d_omega2[sample_index] + 23.38)
    mc_out_dot = 8 * 10**(-9) * d_Hpev[sample_index] * np.sqrt(2*config.Dc[-1] * max(config.pc - config.pft, 0))
    mft_v_dot = 8 * 10**(-7) *np.sqrt(2*config.Dft * max(config.pft - config.pbuff1, 0))
    mft_l_dot = 4 * 10**(-9) * d_Lpev[sample_index] * np.sqrt(2*config.Dft * max(config.pft - config.pe, 0)) #0.000145 * Lpev  #
    me_out_dot = 10**(-3) * (3.843e-3 * d_omegab[sample_index] - 3.99e+1 * Pr1 + 2.18e-2 * d_omega1[sample_index] + 42.36)
    me_in_dot = mft_l_dot
    mbuff2_in_dot = mc_in_dot

    tauc_in = 1
    tauc_out = 100
    taue_in = 40
    taue_out = 1
    tauft_v = 20
    tauft_l = 20
    taubuff2_in = 1
    tau_HPV = 20
    tau_LPV = 10


    X0 = ([config.pe] + config.he + config.Te_wall + config.Tmpg
        + [config.pc] + config.hc+ config.Tc_wall + config.Tc_w
        + [config.pihx1, config.hihx1, config.Tihx_wall, config.pihx2, config.hihx2,
        config.pbuff1, config.hbuff1, config.Tbuff1_wall, config.pbuff2, config.hbuff2, config.Tbuff2_wall,
        config.pft, config.hft])

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
    SH_ss = []
    pc_ss = []
    Tw_out_ss = []
    t_ss = []


    start_time = time.time()
    total_time_steps = 200

    mw_dot = d_mw_dot[sample_index]
    Tw_in = d_Tw_in[sample_index]
    Tmpg_in = d_Tmpg_in[sample_index]
    mmpg_dot = d_mmpg_dot[sample_index]
    Theater1 = d_Theater1[sample_index]
    Theater2 = d_Theater2[sample_index]
    omega1 = d_omega1[sample_index]
    omega2 = d_omega2[sample_index]
    omega3 = d_omega3[sample_index]
    Lpev = d_Lpev[sample_index]
    Hpev = d_Hpev[sample_index]
    omegab = d_omegab[sample_index]

    for current_time_step in range(total_time_steps):
        dt = 1
        lpev = Lpev
        hpev = Hpev
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
            omegab,
        ) = algebraic_out
        current_time = current_time_step * dt
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
        t_ss.append(current_time)
        Tmpg_out_ss.append(np.mean(a_Tmpg_out))
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
        Pgc_ss.append(np.mean(a_Pgc))
        Pevap_ss.append(np.mean(a_Pevap))
        COP_ss.append(np.mean(a_COP))
        SH_ss.append(np.mean(a_SH))

    sample_index += 1

    # Pbuff1_av.append(np.mean(Pbuff1_ss[-10:]))
    # Pcooler1_av.append(np.mean(Pcooler1_ss[-10:]))
    # Pcooler3_av.append(np.mean(Pcooler3_ss[-10:]))
    # Pfhx_av.append(np.mean(Pfhx_ss[-10:]))
    # Tc_in_av.append(np.mean(Tc_in_ss[-10:]))

    Pgc_av.append(np.mean(Pgc_ss[-10:]))
    Pevap_av.append(np.mean(Pevap_ss[-10:]))
    pc_av.append(np.mean(pc_ss[-10:]))
    pe_av.append(np.mean(pe_ss[-10:]))
    SH_av.append(np.mean(SH_ss[-10:]))
    Tc_out_av.append(np.mean(Tc_out_ss[-10:]))
    Te_out_av.append(np.mean(Te_out_ss[-10:]))
    Tw_out_av.append(np.mean(Tw_out_ss[-10:]))
    Tmpg_out_av.append(np.mean(Tmpg_out_ss[-10:]))
    hc_out_av.append(np.mean(hc_out_ss[-10:]))
    he_out_av.append(np.mean(he_out_ss[-10:]))
    COP_av.append(np.mean(COP_ss[-10:]))
    Pheat_out_av.append(np.mean(Pheat_out_ss[-10:]))
    mc_in_dot_av.append(np.mean(mc_in_dot_ss[-10:]) * 1e3)
    me_out_dot_av.append(np.mean(me_out_dot_ss[-10:]) * 1e3)

# RK23, DOP853, Radau, BDF, LSODA
end_time = time.time()

# Calculate and print the time taken
execution_time = end_time - start_time
print(f"Time taken by solve_ivp: {execution_time:.4f} seconds")

import pandas as pd
import os

# Prepare the DataFrame from your averaged values
df_av = pd.DataFrame({
    "Pgc_av": Pgc_av,
    "Pevap_av": Pevap_av,
    "pc_av": pc_av,
    "pe_av": pe_av,
    "SH_av": SH_av,
    "Tc_out_av": Tc_out_av,
    "Te_out_av": Te_out_av,
    "Tw_out_av": Tw_out_av,
    "Tmpg_out_av": Tmpg_out_av,
    "hc_out_av": hc_out_av,
    "he_out_av": he_out_av,
    "COP_av": COP_av,
    "Pheat_out_av": Pheat_out_av,
    "mc_in_dot_av": mc_in_dot_av,
    "me_out_dot_av": me_out_dot_av
})

# Save in the same folder as your script
output_path = os.path.join(os.path.dirname(__file__), "ss.csv")
df_av.to_csv(output_path, index=False)

print(f"CSV file saved to: {output_path}")

# error_Pc = [np.abs(pred - real) / real * 100 for real, pred in zip(Pgc_av, d_Pgc)]
# error_Pe = [np.abs(pred - real) / real * 100 for real, pred in zip(Pevap_av, d_Pevap)]

# # # Identify indices of the top 3 errors for Pc and Pe
# top_Pc_error_indices = sorted(range(len(error_Pc)), key=lambda i: error_Pc[i], reverse=True)[:3]
# top_Pe_error_indices = sorted(range(len(error_Pe)), key=lambda i: error_Pe[i], reverse=True)[:3]

# print(top_Pc_error_indices)
# print(top_Pe_error_indices)


# pc_real = [x * 1e-5 for x in d_pc]
# pe_real = [x * 1e-5 for x in d_pe]

# # top_pc_error_indices = sorted(range(len(error_pc)), key=lambda i: error_pc[i], reverse=True)[:3]

# Tc_in_real = [x - 273.15 for x in d_Tc_in]
# Tc_out_real = [x - 273.15 for x in d_Tc_out]
# Te_out_real = [x - 273.15 for x in d_Te_out]
# Tw_out_real = [x - 273.15 for x in d_Tw_out]
# Tmpg_out_real = [x - 273.15 for x in d_Tmpg_out]

# mc_dot_real = [x * 1e3 for x in d_mc_dot]
# me_dot_real = [x * 1e3 for x in d_me_dot]

# from scipy.stats import linregress

# # Function to calculate MAPE
# def calculate_mape(actual, predicted):
#     actual, predicted = np.array(actual), np.array(predicted)
#     return np.mean(np.abs((actual - predicted) / actual)) * 100

# # Function to calculate R² (manual method, since linregress assumes a linear fit)
# def calculate_r2(actual, predicted):
#     actual, predicted = np.array(actual), np.array(predicted)
#     ss_tot = np.sum((actual - np.mean(actual)) ** 2)
#     ss_res = np.sum((actual - predicted) ** 2)
#     return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

# def calculate_mae(y_true, y_pred):
#     return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# # MAPE Calculations (non-temperature)
# mape_Pc = calculate_mape(d_Pgc, Pgc_av)
# mape_Pe = calculate_mape(d_Pevap, Pevap_av)
# mape_pc = calculate_mape(pc_real, pc_av)
# mape_pe = calculate_mape(pe_real, pe_av)
# mape_Pheat_out = calculate_mape(d_Pheat_out, Pheat_out_av)
# mape_COP = calculate_mape(d_COP, COP_av)

# # MAE Calculations (temperature)
# mae_Tc_out = calculate_mae(Tc_out_real, Tc_out_av)
# mae_Te_out = calculate_mae(Te_out_real, Te_out_av)
# mae_Tw_out = calculate_mae(Tw_out_real, Tw_out_av)
# mae_Tmpg_out = calculate_mae(Tmpg_out_real, Tmpg_out_av)
# mae_SH = calculate_mae(d_SH, SH_av)

# # R² Calculations (same)
# r2_Pc = calculate_r2(d_Pgc, Pgc_av)
# r2_Pe = calculate_r2(d_Pevap, Pevap_av)
# r2_pc = calculate_r2(pc_real, pc_av)
# r2_pe = calculate_r2(pe_real, pe_av)
# r2_Tc_out = calculate_r2(Tc_out_real, Tc_out_av)
# r2_Te_out = calculate_r2(Te_out_real, Te_out_av)
# r2_SH = calculate_r2(d_SH, SH_av)

# r2_Tw_out = calculate_r2(Tw_out_real, Tw_out_av)
# r2_Tmpg_out = calculate_r2(Tmpg_out_real, Tmpg_out_av)
# r2_Pheat_out = calculate_r2(d_Pheat_out, Pheat_out_av)
# r2_COP = calculate_r2(d_COP, COP_av)


# print(f'GC power MAPE: {mape_Pc:.2f}% | R²: {r2_Pc:.4f}')
# print(f'EVAP power MAPE: {mape_Pe:.2f}% | R²: {r2_Pe:.4f}')
# print(f'GC pressure MAPE: {mape_pc:.2f}% | R²: {r2_pc:.4f}')
# print(f'EVAP pressure MAPE: {mape_pe:.2f}% | R²: {r2_pe:.4f}')
# print(f'GC Temperature MAE: {mae_Tc_out:.2f} K | R²: {r2_Tc_out:.4f}')
# print(f'SH MAE: {mae_SH:.2f} K | R²: {r2_SH:.4f}')
# print(f'EVAP Temperature MAE: {mae_Te_out:.2f} K | R²: {r2_Te_out:.4f}')
# print(f'Water outlet Temperature MAE: {mae_Tw_out:.2f} K | R²: {r2_Tw_out:.4f}')
# print(f'MPG outlet Temperature MAE: {mae_Tmpg_out:.2f} K | R²: {r2_Tmpg_out:.4f}')
# print(f'Heating power MAPE: {mape_Pheat_out:.2f}% | R²: {r2_Pheat_out:.4f}')
# print(f'COP MAPE: {mape_COP:.2f}% | R²: {r2_COP:.4f}')

# import matplotlib.pyplot as plt


# fig1 = plt.figure(1)
# plt.plot(d_COP, d_COP, c='k', label='Ideal line')
# COP_sorted = np.sort(d_COP)
# plt.plot(COP_sorted, 0.9 * COP_sorted, linestyle='--', color='blue', label='5% error')
# plt.plot(COP_sorted, 1.1 * COP_sorted, linestyle='--', color='blue')
# plt.scatter(d_COP, COP_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('measured COP [-]', fontsize=12)
# plt.ylabel('predicted COP [-]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# # Plot 2: Gas Cooler Power
# fig2 = plt.figure(2)
# plt.plot(d_Pgc, d_Pgc, c='k', label='Ideal line')
# Pc_data_sorted = np.sort(d_Pgc)
# plt.plot(Pc_data_sorted, 0.85 * Pc_data_sorted, linestyle='--', color='blue', label='15% error')
# plt.plot(Pc_data_sorted, 1.15 * Pc_data_sorted, linestyle='--', color='blue')
# plt.scatter(d_Pgc, Pgc_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured Gas Cooler Power [W]', fontsize=12)
# plt.ylabel('Predicted Gas Cooler Power [W]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# # Plot 3: Evaporator Power
# fig3 = plt.figure(3)
# plt.plot(d_Pevap, d_Pevap, c='k', label='Ideal line')
# Pe_data_sorted = np.sort(d_Pevap)
# plt.plot(Pe_data_sorted, 0.85 * Pe_data_sorted, linestyle='--', color='blue', label='15% error')
# plt.plot(Pe_data_sorted, 1.15 * Pe_data_sorted, linestyle='--', color='blue')
# plt.scatter(d_Pevap, Pevap_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured Evaporator Power [W]', fontsize=12)
# plt.ylabel('Predicted Evaporator Power [W]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# # Plot 4: Gas Cooler Pressure
# fig4 = plt.figure(4)
# plt.plot(pc_real, pc_real, c='k', label='Ideal line')
# pc_data_sorted = np.sort(pc_real)
# plt.plot(pc_data_sorted, 0.95 * pc_data_sorted, linestyle='--', color='blue', label='5% error')
# plt.plot(pc_data_sorted, 1.05 * pc_data_sorted, linestyle='--', color='blue')
# plt.scatter(pc_real, pc_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured Gas Cooler Pressure [Bar]', fontsize=12)
# plt.ylabel('Predicted Gas Cooler Pressure [Bar]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# # Plot 5: Evaporation Pressure
# fig5 = plt.figure(5)
# plt.plot(pe_real, pe_real, c='k', label='Ideal line')
# pe_data_sorted = np.sort(pe_real)
# plt.plot(pe_data_sorted, 0.95 * pe_data_sorted, linestyle='--', color='blue', label='5% error')
# plt.plot(pe_data_sorted, 1.05 * pe_data_sorted, linestyle='--', color='blue')
# plt.scatter(pe_real, pe_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured Evaporation Pressure [Bar]', fontsize=12)
# plt.ylabel('Predicted Evaporation Pressure [Bar]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# # Plot 6: Gas Cooler Outlet Temperature
# fig6 = plt.figure(6)
# plt.plot(Tw_out_real, Tw_out_real, c='k', label='Ideal line')
# Tw_out_sorted = np.sort(Tw_out_real)
# plt.plot(Tw_out_sorted, [x - 5 for x in Tw_out_sorted], linestyle='--', color='blue', label='+- 5K error')
# plt.plot(Tw_out_sorted, [x + 5 for x in Tw_out_sorted], linestyle='--', color='blue')
# plt.scatter(Tw_out_real, Tw_out_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured Water Outlet Temperature [°C]', fontsize=12)
# plt.ylabel('Predicted Water Outlet Temperature [°C]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# # Plot 7: Evaporator Outlet Temperature
# fig7 = plt.figure(7)
# plt.plot(Tmpg_out_real, Tmpg_out_real, c='k', label='Ideal line')
# Tmpg_out_sorted = np.sort(Tmpg_out_real)
# plt.plot(Tmpg_out_sorted, [x - 5 for x in Tmpg_out_sorted], linestyle='--', color='blue', label='+- 5K error')
# plt.plot(Tmpg_out_sorted, [x + 5 for x in Tmpg_out_sorted], linestyle='--', color='blue')
# plt.scatter(Tmpg_out_real, Tmpg_out_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured MPG Outlet Temperature [°C]', fontsize=12)
# plt.ylabel('Predicted MPG Outlet Temperature [°C]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()


# # Plot 6: Gas Cooler Outlet Temperature
# fig8 = plt.figure(8)
# plt.plot(Tc_out_real, Tc_out_real, c='k', label='Ideal line')
# Tc_out_sorted = np.sort(Tc_out_real)
# plt.plot(Tc_out_sorted, [x - 5 for x in Tc_out_sorted], linestyle='--', color='blue', label='+- 5K error')
# plt.plot(Tc_out_sorted, [x + 5 for x in Tc_out_sorted], linestyle='--', color='blue')
# plt.scatter(Tc_out_real, Tc_out_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured Gas Cooler Outlet Temperature [°C]', fontsize=12)
# plt.ylabel('Predicted Gas Cooler Outlet Temperature [°C]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# # Plot 7: Evaporator Outlet Temperature
# fig9 = plt.figure(9)
# plt.plot(Te_out_real, Te_out_real, c='k', label='Ideal line')
# Te_out_sorted = np.sort(Te_out_real)
# plt.plot(Te_out_sorted, [x - 5 for x in Te_out_sorted], linestyle='--', color='blue', label='+- 5K error')
# plt.plot(Te_out_sorted, [x + 5 for x in Te_out_sorted], linestyle='--', color='blue')
# plt.scatter(Te_out_real, Te_out_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured Evaporator Outlet Temperature [°C]', fontsize=12)
# plt.ylabel('Predicted Evaporator Outlet Temperature [°C]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# # Plot 6: Gas Cooler Outlet Temperature
# fig10 = plt.figure(10)
# plt.plot(mc_dot_real, mc_dot_real, c='k', label='Ideal line')
# mc_dot_sorted = np.sort(mc_dot_real)
# plt.plot(mc_dot_sorted, 0.9* mc_dot_sorted, linestyle='--', color='blue', label='10% error')
# plt.plot(mc_dot_sorted, 1.1 *mc_dot_sorted, linestyle='--', color='blue')
# plt.scatter(mc_dot_real, mc_in_dot_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured Gas Cooler Mass Flow [g/s]', fontsize=12)
# plt.ylabel('Predicted Gas Cooler Mass Flow [g/s]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# fig11 = plt.figure(11)
# plt.plot(me_dot_real, me_dot_real, c='k', label='Ideal line')
# me_dot_sorted = np.sort(me_dot_real)
# plt.plot(me_dot_sorted, 0.9* me_dot_sorted, linestyle='--', color='blue', label='10% error')
# plt.plot(me_dot_sorted, 1.1 *me_dot_sorted, linestyle='--', color='blue')
# plt.scatter(me_dot_real, me_out_dot_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.xlabel('Measured Evaporator Mass Flow [g/s]', fontsize=12)
# plt.ylabel('Predicted Evaporator Mass Flow [g/s]', fontsize=12)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.grid(True)
# plt.legend()

# plt.show()
