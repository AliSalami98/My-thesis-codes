from scipy.integrate import solve_ivp
from model import cycle
from post_processing import post_process
# from plot import plot_and_print
import numpy as np
import config
import time

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
from data_filling_tr import (
    d_omegab,
    d_omega1,
    d_omega2,
    d_omega3,
    d_Theater1,
    d_Theater2,
    d_mw_dot,
    d_mmpg_dot,
    d_Tw_in,
    d_Tmpg_in,
    d_Lpev,
    d_Hpev,
    d_pc,
    d_p25,
    d_pi,
    d_pe,
    d_Pcomb,
    counter_threshold
)
from algebraic_model import compute_cycle_inputs

file_name = 'BF2.csv'

pe_in = d_pe[0] - 2e5
state = get_state(CP.PQ_INPUTS, d_p25[0], 0)
he_in = state.hmass()
state = get_state(CP.HmassP_INPUTS, he_in, pe_in)
De_in = state.rhomass()
Te_in = state.T()

config.pe =  pe_in
for i in range(config.Ne):
    config.Tmpg[i] = d_Tmpg_in[0]
    config.Te_wall[i] = (d_Tmpg_in[0])
    config.he[i] = he_in
    config.De[i] = De_in


mw2_dot = 0.7 * d_mw_dot[0]
mw1_dot_2 = 0.5 * 0.3 * d_mw_dot[0]
mw1_dot_1 = 0.5 * 0.3 * d_mw_dot[0]

pc_in = d_pc[0] + 2e5
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

Tr1 = d_Theater1[0]/d_Tw_in[0]
Pr1 = d_p25[0]/config.pe
# a1 = np.array([[Pr1], [Tr1], [omega1[0]]])
a1 = np.array([[config.pe], [d_p25[0]], [d_omega1[0]], [d_Theater1[0]], [d_Tw_in[0]]])
a1_T = a1.T

pbuff1_in = d_p25[0]
# hbuff1_in = he_in
Tbuff1_in = 320 # model2_PR.predict(poly2_reg.transform(a1_T))[0]
state = get_state(CP.PT_INPUTS, pbuff1_in, Tbuff1_in)
hbuff1_in = state.hmass()
Dbuff1_in = state.rhomass()
Tbuff1_w_in = d_Tw_in[0]

config.Tbuff1_w = d_Tw_in[0]
config.Tbuff1_wall = (d_Tw_in[0])
config.pbuff1 =  pbuff1_in
config.hbuff1 = hbuff1_in
config.Dbuff1 = Dbuff1_in


pbuff2_in = d_pi[0]
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

pft_in = d_p25[0]
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

mc_in_dot = 10**(-3) * (5.14e-3 * d_omegab[0] - 2.14e1 * pc_in/d_p25[0] - 1.11e-3 * d_omega2[0] + 23.38)
mc_out_dot = 8 * 10**(-9) * d_Hpev[0] * np.sqrt(2*config.Dc[-1] * max(config.pc - config.pft, 0))
mft_v_dot = 8 * 10**(-7) *np.sqrt(2*config.Dft * max(config.pft - config.pbuff1, 0))
mft_l_dot = 4 * 10**(-9) * d_Lpev[0] * np.sqrt(2*config.Dft * max(config.pft - config.pe, 0)) #0.000145 * Lpev  #
me_out_dot = 10**(-3) * (3.843e-3 * d_omegab[0] - 3.99e+1 * Pr1 + 2.18e-2 * d_omega1[0] + 42.36)
me_in_dot = mft_l_dot
mbuff2_in_dot = mc_in_dot

tauc_in = 1
tauc_out = 100
taue_in = 40
taue_out = 1
tauft_v = 20
tauft_l = 20
taubuff2_in = 1
tau_HPV = 20/counter_threshold
tau_LPV = 10/counter_threshold

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
Prec_total_ss = []
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
a_Hpev = []

start_time = time.time()

dt = 1

lpev = d_Lpev[0]
hpev = d_Hpev[0]
Lpev_prev = 0
for current_time_step in range(total_time_steps):
    current_time = current_time_step * dt
    print(current_time)
    mw_dot = d_mw_dot[current_time]
    Tw_in = d_Tw_in[current_time]
    Tmpg_in = d_Tmpg_in[current_time]
    mmpg_dot = d_mmpg_dot[current_time]
    Theater1 = d_Theater1[current_time]
    Theater2 = d_Theater2[current_time]
    omega1 = d_omega1[current_time]
    omega2 = d_omega2[current_time]
    omega3 = d_omega3[current_time]
    Lpev = d_Lpev[current_time]
    Hpev = d_Hpev[current_time]
    omegab = d_omegab[current_time]

    # if Lpev > Lpev_prev:
    #     tau_LPV = 100
    lpev = lpev + dt/tau_LPV * (Lpev - lpev)
    hpev = hpev + dt/tau_HPV * (Hpev - hpev)

    Lpev_prev = Lpev
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
    pbuff2_ss.append(np.mean(a_pbuff2))
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
    Prec_total_ss.append(np.mean(a_Prec_total))
    COP_ss.append(np.mean(a_COP))
    SH_ss.append(np.mean(a_SH))


# RK23, DOP853, Radau, BDF, LSODA
end_time = time.time()

# Calculate and print the time taken
execution_time = end_time - start_time
print(f"Time taken by solve_ivp: {execution_time:.4f} seconds")

import pandas as pd
import os

# Prepare the DataFrame from your collected lists
df = pd.DataFrame({
    "t_ss": t_ss,
    "Tmpg_out_ss": Tmpg_out_ss,
    "Tw_out_ss": Tw_out_ss,
    "pc_ss": pc_ss,
    "hc_out_ss": hc_out_ss,
    "hihx1_out_ss": hihx1_out_ss,
    "hft_ss": hft_ss,
    "hihx2_out_ss": hihx2_out_ss,
    "he_out_ss": he_out_ss,
    "pft_ss": pft_ss,
    "pbuff1_ss": pbuff1_ss,
    "pbuff2_ss": pbuff2_ss,
    "pe_ss": pe_ss,
    "Tc_out_ss": Tc_out_ss,
    "Te_out_ss": Te_out_ss,
    "mc_in_dot_ss": mc_in_dot_ss,
    "mc_out_dot_ss": mc_out_dot_ss,
    "mft_in_dot_ss": mft_in_dot_ss,
    "mft_out_dot_ss": mft_out_dot_ss,
    "mbuff1_in_dot_ss": mbuff1_in_dot_ss,
    "mbuff1_out_dot_ss": mbuff1_out_dot_ss,
    "me_in_dot_ss": me_in_dot_ss,
    "me_out_dot_ss": me_out_dot_ss,
    "Pheat_out_ss": Pheat_out_ss,
    "Pgc_ss": Pgc_ss,
    "Pevap_ss": Pevap_ss,
    "Prec_total_ss": Prec_total_ss,
    "COP_ss": COP_ss,
    "SH_ss": SH_ss
})

# Save in the same folder as your script
output_path = os.path.join(os.path.dirname(__file__), file_name)
df.to_csv(output_path, index=False)

print(f"CSV file saved to: {output_path}")


# plot_and_print(
#     y,
#     t,
#     t_ss,
#     Pgc_ss,
#     Pevap_ss,
#     Pihx1_ss,
#     Pihx2_ss,
#     Pbuff1_ss,
#     Pbuff2_ss,
#     Pcooler1_pred,
#     Pcooler23_pred,
#     Pfhx,
#     me_dot_ss,
#     mihx2_dot_ss,
#     mbuff1_dot_ss,
#     mbuff2_dot_ss,
#     mc_dot_ss,
#     mihx1_dot_ss,
#     COP_ss,
#     Pheat_out_ss,
#     Pcoolers_ss,
#     Pbuffers_ss,
#     mc_in_dot_ss,
#     mbuff1_in_dot_ss,
#     pft_ss,
#     pc_ss,
#     pe_ss,
#     Tc_out_ss,
#     Te_out_ss,
#     hc_out_ss,
#     he_out_ss,
#     mc_out_dot_ss,
#     mft_in_dot_ss,
#     mft_out_dot_ss,
#     me_in_dot_ss,
#     me_out_dot_ss,
#     hft_ss,
#     mbuff1_out_dot_ss,
#     pbuff1_ss,
#     pbuff2_ss,
#     mft_dot_ss,
#     hihx1_out_ss,
#     hihx2_out_ss,
#     Tw_out_ss,
#     Tmpg_out_ss,
#     SH_ss,
#     total_time_steps,
#     Prec_total_ss
# )
