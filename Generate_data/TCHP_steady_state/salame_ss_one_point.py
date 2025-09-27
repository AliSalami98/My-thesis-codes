from scipy.integrate import solve_ivp
from model import cycle
from post_processing import post_process
from plot import plot_ph
import numpy as np
import config
import time

from config import CP
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
    d_Prec,
    d_Tc_in,
    d_Tc_out,
    d_Te_out,
    d_Pheat_out,
    d_mc_dot,
    d_me_dot,
    d_COP
)

Pc_av = []
Pe_av = []
Pihx1_av = []
Pihx2_av = []
Pbuff1_av = []
Pbuff2_av = []
Pcooler1_av = []
Pcooler2_av = []
Pcooler3_av = []
Prec_av = []
Pheat_out_av = []
me_out_dot_av = []
mc_in_dot_av = []

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

sample_index = 5
# while sample_index < len(d_Hpev):
print(sample_index)
pe_in = d_pe[sample_index] - 5e5
state = get_state(CP.PQ_INPUTS, d_p25[sample_index], 0)
he_in = state.hmass()
state = get_state(CP.HmassP_INPUTS, he_in, pe_in)
De_in = state.rhomass()
Te_in = state.T()
cp_mpg = 3600


config.pe =  pe_in
for i in range(config.Ne):
    config.Tmpg[i] = d_Tmpg_in[sample_index]
    config.Te_wall[i] = (d_Tmpg_in[sample_index])
    config.he[i] = he_in
    config.De[i] = De_in


mw2_dot = 0.7 * d_mw_dot[sample_index]

mCH4_dot = (0.0022 * d_omegab[sample_index] - 2.5965) * 0.657/60000
LHV = 50e6
Tfume = (d_Pcomb[sample_index] - 142)/19.64 + 273.15
Theater3 = Tfume - 100
mw1_dot_2 = 0.5 * 0.3 * d_mw_dot[sample_index]
mw1_dot_1 = 0.5 * 0.3 * d_mw_dot[sample_index]
Tamb = 273.15

pc_in = d_pc[sample_index] + 5e5
if 73.3e5 < pc_in < 73.8e5:
    pc_in = 73.3e5
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

pft_in = d_p25[sample_index]
hft_in = (he_in + hc_in)/2 #hc_in
state = get_state(CP.HmassP_INPUTS, hft_in, pft_in)
Tft_in = state.T()
Dft_in = state.rhomass()

config.pft =  pft_in
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
    config.pbuff1, config.hbuff1, config.Tbuff1_wall,
    config.pft, config.hft,
    d_Lpev[sample_index], d_Hpev[sample_index]])

Tc_out_ss = []
hc_out_ss = []
pe_ss = []
Te_out_ss = []
he_out_ss = []
hTC2_in_ss = []
hbuff1_in_ss = []
hc_in_ss = []
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
dxdt_ss = []


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
dt = 1

error = 10e6
current_time_step = 0
# while error > 10000:
for current_time_step in range(total_time_steps):
    current_time = current_time_step * dt
    sol = solve_ivp(cycle, config.t0, X0, method = 'RK23', max_step=0.5, args = (
        mw_dot,
        Tw_in,
        Tmpg_in,
        mmpg_dot,
        Theater1,
        Theater2,
        omega1,
        omega2,
        omega3,
        Lpev,
        Hpev,
        omegab,
        tau_LPV,
        tau_HPV
        )
    )

    dxdt = cycle(
        1, X0,
        mw_dot, Tw_in, Tmpg_in, mmpg_dot,
        Theater1, Theater2, omega1, omega2, omega3,
        Lpev, Hpev, omegab, tau_LPV, tau_HPV
    )

    dxdt_ss.append(dxdt)

    # Use vector norm or max absolute value for scalar error
    error = np.max(np.abs(dxdt))  # or np.linalg.norm(dxdt)
    # print(error)

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
        Pcooler1_pred,
        Pcooler23_pred,
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
        a_Hpev,
        a_hc_in,
        a_hTC2_in,
        a_hbuff1_in
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
        omega2,
        omega3,
        Lpev,
        Hpev,
        omegab,
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
    hc_in_ss.append(np.mean(a_hc_in))
    hTC2_in_ss.append(np.mean(a_hTC2_in))
    hbuff1_in_ss.append(np.mean(a_hbuff1_in))
    current_time_step += 1

import matplotlib.pyplot as plt
# Convert dxdt_ss to array for processing
dxdt_array = np.array(dxdt_ss)  # shape: (time_steps, len(X0))
dxdt_norms = np.linalg.norm(dxdt_array, axis=1)  # L2 norm for each time step

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t_ss, dxdt_norms, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Time [s]', fontsize=14)
plt.ylabel(r'$||\dot{x}||$ [units of x/s]', fontsize=14)
plt.title('Norm of $\dot{x}$ vs. Time', fontsize=15)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# Pbuff1_av.append(np.mean(Pbuff1_ss[-10:]))
# Pcooler1_av.append(np.mean(Pcooler1_ss[-10:]))
# Pcooler3_av.append(np.mean(Pcooler3_ss[-10:]))
# Prec_av.append(np.mean(Prec_ss[-10:]))
# Tc_in_av.append(np.mean(Tc_in_ss[-10:]))

Pc_av.append(np.mean(Pgc_ss[-10:]))
Pe_av.append(np.mean(Pevap_ss[-10:]))
pc_av.append(np.mean(pc_ss[-10:]))
pe_av.append(np.mean(pe_ss[-10:]))
Tc_out_av.append(np.mean(Tc_out_ss[-10:]))
Te_out_av.append(np.mean(Te_out_ss[-10:]))
Tw_out_av.append(np.mean(Tw_out_ss[-10:]))
Tmpg_out_av.append(np.mean(Tmpg_out_ss[-10:]))
hc_out_av.append(np.mean(hc_out_ss[-10:]) * 1e-3)
he_out_av.append(np.mean(he_out_ss[-10:]) * 1e-3)
COP_av.append(np.mean(COP_ss[-10:]))
Pheat_out_av.append(np.mean(Pheat_out_ss[-10:]))
mc_in_dot_av.append(np.mean(mc_in_dot_ss[-10:]) * 1e3)
me_out_dot_av.append(np.mean(me_out_dot_ss[-10:]) * 1e3)
pft_av = np.mean(pft_ss[-10:])
pbuff1_av = np.mean(pbuff1_ss[-10:])
hihx1_out_av = np.mean(hihx1_out_ss[-10:]) * 1e-3
hihx2_out_av = np.mean(hihx2_out_ss[-10:]) * 1e-3
hft_av = np.mean(hft_ss[-10:]) * 1e-3
state = get_state(CP.PQ_INPUTS, pbuff1_in, 0)
hft_l_av = state.hmass() * 1e-3
hTC2_in_av = np.mean(hTC2_in_ss[-10:]) * 1e-3
hc_in_av = np.mean(hc_in_ss[-10:]) * 1e-3
hbuff1_in_av = np.mean(hbuff1_in_ss[-10:]) * 1e-3

pressure_sim = [pft_av, pbuff1_av, pc_av[-1], pc_av[-1], pc_av[-1], pft_av,
                 pft_av, pe_av[-1], pe_av[-1], pe_av[-1], pbuff1_av, pbuff1_av]        

enthalpy_sim = [hft_av, hTC2_in_av, hc_in_av, hc_out_av[-1], hihx1_out_av, hft_av,
                 hft_l_av, hft_l_av, he_out_av[0], hihx2_out_av, hbuff1_in_av, hTC2_in_av]
# RK23, DOP853, Radau, BDF, LSODA
end_time = time.time()

# Calculate and print the time taken
execution_time = end_time - start_time
print(f"Time taken by solve_ivp: {execution_time:.4f} seconds")

plot_ph(
    pressure_sim,
    enthalpy_sim,
    sample_index

)