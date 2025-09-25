# import numpy as np
import time 
import cProfile
import pstats
import numpy as np
from scipy.integrate import solve_ivp
from model import model
from post_processing import post_process
from plot import plot_and_print
from config import( 
    theta0,
    nk, 
    nkr, 
    nreg, 
    nhr, 
    nh,
    Vn
)
from utils import (
	CP,
	state,

)
# profiler = cProfile.Profile()
# profiler.enable()
# Code block or function call you want to profile
from data_filling_ss import data
k = 8
Tin = data['Tin [K]'][k]
pin = data['pin [pa]'][k]
pout = data['pout [pa]'][k]
state.update(CP.PT_INPUTS, pin, Tin)
hin= state.hmass()
sin= state.smass()
Din = state.rhomass()
Th_wall_ext = data['Th_wall [K]'][k]
Tw_in = data['Tw_in [K]'][k]
omega= data['omega [rpm]'][k]

Th_wall = Th_wall_ext
Tk_wall_ext = Tw_in
Tk_wall = Tk_wall_ext
# print(pout/pin)
pc = pin
pe = pc - 0.1*10**5
N = nk + nkr + nreg + nhr + nh +2
Twall= N*[0]
Twall_ext= N*[0]
Dn = N*[0]
mn = N*[0]
Tn = N*[0]
v = (N-1)*[0.01]
pn = N*[0]
for i in range(N-1):
    if i < nk + 1:
        Twall[i] = Tk_wall
        Tn[i] = Tin
    else:
        Twall[i] = (Th_wall - Tk_wall)/(N-3) * (i-1) + Tk_wall
        Tn[i] = (Th_wall - Tin)/(N-3) * (i-1) + Tin
    pn[i] = (pe-pc)/(N-1) * (i) + pc
    state.update(CP.PT_INPUTS, pn[i], Tn[i])
    Dn[i] = state.rhomass()
    mn[i] = Dn[i]*Vn[i]
    
mint_dot = 0
mout_dot = 0
# Tn[0] = Tk_wall
# Twall[0] = Tk_wall
# pn[0] = pc
# state.update(CP.PT_INPUTS, pn[0], Tn[0])
# Dn[0] = state.rhomass()
# mn[0] = Dn[0]*Vn[0]

Tn[-1] = Th_wall
Twall[-1] = Th_wall
pn[-1] = pe
state.update(CP.PT_INPUTS, pn[-1], Tn[-1])
Dn[-1] = state.rhomass()
mn[-1] = Dn[-1]*Vn[-1]
# print(Tn)
# print(Twall)
X0 = mn + Tn + v + [mint_dot, mout_dot]

theta_eval = np.arange(theta0[0], theta0[-1], 0.01)

n_cycles = 0

while n_cycles < 1: 
    
    sol = solve_ivp(model,theta0, X0, method = 'RK23', args = (pin, pout, sin, Din, hin, omega, Twall, Twall_ext, Tw_in))
    #RK45, RK23, DOP853, Radau, BDF, LSODA
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('time')
    # stats.print_stats()

    y = sol.y
    theta = sol.t
    X0 = y[:, -1]
    ##########################################################################################################

    (
        a_Pout,
        a_mck_dot,
        a_meh_dot,
        a_mkr_dot,
        a_mrh_dot,
        a_mint_dot,
        a_mout_dot,
        a_alpha,
        a_W,
        a_Deltap,
        a_Deltapk,
        a_Deltapkr,
        a_Deltapr,
        a_Deltaphr,
        a_Deltaph,
        a_mdot,
        a_Te,
        a_Th,
        a_Tr,
        a_Tc,
        a_Tk,
        a_Tk_wall,
        a_Tr_wall,
        a_Th_wall,
        a_Tout,
        a_hout,
        a_Vc,
        a_Ve,
        a_pc,
        a_pk,
        a_pr,
        a_ph,
        a_pe,
        a_theta,
        a_Dc,
        a_Dk,
        a_Dkr,
        a_Dr,
        a_Dhr,
        a_Dh,
        a_De,
        a_mc,
        a_mk,
        a_mkr,
        a_mr,
        a_mhr,
        a_mh,
        a_me,
        a_vk,
        a_vkr,
        a_vr,
        a_vhr,
        a_vh,
        a_Qc,
        a_Qk,
        a_Qkr,
        a_Qr,
        a_Qhr,
        a_Qh,
        a_Qe,
    ) = post_process(y, theta, sin,  pin, pout, Din, hin, omega, Twall)
    n_cycles += 1

Pheat_ss = max(np.mean(a_Qh),0) + max(np.mean(a_Qhr),0) + max(np.mean(a_Qr),0) + max(np.mean(a_Qkr), 0) + max(np.mean(a_Qk), 0)
Pcool_ss = min(np.mean(a_Qh),0) + min(np.mean(a_Qhr),0) + min(np.mean(a_Qr),0) + min(np.mean(a_Qkr), 0) + min(np.mean(a_Qk), 0) + min(np.mean(a_Qc), 0)
eff_ss = np.mean(a_Pout)/(Pheat_ss - np.mean(a_W))
# Pout_ss = a_Pout
# mck_dot_ss = a_mck_dot
# meh_dot_ss = a_meh_dot
# mkr_dot_ss = a_mkr_dot
# mrh_dot_ss = a_mrh_dot
# mint_dot_ss = a_mint_dot
# mout_dot_ss = a_mout_dot
# alpha_ss = a_alpha
# W_ss = a_W
# Deltap_ss = a_Deltap
# Deltapk_ss = a_Deltapk
# Deltapkr_ss = a_Deltapkr
# Deltapr_ss = a_Deltapr
# Deltaphr_ss = a_Deltaphr
# Deltaph_ss = a_Deltaph
# mdot_ss = a_mdot
# Te_ss = a_Te
# Th_ss = a_Th
# Tr_ss = a_Tr
# Tc_ss = a_Tc
# Tk_ss = a_Tk
# Tk_wall_ss = a_Tk_wall
# Tr_wall_ss = a_Tr_wall
# Th_wall_ss = a_Th_wall
# Tout_ss = a_Tout
# hout_ss = a_hout
# Vc_ss = a_Vc
# Ve_ss = a_Ve
# pc_ss = a_pc
# pk_ss = a_pk
# pr_ss = a_pr
# ph_ss = a_ph
# pe_ss = a_pe
# theta_ss = a_theta
# Dc_ss = a_Dc
# Dk_ss = a_Dk
# Dkr_ss = a_Dkr
# Dr_ss = a_Dr
# Dhr_ss = a_Dhr
# Dh_ss = a_Dh
# De_ss = a_De
# pc_ss = a_pc
# pk_ss = a_pk
# ph_ss = a_ph
# pe_ss = a_pe
# mc_ss = a_mc
# mk_ss = a_mk
# mkr_ss = a_mkr
# mr_ss = a_mr
# mhr_ss = a_mhr
# mh_ss = a_mh
# me_ss = a_me
# vk_ss = a_vk
# vkr_ss = a_vkr
# vr_ss = a_vr
# vhr_ss = a_vhr
# vh_ss = a_vh
# Qc_ss = a_Qc
# Qk_ss = a_Qk
# Qkr_ss = a_Qkr
# Qr_ss = a_Qr
# Qhr_ss = a_Qhr
# Qh_ss = a_Qh
# Qe_ss = a_Qe

plot_and_print(
    y,
    theta,
    hin,
    a_Pout,
    a_mck_dot,
    a_meh_dot,
    a_mkr_dot,
    a_mrh_dot,
    a_mint_dot,
    a_mout_dot,
    a_W,
    a_Deltap,
    a_Deltapk,
    a_Deltapkr,
    a_Deltapr,
    a_Deltaphr,
    a_Deltaph,
    a_mdot,
    a_Te,
    a_Th,
    a_Tr,
    a_Tc,
    a_Tk,
    a_Tk_wall,
    a_Tr_wall,
    a_Th_wall,
    a_Vc,
    a_Ve,
    a_pc,
    a_pk,
    a_pr,
    a_ph,
    a_pe,
    a_Tout,
    a_hout,
    a_theta,
    a_Dc,
    a_Dk,
    a_Dkr,
    a_Dr,
    a_Dhr,
    a_Dh,
    a_De,
    a_mc,
    a_mk,
    a_mkr,
    a_mr,
    a_mhr,
    a_mh,
    a_me,
    a_vk,
    a_vkr,
    a_vr,
    a_vhr,
    a_vh,
    a_Qc,
    a_Qk,
    a_Qkr,
    a_Qr,
    a_Qhr,
    a_Qh,
    a_Qe,
    a_alpha,
    Pheat_ss,
    Pcool_ss,
    eff_ss,
    k
)
