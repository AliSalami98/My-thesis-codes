# import numpy as np
import time 
import cProfile
import pstats
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
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
    T0,
    h0,
    s0

)

sample_index = 0

from data_filling_ss import data, a_Pr
Pheat_real = np.array(data['Pheating [W]'][sample_index])
Pcool_real = np.array(data['Pcooling [W]'][sample_index])
Pmech_real = np.array(data['Pmech [W]'][sample_index])
Pout_real = np.array(data['Pcomp [W]'][sample_index])
Tdis_real = np.array(data['Tdis [K]'][sample_index]) - 273.15
mdot_real = np.array(data['mdot [g/s]'][sample_index])
Ex_eff_real = np.array(data['Ex_eff [%]'][sample_index])

import time

start_time = time.time()  # Start timing

Tin = data['Tsuc [K]'][sample_index]
pin = data['psuc [pa]'][sample_index] - 0.5e5
pout = data['pdis [pa]'][sample_index] + 0.5e5
state.update(CP.PT_INPUTS, pin, Tin)
hin= state.hmass()
sin= state.smass()
Din = state.rhomass()
Th_wall_ext = data['Theater [K]'][sample_index]
Tw_in = data['Tw_in [K]'][sample_index]
omega= data['omega [rpm]'][sample_index]

Th_wall = Th_wall_ext
Tk_wall_ext = Tw_in
Tk_wall = Tk_wall_ext

pc = pin + 0.5e5
pe = pc - 0.1*10**5
N = nk + nkr + nreg + nhr + nh +2
Twall= N*[0]
Twall_ext= N*[0]
Dn = N*[0]
Tn = N*[0]
mn = N*[0]
v = (N-1)*[0.01]
pn = N*[0]
for i in range(N):
    if i < nk + nkr + 1:
        Twall[i] = Tk_wall
    elif nk + nkr + nreg + 1 <= i < N:
        Twall[i] = Th_wall
    else:
        Twall[i] = (Th_wall - Tk_wall)/(nreg + 1) * (i - (nk + nkr)) + Tk_wall

    Tn[i] = Twall[i]
    pn[i] = (pe-pc)/(N-1) * (i) + pc
    state.update(CP.PT_INPUTS, pn[i], Tn[i])
    Dn[i] = state.rhomass()
    mn[i] = Dn[i]*Vn[i]

# for i in range(N):
#     if i < nk + 1:
#         Twall[i] = Tk_wall
#         Tn[i] = Tin
#     elif nk + nkr + nreg + nhr + 1 <= i < N:
#         Twall[i] = Th_wall
#         Tn[i] = Th_wall
#     else:
#         Twall[i] = (Th_wall - Tk_wall)/(nkr + nreg + nhr + 1) * (i - (nk)) + Tk_wall
#         Tn[i] = (Th_wall - Tin)/(nkr + nreg + nhr + 1) * (i - (nk)) + Tin

#     pn[i] = (pe-pc)/(N-1) * (i) + pc
#     state.update(CP.PT_INPUTS, pn[i], Tn[i])
#     Dn[i] = state.rhomass()
#     mn[i] = Dn[i]*Vn[i]    
mint_dot = 0
mout_dot = 0

X0 = mn + Tn + v + [mint_dot, mout_dot] + Twall

pc_mean = pc/1e5

n_cycles = 0
mdot_error = 100
T_error = 100
alpha_error = 1000
everything = 'ayri'
Q_array_sum = 1000
pinit = 0
pfinal = 0
p_error = 10

while (n_cycles < 4 or np.abs(alpha_error) > 200 or np.abs(Q_array_sum) > 200) and n_cycles < 10:
    c = [0]
    d = [0]
    sol = solve_ivp(model,theta0, X0, method = 'RK23', args = (pin, pout, sin, Din, hin, omega, Twall_ext, Tw_in, c, d))
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
        a_Edest_c,
        a_Edest_k,
        a_Edest_kr,
        a_Edest_r,
        a_Edest_hr,
        a_Edest_h,
        a_Edest_e,
        a_Ex_eff,
        Q_array
    ) = post_process(y, theta, sin,  pin, pout, Din, hin, omega)
    mdot_error = (np.mean(a_mint_dot) - np.mean(a_mout_dot))
    alpha_error = np.mean(a_alpha)
    row_means = np.mean(Q_array, axis=0)

    Q_array_sum = np.sum(row_means)


    for i in range(nk + nkr + 1, nk + nkr + nreg + 1):
        X0[3 * N + 1 + i] -= 1e-2 * row_means[i - (nk + nkr + 1)] #Q_array[i - (nk + nkr + 1), -1] #

    # # for i in range(N):
    # if np.abs(mdot_error) > 0.1:
    #     X0[0] += 5e-6 * mdot_error

    pinit = a_pc[0]
    pfinal = a_pc[-1]
    p_error = pfinal - pinit

    # print(f"perror = {p_error:.6f}")
    # print(f"alpha_error = {alpha_error:.2f}")
    # print(f"mdot_error = {mdot_error:.4f}")
    # print(f"Q_array_sum = {Q_array_sum:.2f}")
    n_cycles += 1

    W_ss = - np.mean(a_W)
    mdot_ss = np.mean(a_mdot) * 1e3
    # hout_ss = np.mean(Hout_dot)/np.mean(a_mout_dot)
    # state.update(CP.HmassP_INPUTS, hout_ss, pout)
    # Tout2_ss = state.T()
    Tout_ss = np.mean(a_Tout) - 273.15
    Pheat_ss = max(np.mean(a_Qh),0) + max(np.mean(a_Qhr),0) + max(np.mean(a_Qr),0) + max(np.mean(a_Qkr), 0) + max(np.mean(a_Qk), 0)
    Pcool_ss = -(min(np.mean(a_Qh),0) + min(np.mean(a_Qhr),0) + min(np.mean(a_Qr),0) + min(np.mean(a_Qkr), 0) + min(np.mean(a_Qk), 0))
    alpha_ss = np.mean(a_alpha)
    Pout_ss = np.mean(a_Pout)
    eff_ss = np.mean(a_Pout)/(Pheat_ss - np.mean(a_W))

    state.update(CP.HmassP_INPUTS, np.mean(a_hout), pout)
    sout = state.smass()
    Edest_c_ss = np.mean(a_Edest_c) + mdot_ss * (hin - np.mean(a_hout) - T0 * (sin - sout))
    Edest_k_ss = np.mean(a_Edest_k)
    Edest_kr_ss = np.mean(a_Edest_kr)
    Edest_r_ss = np.mean(a_Edest_r)
    Edest_hr_ss = np.mean(a_Edest_hr)
    Edest_h_ss = np.mean(a_Edest_h)
    Edest_e_ss = np.mean(a_Edest_e)
    alpha_ss = np.mean(a_alpha)
    Ex_eff_ss = (1 - (Edest_c_ss + Edest_k_ss + Edest_kr_ss + Edest_r_ss + Edest_hr_ss + Edest_h_ss + Edest_e_ss)/(W_ss + Pheat_ss * (1 - T0/Th_wall))) #np.mean(a_Ex_eff)

    # Relative deviations (single values, not lists) + print real vs simulated
    print(f"Pheating: Sim = {Pheat_ss:.3f}, Real = {Pheat_real:.3f}, "
        f"MAPE = {100*np.abs(Pheat_ss-Pheat_real)/np.abs(Pheat_real):.2f}%")

    print(f"Pcooling: Sim = {Pcool_ss:.3f}, Real = {Pcool_real:.3f}, "
        f"MAPE = {100*np.abs(Pcool_ss-Pcool_real)/np.abs(Pcool_real):.2f}%")

    print(f"Pmech:    Sim = {W_ss:.3f}, Real = {Pmech_real:.3f}, "
        f"MAPE = {100*np.abs(W_ss-Pmech_real)/np.abs(Pmech_real):.2f}%")

    print(f"Tout [°C]: Sim = {Tout_ss:.2f}, Real = {Tdis_real:.2f}, "
        f"MAE = {np.abs(Tout_ss-Tdis_real):.2f}")

    print(f"mdot:     Sim = {mdot_ss:.3f}, Real = {mdot_real:.3f}, "
        f"MAPE = {100*np.abs(mdot_ss-mdot_real)/np.abs(mdot_real):.2f}%")

    print(f"Pout:     Sim = {Pout_ss:.3f}, Real = {Pout_real:.3f}, "
        f"MAPE = {100*np.abs(Pout_ss-Pout_real)/np.abs(Pout_real):.2f}%")

    print(f"Ex_eff:   Sim = {Ex_eff_ss:.3f}, Real = {Ex_eff_real:.3f}, "
        f"MAPE = {100*np.abs(Ex_eff_ss-Ex_eff_real)/np.abs(Ex_eff_real):.2f}%")


# # Relative deviations (single values, not lists)
# MAPE_Pheat = 100 * np.mean(np.abs(np.array(Pheat_ss) - Pheat_real) / np.abs(Pheat_real))
# print('MAPE of Pheating [%]', MAPE_Pheat)

# MAPE_Pcool = 100 * np.mean(np.abs(np.array(Pcool_ss) - Pcool_real) / np.abs(Pcool_real))
# print('MAPE of Pcooling [%]', MAPE_Pcool)

# MAPE_Pmech = 100 * np.mean(np.abs(np.array(W_ss) - Pmech_real) / np.abs(Pmech_real))
# print('MAPE of Pmech [%]', MAPE_Pmech)

# MAE_Tout = np.mean(np.abs(np.array(Tout_ss) - Tdis_real))
# print('MAE of Tout [°C]', MAE_Tout)

# MAPE_mdot = 100 * np.mean(np.abs(np.array(mdot_ss) - mdot_real) / np.abs(mdot_real))
# print('MAPE of mdot [%]', MAPE_mdot)

# MAPE_Pout = 100 * np.mean(np.abs(np.array(Pout_ss) - Pout_real) / np.abs(Pout_real))
# print('MAPE of Pout [%]', MAPE_Pout)

# MAPE_Ex_eff = 100 * np.mean(np.abs(np.array(Ex_eff_ss) - Ex_eff_real) / np.abs(Ex_eff_real))
# print('MAPE of Ex_eff [%]', MAPE_Ex_eff)
