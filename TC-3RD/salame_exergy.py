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
# profiler = cProfile.Profile()
# profiler.enable()
# Code block or function call you want to profile
from data_Theater import data, a_Pr

import time

def multi_solving(k):
    start_time = time.time()  # Start timing

    Tin = data['Tsuc [K]'][k]
    pin = data['psuc [pa]'][k] - 0.5e5
    pout = data['pdis [pa]'][k] + 0.5e5
    state.update(CP.PT_INPUTS, pin, Tin)
    hin= state.hmass()
    sin= state.smass()
    Din = state.rhomass()
    Th_wall_ext = data['Theater [K]'][k]
    Tw_in = data['Tw_in [K]'][k]
    omega= data['omega [rpm]'][k]

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
    # for i in range(N):
    #     if i < nk + nkr + 1:
    #         Twall[i] = Tk_wall
    #         Tn[i] = Tin
    #     elif nk + nkr + nreg + 1 <= i < N:
    #         Twall[i] = Th_wall
    #         Tn[i] = Th_wall
    #     else:
    #         Twall[i] = (Th_wall - Tk_wall)/(nkr + nreg + 1) * (i - (nk + nkr)) + Tk_wall
    #         Tn[i] = (Th_wall - Tin)/(nkr + nreg + 1) * (i - (nk + nkr)) + Tin

    #     pn[i] = (pe-pc)/(N-1) * (i) + pc
    #     state.update(CP.PT_INPUTS, pn[i], Tn[i])
    #     Dn[i] = state.rhomass()
    #     mn[i] = Dn[i]*Vn[i]

    for i in range(N):
        if i < nk + 1:
            Twall[i] = Tk_wall
            Tn[i] = Tin
        elif nk + nkr + nreg + nhr + 1 <= i < N:
            Twall[i] = Th_wall
            Tn[i] = Th_wall
        else:
            Twall[i] = (Th_wall - Tk_wall)/(nkr + nreg + nhr + 1) * (i - (nk)) + Tk_wall
            Tn[i] = (Th_wall - Tin)/(nkr + nreg + nhr + 1) * (i - (nk)) + Tin

        pn[i] = (pe-pc)/(N-1) * (i) + pc
        state.update(CP.PT_INPUTS, pn[i], Tn[i])
        Dn[i] = state.rhomass()
        mn[i] = Dn[i]*Vn[i]    
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
    mdot_ss = np.mean(a_mdot)
    # hout_ss = np.mean(Hout_dot)/np.mean(a_mout_dot)
    # state.update(CP.HmassP_INPUTS, hout_ss, pout)
    # Tout2_ss = state.T()
    Tout_ss = np.mean(a_Tout)
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

    # print([Edest_comp_ss, Edest_k_ss, Edest_kr_ss, Edest_r_ss, Edest_hr_ss, Edest_h_ss])
    return [Pheat_ss, Pcool_ss, W_ss, mdot_ss, Tout_ss, Pout_ss, Edest_c_ss, Edest_k_ss, Edest_kr_ss, Edest_r_ss, Edest_hr_ss, Edest_h_ss, Edest_e_ss, Ex_eff_ss, alpha_ss]

# Function to run in parallel
def run_in_parallel(inputs):
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(multi_solving, inputs))
    return results

# List of tuples, each containing a different set of parameters and initial conditions
inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

import pandas as pd
import os
import csv
from concurrent.futures import ProcessPoolExecutor
# Save directory
save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\Exergy'
os.makedirs(save_dir, exist_ok=True)
csv_path = os.path.join(save_dir, "Results_Theater.csv")

# Define header for CSV
header = ['Pr', 'Pheat [kW]', 'Pcool [kW]', 'W [kW]', 'mdot [kg/s]', 'Tout [C]', 'Pout [bar]',
          'Edest_c [kW]', 'Edest_k [kW]', 'Edest_kr [kW]', 'Edest_r [kW]', 'Edest_hr [kW]',
          'Edest_h [kW]', 'Edest_e [kW]', 'Ex_eff [%]', 'alpha']

# Create and write header to CSV before starting
with open(csv_path, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

# Your input cases
inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

os.makedirs(save_dir, exist_ok=True)
def run_in_parallel(inputs):
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = executor.map(multi_solving, inputs)
        for i, sol in enumerate(results):
            print(f"Processed case {i}")
            row = {
                'Pr': a_Pr[i%5],
                'Pheat [kW]': sol[0] / 1e3,
                'Pcool [kW]': sol[1] / 1e3,
                'W [kW]': sol[2] / 1e3,
                'mdot [kg/s]': sol[3],
                'Tout [C]': sol[4] - 273.15,
                'Pout [bar]': sol[5] / 1e5,
                'Edest_c [kW]': sol[6] / 1e3,
                'Edest_k [kW]': sol[7] / 1e3,
                'Edest_kr [kW]': sol[8] / 1e3,
                'Edest_r [kW]': sol[9] / 1e3,
                'Edest_hr [kW]': sol[10] / 1e3,
                'Edest_h [kW]': sol[11] / 1e3,
                'Edest_e [kW]': sol[12] / 1e3,
                'Ex_eff [%]': sol[13] * 100,
                'alpha': sol[14]
            }

            # Append row to CSV
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writerow(row)

if __name__ == '__main__':
    run_in_parallel(inputs)
    print(f"Saved exergy destruction results iteratively to {csv_path}")
    # fig1 = plt.figure(1)
    # plt.scatter(Pr[0:5], Edest_comp_ss[0:5], c = 'g', s = 50, marker = 'x', label = '30 bar sim')
    # plt.plot(Pr[0:5], Edest_comp_ss[0:5], linestyle=':', c = 'g')
    # plt.scatter(Pr[5:10], Edest_comp_ss[5:10], c = 'b', s = 50, marker = 'x', label = '40 bar sim')
    # plt.plot(Pr[5:10], Edest_comp_ss[5:10], c = 'b', linestyle='--')
    # plt.scatter(Pr[10:15], Edest_comp_ss[10:15], c = 'r', s = 50, marker = 'x', label = '56 bar sim')
    # plt.plot(Pr[10:15], Edest_comp_ss[10:15], c = 'r')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('Edest_comp [kW]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "Qheater.eps"), format='eps')

    # fig2 = plt.figure(2)
    # plt.scatter(Pr[0:5], Edest_k_ss[0:5], c = 'g', s = 50, marker = 'x')
    # plt.plot(Pr[0:5], Edest_k_ss[0:5],linestyle=':', c = 'g')
    # plt.scatter(Pr[5:10], Edest_k_ss[5:10], c = 'b', s = 50, marker = 'x')
    # plt.plot(Pr[5:10], Edest_k_ss[5:10], c = 'b', linestyle='--')
    # plt.scatter(Pr[10:15], Edest_k_ss[10:15], c = 'r', s = 50, marker = 'x')
    # plt.plot(Pr[10:15], Edest_k_ss[10:15], c = 'r')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('Edest_cooler [kW]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "Qcooler.eps"), format='eps')

    # # plt.legend()

    # fig3 = plt.figure(3)
    # # plt.errorbar(Pr[0:5], Pmech_real[0:5], yerr= Pmech_uncertainty[0:5], fmt='o', c='g')
    # plt.scatter(Pr[0:5], Edest_kr_ss[0:5], c = 'g', s = 50, marker = 'x')
    # plt.plot(Pr[0:5], Edest_kr_ss[0:5], linestyle=':', c = 'g')
    # plt.scatter(Pr[5:10], Edest_kr_ss[5:10], c = 'b', s = 50, marker = 'x')
    # plt.plot(Pr[5:10], Edest_kr_ss[5:10], c = 'b', linestyle='--')
    # plt.scatter(Pr[10:15], Edest_kr_ss[10:15], c = 'r', s = 50, marker = 'x')
    # plt.plot(Pr[10:15], Edest_kr_ss[10:15], c = 'r')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('Edest_kr [kW]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "Wmech.eps"), format='eps')

    # # plt.legend()

    # T_error = 1
    # fig4 = plt.figure(4)
    # plt.scatter(Pr[0:5], Edest_hr_ss[0:5], c = 'g', s = 50, marker = 'x')
    # plt.plot(Pr[0:5], Edest_hr_ss[0:5], c = 'g', linestyle=':')
    # plt.scatter(Pr[5:10], Edest_hr_ss[5:10], c = 'b', s = 50, marker = 'x')
    # plt.plot(Pr[5:10], Edest_hr_ss[5:10], c = 'b', linestyle='--')
    # plt.scatter(Pr[10:15], Edest_hr_ss[10:15], c = 'r', s = 50, marker = 'x')
    # plt.plot(Pr[10:15], Edest_hr_ss[10:15], c = 'r')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('Edest_hr [kW]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "Tdis.eps"), format='eps')
    # # plt.legend()

    # fig5 = plt.figure(5)

    # plt.scatter(Pr[0:5], Edest_r_ss[0:5], c = 'g', s = 50, marker = 'x')
    # plt.plot(Pr[0:5], Edest_r_ss[0:5], c = 'g', linestyle=':')
    # plt.scatter(Pr[5:10], Edest_r_ss[5:10], c = 'b', s = 50, marker = 'x')
    # plt.plot(Pr[5:10], Edest_r_ss[5:10], c = 'b', linestyle='--')
    # plt.scatter(Pr[10:15], Edest_r_ss[10:15], c = 'r', s = 50, marker = 'x')
    # plt.plot(Pr[10:15], Edest_r_ss[10:15], c = 'r')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('Edest_r [kW]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "mdot.eps"), format='eps')

    # fig6 = plt.figure(6)
    # plt.scatter(Pr[0:5], Edest_h_ss[0:5], c = 'g', s = 50, marker = 'x')
    # plt.plot(Pr[0:5], Edest_h_ss[0:5], c = 'g',linestyle=':')
    # plt.scatter(Pr[5:10], Edest_h_ss[5:10], c = 'b', s = 50, marker = 'x')
    # plt.plot(Pr[5:10], Edest_h_ss[5:10], c = 'b', linestyle='--')
    # plt.scatter(Pr[10:15], Edest_h_ss[10:15], c = 'r', s = 50, marker = 'x')
    # plt.plot(Pr[10:15], Edest_h_ss[10:15], c = 'r')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('Edest_h [kW]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.tight_layout()

    # plt.show()
