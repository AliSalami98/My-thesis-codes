# import numpy as np
import time 
import cProfile
import pstats
from concurrent.futures import ProcessPoolExecutor

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
from data_filling_ss import data, Pr
import time

def multi_solving(k):
    start_time = time.time()  # Start timing

    Tin = data['Tin [K]'][k]
    pin = data['pin [pa]'][k] - 0.5*10**5
    pout = data['pout [pa]'][k] + 0.5*10**5
    state.update(CP.PT_INPUTS, pin, Tin)
    hin= state.hmass()
    sin= state.smass()
    Din = state.rhomass()
    Th_wall_ext = data['Th_wall [K]'][k]
    Tw_in = data['Tw_in [K]'][k]
    omega= data['omega [rpm]'][k]

    Th_wall = Th_wall_ext - 100
    Tk_wall_ext = Tw_in
    Tk_wall = Tk_wall_ext

    pc = pin
    pe = pc - 0.1*10**5
    N = nk + nkr + nreg + nhr + nh +2
    Twall= N*[0]
    Twall_ext= N*[0]
    Dn = N*[0]
    Tn = N*[0]
    mn = N*[0]
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

    T_mean = np.mean(Tn)
    n_cycles = 0
    mdot_error = 100
    T_error = 100
    alpha_error = 1000
    everything = 'ayri'
    Q_array_sum = 1000
    pinit = 0
    pfinal = 0
    p_error = 10

    # PID controller variables
    Kp = 1e4
    Ki = 0
    Kd = 0  #5e3

    integral_error = 0
    prev_error = 0

    # while (everything != 'okay'):

    while (np.abs(alpha_error) > 100 or np.abs(Q_array_sum) > 100) and n_cycles < 1:
        c = [0]
        d = [0]
        sol = solve_ivp(model,theta0, X0, method = 'RK23', args = (pin, pout, sin, Din, hin, omega, Twall, Twall_ext, Tw_in, c, d))
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
        ) = post_process(y, theta, sin,  pin, pout, Din, hin, omega, Twall)
        mdot_error = (np.mean(a_mint_dot) - np.mean(a_mout_dot))
        alpha_error = np.mean(a_alpha)
        row_means = np.mean(Q_array, axis=0)

        Q_array_sum = np.sum(row_means)


        for i in range(nk + nkr + 1, nk + nkr + nreg + 1):
            Twall[i] -= 1e-2 * row_means[i - (nk + nkr + 1)]

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
    Pcool_ss = -(min(np.mean(a_Qh),0) + min(np.mean(a_Qhr),0) + min(np.mean(a_Qr),0) + min(np.mean(a_Qkr), 0) + min(np.mean(a_Qk), 0) + min(np.mean(a_Qc), 0))
    alpha_ss = np.mean(a_alpha)
    Pout_ss = np.mean(a_Pout)
    eff_ss = np.mean(a_Pout)/(Pheat_ss - np.mean(a_W))
    elapsed_time = time.time() - start_time  # End timing
    print(f"[{k}] Execution time: {elapsed_time:.2f} seconds")
    
    return [Pheat_ss, Pcool_ss, W_ss, mdot_ss, Tout_ss, alpha_ss, Pout_ss, eff_ss]

# Function to run in parallel
def run_in_parallel(inputs):
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(multi_solving, inputs))
    return results

# List of tuples, each containing a different set of parameters and initial conditions
inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

import os
save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow'
os.makedirs(save_dir, exist_ok=True)

W_ss = []
mdot_ss = []
Tout_ss = []
Pheat_ss = []
Pcool_ss = []
alpha_ss = []
Pout_ss = []
eff_ss = []
# Protect the program's entry point
if __name__ == '__main__':
    results = run_in_parallel(inputs)

    # Example usage of the results
    for i, sol in enumerate(results):
        Pheat_ss.append(sol[0] / 1e3)
        Pcool_ss.append(sol[1] / 1e3)
        W_ss.append(sol[2] / 1e3)
        mdot_ss.append(sol[3] * 1e3)
        Tout_ss.append(sol[4] - 273.15)
        alpha_ss.append(sol[5])
        Pout_ss.append(sol[6] / 1e3)
        eff_ss.append(sol[7])
        # print(f"Solution for initial condition {inputs[i]}: {sol}")

    Pheat_real = np.array(data['Pheating [W]'])/1e3
    Pcool_real = np.array(data['Pcooling [W]'])/1e3
    Pmech_real = np.array(data['Pmech [W]'])/1e3
    Pout_real = np.array(data['Pout [W]'])/1e3
    Tdis_real = np.array(data['Tout [K]']) - 273.15
    mdot_real = np.array(data['mdot [g/s]'])

    # Relative deviations:
    MAPE_Pheat = [100 * np.abs(x - y)/np.abs(y) for x, y in zip(Pheat_ss, Pheat_real)]
    print('MAPE of Pheating', np.mean(MAPE_Pheat))
    MAPE_Pcool = [100 * np.abs(x - y)/np.abs(y) for x, y in zip(Pcool_ss, Pcool_real)]
    print('MAPE of Pcooling', np.mean(MAPE_Pcool))
    MAPE_Pmech = [100 * np.abs(x - y)/np.abs(y) for x, y in zip(W_ss, Pmech_real)]
    print('MAPE of Pmech', np.mean(MAPE_Pmech))
    MAE_Tout = [np.abs(x - y) for x, y in zip(Tout_ss, Tdis_real)]
    print('MAE of Tout [°C]', np.mean(MAE_Tout))
    MAPE_mdot = [100 * np.abs(x - y)/np.abs(y) for x, y in zip(mdot_ss, mdot_real)]
    print('MAPE of mdot', np.mean(MAPE_mdot))
    import matplotlib.pyplot as plt

    Pheating_uncertainty = [0.05*x for x in Pheat_real]

    fig1 = plt.figure(1)
    # plt.errorbar(Pr[0:5], Pheat_real[0:5], yerr= Pheating_uncertainty[0:5], fmt='o', c='g', label='30 bar exp')
    plt.scatter(Pr[0:5], Pheat_real[0:5], c='g', label='30 bar exp')
    plt.scatter(Pr[0:5], Pheat_ss[0:5], c = 'g', s = 50, marker = 'x', label = '30 bar sim')
    plt.plot(Pr[0:5], Pheat_ss[0:5], linestyle=':', c = 'g')
    # plt.errorbar(Pr[5:10], Pheat_real[5:10], yerr=Pheating_uncertainty[5:10], fmt='^', c='b', label='40 bar exp')
    plt.scatter(Pr[5:10], Pheat_real[5:10], c='b', label='40 bar exp')
    plt.scatter(Pr[5:10], Pheat_ss[5:10], c = 'b', s = 50, marker = 'x', label = '40 bar sim')
    plt.plot(Pr[5:10], Pheat_ss[5:10], c = 'b', linestyle='--')
    # plt.errorbar(Pr[10:15], Pheat_real[10:15], yerr=Pheating_uncertainty[10:15], fmt='s', c='r', label='56 bar exp')
    plt.scatter(Pr[10:15], Pheat_real[10:15], c='r', label='56 bar exp')
    plt.scatter(Pr[10:15], Pheat_ss[10:15], c = 'r', s = 50, marker = 'x', label = '56 bar sim')
    plt.plot(Pr[10:15], Pheat_ss[10:15], c = 'r')
    plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    plt.ylabel('Heating Power $\dot{Q}_{heater}$ [kW]', fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Qheater.eps"), format='eps')

    Pcooling_uncertainty = [0.05*x for x in Pcool_real]
    fig2 = plt.figure(2)
    # plt.errorbar(Pr[0:5], Pcool_real[0:5], yerr= Pcooling_uncertainty[0:5], fmt='o', c='g')
    plt.scatter(Pr[0:5], Pcool_real[0:5], c='g')
    plt.scatter(Pr[0:5], Pcool_ss[0:5], c = 'g', s = 50, marker = 'x')
    plt.plot(Pr[0:5], Pcool_ss[0:5],linestyle=':', c = 'g')
    # plt.errorbar(Pr[5:10], Pcool_real[5:10], yerr=Pcooling_uncertainty[5:10], fmt='^', c='b')
    plt.scatter(Pr[5:10], Pcool_real[5:10], c='b')
    plt.scatter(Pr[5:10], Pcool_ss[5:10], c = 'b', s = 50, marker = 'x')
    plt.plot(Pr[5:10], Pcool_ss[5:10], c = 'b', linestyle='--')
    # plt.errorbar(Pr[10:15], Pcool_real[10:15], yerr=Pcooling_uncertainty[10:15], fmt='s', c='r')
    plt.scatter(Pr[10:15], Pcool_real[10:15], c='r')
    plt.scatter(Pr[10:15], Pcool_ss[10:15], c = 'r', s = 50, marker = 'x')
    plt.plot(Pr[10:15], Pcool_ss[10:15], c = 'r')
    plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    plt.ylabel('Cooling Power $\dot{Q}_{cooler}$ [kW]', fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Qcooler.eps"), format='eps')

    # plt.legend()

    Pmech_uncertainty = [0.05*x for x in Pmech_real]
    fig3 = plt.figure(3)
    # plt.errorbar(Pr[0:5], Pmech_real[0:5], yerr= Pmech_uncertainty[0:5], fmt='o', c='g')
    plt.scatter(Pr[0:5], Pmech_real[0:5], c='g')
    plt.scatter(Pr[0:5], W_ss[0:5], c = 'g', s = 50, marker = 'x')
    plt.plot(Pr[0:5], W_ss[0:5], linestyle=':', c = 'g')
    # plt.errorbar(Pr[5:10], Pmech_real[5:10], yerr=Pmech_uncertainty[5:10], fmt='^', c='b')
    plt.scatter(Pr[5:10], Pmech_real[5:10], c='b')
    plt.scatter(Pr[5:10], W_ss[5:10], c = 'b', s = 50, marker = 'x')
    plt.plot(Pr[5:10], W_ss[5:10], c = 'b', linestyle='--')
    # plt.errorbar(Pr[10:15], Pmech_real[10:15], yerr=Pmech_uncertainty[10:15], fmt='s', c='r')
    plt.scatter(Pr[10:15], Pmech_real[10:15], c='r')
    plt.scatter(Pr[10:15], W_ss[10:15], c = 'r', s = 50, marker = 'x')
    plt.plot(Pr[10:15], W_ss[10:15], c = 'r')
    plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    plt.ylabel('Mechanical Power $\dot{W}_{mech}$ [kW]', fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Wmech.eps"), format='eps')

    # plt.legend()

    T_error = 1
    fig4 = plt.figure(4)
    # plt.errorbar(Pr[0:5], Tdis_real[0:5], yerr= T_error, fmt='o', c='g')
    plt.scatter(Pr[0:5], Tdis_real[0:5], c='g')
    plt.scatter(Pr[0:5], Tout_ss[0:5], c = 'g', s = 50, marker = 'x')
    plt.plot(Pr[0:5], Tout_ss[0:5], c = 'g', linestyle=':')
    # plt.errorbar(Pr[5:10], Tdis_real[5:10], yerr=T_error, fmt='^', c='b')
    plt.scatter(Pr[5:10], Tdis_real[5:10], c='b')
    plt.scatter(Pr[5:10], Tout_ss[5:10], c = 'b', s = 50, marker = 'x')
    plt.plot(Pr[5:10], Tout_ss[5:10], c = 'b', linestyle='--')
    # plt.errorbar(Pr[10:15], Tdis_real[10:15], yerr=T_error, fmt='s', c='r')
    plt.scatter(Pr[10:15], Tdis_real[10:15], c='r')
    plt.scatter(Pr[10:15], Tout_ss[10:15], c = 'r', s = 50, marker = 'x')
    plt.plot(Pr[10:15], Tout_ss[10:15], c = 'r')
    plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    plt.ylabel('Discharge Temperature $T_{dis}$ [°C]', fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Tdis.eps"), format='eps')
    # plt.legend()

    mdot_uncertainty = [0.05*x for x in mdot_real]
    fig5 = plt.figure(5)
    # plt.errorbar(Pr[0:5], mdot_real[0:5], yerr= mdot_uncertainty[0:5], fmt='o', c='g')
    plt.scatter(Pr[0:5], mdot_real[0:5], c='g')
    plt.scatter(Pr[0:5], mdot_ss[0:5], c = 'g', s = 50, marker = 'x')
    plt.plot(Pr[0:5], mdot_ss[0:5], c = 'g', linestyle=':')
    # plt.errorbar(Pr[5:10], mdot_real[5:10], yerr=mdot_uncertainty[5:10], fmt='^', c='b')
    plt.scatter(Pr[5:10], mdot_real[5:10], c='b')
    plt.scatter(Pr[5:10], mdot_ss[5:10], c = 'b', s = 50, marker = 'x')
    plt.plot(Pr[5:10], mdot_ss[5:10], c = 'b', linestyle='--')
    # plt.errorbar(Pr[10:15], mdot_real[10:15], yerr=mdot_uncertainty[10:15], fmt='s', c='r')
    plt.scatter(Pr[10:15], mdot_real[10:15], c='r')
    plt.scatter(Pr[10:15], mdot_ss[10:15], c = 'r', s = 50, marker = 'x')
    plt.plot(Pr[10:15], mdot_ss[10:15], c = 'r')
    plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    plt.ylabel('Mass Flow Rate $\dot{m}_f$ [g/s]', fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mdot.eps"), format='eps')

    fig6 = plt.figure(6)
    plt.scatter(Pr[0:5], alpha_ss[0:5], c = 'g', s = 50, marker = 'x')
    plt.plot(Pr[0:5], alpha_ss[0:5], c = 'g',linestyle=':')
    plt.scatter(Pr[5:10], alpha_ss[5:10], c = 'b', s = 50, marker = 'x')
    plt.plot(Pr[5:10], alpha_ss[5:10], c = 'b', linestyle='--')
    plt.scatter(Pr[10:15], alpha_ss[10:15], c = 'r', s = 50, marker = 'x')
    plt.plot(Pr[10:15], alpha_ss[10:15], c = 'r')
    plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    plt.ylabel('alpha', fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    fig7 = plt.figure(7)
    plt.scatter(Pr[0:5], Pout_real[0:5], c = 'g', s = 50, marker = 'o')
    plt.scatter(Pr[0:5], Pout_ss[0:5], c = 'g', s = 50, marker = 'x')
    plt.plot(Pr[0:5], Pout_ss[0:5], c = 'g', linestyle=':')
    plt.scatter(Pr[5:10], Pout_real[5:10], c = 'b', s = 50, marker = '^')
    plt.scatter(Pr[5:10], Pout_ss[5:10], c = 'b', s = 50, marker = 'x')
    plt.plot(Pr[5:10], Pout_ss[5:10], c = 'b', linestyle='--')
    plt.scatter(Pr[10:15], Pout_real[10:15], c = 'r', s = 50, marker = 's')
    plt.scatter(Pr[10:15], Pout_ss[10:15], c = 'r', s = 50, marker = 'x')
    plt.plot(Pr[10:15], Pout_ss[10:15], c = 'r')
    plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    plt.ylabel('Pout [W]', fontsize = 14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # fig8 = plt.figure(8)
    # plt.scatter(Pr[0:3], data['eff [%]'][0:3], c = 'g', s = 100, marker = 'o')
    # plt.scatter(Pr[0:3], eff_ss[0:3], c = 'g', s = 100, marker = 'x')
    # plt.plot(Pr[0:3], eff_ss[0:3], c = 'g', linestyle='--')
    # plt.scatter(Pr[3:6], data['eff [%]'][3:6], c = 'b', s = 100, marker = '^')
    # plt.scatter(Pr[3:6], eff_ss[3:6], c = 'b', s = 100, marker = 'x')
    # plt.plot(Pr[3:6], eff_ss[3:6], c = 'b', linestyle='--')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('eff [%]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.legend()

    plt.show()


    # fig4 = plt.figure(4)
    # plt.scatter(Pr[0:3], data['Pout [W]'][0:3], c = 'g', s = 100, marker = 'o')
    # plt.scatter(Pr[0:3], Pout_ss[0:3], c = 'r', s = 100, marker = 'x')
    # plt.plot(Pr[0:3], Pout_ss[0:3], c = 'r', linestyle='--')
    # plt.scatter(Pr[3:6], data['Pout [W]'][3:6], c = 'b', s = 100, marker = '^')
    # plt.scatter(Pr[3:6], Pout_ss[3:6], c = 'b', s = 100, marker = 'x')
    # plt.plot(Pr[3:6], Pout_ss[3:6], c = 'b', linestyle='--')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('Pout [W]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.legend()

    # fig5 = plt.figure(5)
    # plt.scatter(Pr[0:3], data['eff [%]'][0:3], c = 'r', s = 100, marker = 'o')
    # plt.scatter(Pr[0:3], eff_ss[0:3], c = 'r', s = 100, marker = 'x')
    # plt.plot(Pr[0:3], eff_ss[0:3], c = 'r', linestyle='--')
    # plt.scatter(Pr[3:6], data['eff [%]'][3:6], c = 'b', s = 100, marker = '^')
    # plt.scatter(Pr[3:6], eff_ss[3:6], c = 'b', s = 100, marker = 'x')
    # plt.plot(Pr[3:6], eff_ss[3:6], c = 'b', linestyle='--')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('eff [%]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.legend()

    # fig6 = plt.figure(6)
    # plt.scatter(Pr[0:3], data['Pout [W]'][0:3], c = 'r', s = 100, marker = 'o')
    # plt.scatter(Pr[0:3], Pout2_ss[0:3], c = 'r', s = 100, marker = 'x')
    # plt.plot(Pr[0:3], Pout2_ss[0:3], c = 'r', linestyle='--')
    # plt.scatter(Pr[3:6], data['Pout [W]'][3:6], c = 'b', s = 100, marker = '^')
    # plt.scatter(Pr[3:6], Pout2_ss[3:6], c = 'b', s = 100, marker = 'x')
    # plt.plot(Pr[3:6], Pout2_ss[3:6], c = 'b', linestyle='--')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('Pout [W]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)
    # plt.legend()

    # fig7 = plt.figure(7)
    # plt.scatter(Pr[0:3], data['eff [%]'][0:3], c = 'r', s = 100, marker = 'o')
    # plt.scatter(Pr[0:3], eff2_ss[0:3], c = 'r', s = 100, marker = 'x')
    # plt.plot(Pr[0:3], eff2_ss[0:3], c = 'r', linestyle='--')
    # plt.scatter(Pr[3:6], data['eff [%]'][3:6], c = 'b', s = 100, marker = '^')
    # plt.scatter(Pr[3:6], eff2_ss[3:6], c = 'b', s = 100, marker = 'x')
    # plt.plot(Pr[3:6], eff2_ss[3:6], c = 'b', linestyle='--')
    # plt.xlabel('Pressure Ratio $P_r$', fontsize = 14)
    # plt.ylabel('eff [%]', fontsize = 14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.grid(True)