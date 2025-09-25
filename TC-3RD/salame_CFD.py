import numpy as np
import time 

import pandas as pd
import os
from scipy.integrate import solve_ivp
from model import model
from post_processing import post_process
from plot import plot_and_print
import cProfile
import pstats
from utils import (
	CP,
	state,

)

from config import(
    theta0,
    nk, 
    nkr, 
    nreg, 
    nhr, 
    nh,
    Vn
)

# profiler = cProfile.Profile()
# profiler.enable()
# Code block or function call you want to profile
from data_filling_CFD import (
    Tc1,
    Tk1, 
    Tr1, 
    Th1,
    Te1, 
    pc1,
    pk1,
    pr1,
    ph1,
    pe1,
    Dc1,
    Dk1,
    Dr1,
    Dh1,
    De1,
    mc1,
    mk1,
    mr1,
    mh1, 
    me1, 
    mint_dot1, 
    mout_dot1, 
    Vc1,
)

file_name = 'a_variables.csv'

Tin = 18+273.15 
pin = 45*10**5
pout = 60 *10**5
state.update(CP.PT_INPUTS, pin, Tin)
hin= state.hmass()
sin= state.smass()
Din = state.rhomass()
Th_wall_ext = 600+273.15
Tw_in = 30+273.15
omega= 180

Th_wall = Th_wall_ext 
Tk_wall_ext = Tw_in
Tk_wall = Tk_wall_ext

pc = pin + 0.5e5
pe = pc - 0.1*10**5
pmean = np.sqrt(pin * pout)
N = nk + nkr + nreg + nhr + nh +2
Twall= N*[0]
Twall_ext= N*[0]
Dn = N*[0]
mn = N*[0]
Tn = N*[0]
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

import time
start_time = time.time()  # Start timing

T_mean = np.mean(Tn)
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

I_error = 0
prev_error = 0
# Wall PID
Kp = 1e4
Ki = 1e2
Kd = 1e3

# Initialize PID memory
I_error_wall = np.zeros(N)
prev_error_wall = np.zeros(N)

Q_setpoint = 0
# while (everything != 'okay'):
n_cycles = 0


while (n_cycles < 4 or np.abs(Q_array_sum) > 200 or np.abs(alpha_error) > 200) and n_cycles < 10: 
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

    # if np.abs(mdot_error) > 0.2:
    #     X0[0] += 5e-6 * mdot_error

    # Apply PID to wall temperatures in regenerator region (assume regenerator spans i = nk to nk+nkr)
    # for i in range(nk, nk + nkr):
    #     error = Q_setpoint - row_means[i - 1]
    #     I_error_wall[i] += error
    #     derivative = error - prev_error_wall[i]
    #     control_signal = Kp * error + Ki * I_error_wall[i] + Kd * derivative
    #     Twall[i] += control_signal  # or -= depending on sign convention
    #     prev_error_wall[i] = error

    # for i in range(nk + nkr + 1, nk + nkr + nreg + 1):
    #     Twall[i] += 5e-2 * mdot_error

    # for i in range(1, nk + nkr + int(nreg/2) + 1):
    #     Twall[i] += 1e-1 * mdot_error

    # for i in range(nk + nkr + int(nreg/2) + 1, N-1):
    #     Twall[i] -= 1e-1 * mdot_error

    pinit = a_pc[0]
    pfinal = a_pc[-1]
    p_error = pfinal - pinit
    n_cycles += 1

    # print(f"perror = {p_error:.6f}")
    print(f"alpha_error = {alpha_error:.2f}")
    print(f"mdot_error = {mdot_error:.4f}")
    print(f"Q_array_sum = {Q_array_sum:.2f}")
    print(f"mdot = {np.mean(a_mdot) * 1e3}")
    print(f"Tout = {np.mean(a_Tout) - 273.15}")
    print(f"Qh_sum = {np.mean(a_Qhr) + np.mean(a_Qh):.2f}")
    print(f"Qk_sum = {np.mean(a_Qkr) + np.mean(a_Qk):.2f}")
    print(f"Wdot = {-np.mean(a_W):.2f}")

        # sol = solve_ivp(model,theta0, X0, method = 'RK23', args = (pin, pout, sin, Din, hin, omega, Twall_ext, Tw_in, c, d))
        # #RK45, RK23, DOP853, Radau, BDF, LSODA
        # # profiler.disable()
        # # stats = pstats.Stats(profiler).sort_stats('time')
        # # stats.print_stats()

        # y = sol.y
        # theta = sol.t

        # ##########################################################################################################

        # (
        #     a_Pout,
        #     a_mck_dot,
        #     a_meh_dot,
        #     a_mkr_dot,
        #     a_mrh_dot,
        #     a_mint_dot,
        #     a_mout_dot,
        #     a_alpha,
        #     a_W,
        #     a_Deltap,
        #     a_Deltapk,
        #     a_Deltapkr,
        #     a_Deltapr,
        #     a_Deltaphr,
        #     a_Deltaph,
        #     a_mdot,
        #     a_Te,
        #     a_Th,
        #     a_Tr,
        #     a_Tc,
        #     a_Tk,
        #     a_Tk_wall,
        #     a_Tr_wall,
        #     a_Th_wall,
        #     a_Tout,
        #     a_hout,
        #     a_Vc,
        #     a_Ve,
        #     a_pc,
        #     a_pk,
        #     a_pr,
        #     a_ph,
        #     a_pe,
        #     a_theta,
        #     a_Dc,
        #     a_Dk,
        #     a_Dkr,
        #     a_Dr,
        #     a_Dhr,
        #     a_Dh,
        #     a_De,
        #     a_mc,
        #     a_mk,
        #     a_mkr,
        #     a_mr,
        #     a_mhr,
        #     a_mh,
        #     a_me,
        #     a_vk,
        #     a_vkr,
        #     a_vr,
        #     a_vhr,
        #     a_vh,
        #     a_Qc,
        #     a_Qk,
        #     a_Qkr,
        #     a_Qr,
        #     a_Qhr,
        #     a_Qh,
        #     a_Qe,
        #     a_Edest_c,
        #     a_Edest_k,
        #     a_Edest_kr,
        #     a_Edest_r,
        #     a_Edest_hr,
        #     a_Edest_h,
        #     a_Edest_e,
        #     a_Ex_eff,
        #     Q_array
        # ) = post_process(y, theta, sin,  pin, pout, Din, hin, omega)
        # mdot_error = (np.mean(a_mint_dot) - np.mean(a_mout_dot))
        # alpha_error = np.mean(a_alpha)
        # row_means = np.mean(Q_array, axis=0)

        # Q_array_sum = np.sum(row_means)

        # # Apply PID to wall temperatures in regenerator region (assume regenerator spans i = nk to nk+nkr)
        # # for i in range(nk, nk + nkr):
        # #     error = Q_setpoint - row_means[i - 1]
        # #     I_error_wall[i] += error
        # #     derivative = error - prev_error_wall[i]
        # #     control_signal = Kp * error + Ki * I_error_wall[i] + Kd * derivative
        # #     Twall[i] += control_signal  # or -= depending on sign convention
        # #     prev_error_wall[i] = error

        # # for i in range(nk + nkr + 1, nk + nkr + nreg + 1):
        # #     Twall[i] += 5e-2 * mdot_error

        # # for i in range(1, nk + nkr + int(nreg/2) + 1):
        # #     Twall[i] += 1e-1 * mdot_error

        # # for i in range(nk + nkr + int(nreg/2) + 1, N-1):
        # #     Twall[i] -= 1e-1 * mdot_error

        # pinit = a_pc[0]
        # pfinal = a_pc[-1]
        # p_error = pfinal - pinit

        # print(f"perror = {p_error:.6f}")
        # print(f"alpha_error = {alpha_error:.2f}")
        # print(f"mdot_error = {mdot_error:.4f}")
        # print(f"Q_array_sum = {Q_array_sum:.2f}")
    
    # else:
    #     everything = 'okay'
    


# Create dictionary of all a_ variables
a_data = {
    'theta': a_theta,
    'Tc [K]': a_Tc,
    'Tk [K]': a_Tk,
    'Tr [K]': a_Tr,
    'Th [K]': a_Th,
    'Te [K]': a_Te,
    'pc [pa]': a_pc,
    'pk [pa]': a_pk,
    'pr [pa]': a_pr,
    'ph [pa]': a_ph,
    'pe [pa]': a_pe,
    'Dc [kg/m3]': a_Dc,
    'Dk [kg/m3]': a_Dk,
    'Dr [kg/m3]': a_Dr,
    'Dh [kg/m3]': a_Dh,
    'De [kg/m3]': a_De,
    'mc [g]': a_mc,
    'mk [g]': a_mk,
    'mkr [g]': a_mkr,
    'mr [g]': a_mr,
    'mhr [g]': a_mhr,
    'mh [g]': a_mh,
    'me [g]': a_me,
    'mint_dot [kg/s]': a_mint_dot,
    'mout_dot [kg/s]': a_mout_dot,
    'Vc [m3]': a_Vc,
    'Ve [m3]': a_Ve,
    'Tk_wall [K]': a_Tk_wall,
    'Tr_wall [K]': a_Tr_wall,
    'Th_wall [K]': a_Th_wall
}

# Convert to DataFrame
df_a = pd.DataFrame(a_data)

# Save path
save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\CFD'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, file_name)

# Save as CSV
df_a.to_csv(save_path, index=False)
print(f"Saved a_ variables to {save_path}")

# elapsed_time = time.time() - start_time  # End timing
# print(f"Execution time: {elapsed_time:.2f} seconds")

# theta1 = np.linspace(0, 360, len(Tc1))

# step = len(a_Tc)//len(Tc1)

# a_Tc = np.array(a_Tc[::step][:len(Tc1)]) - 273.15
# a_Tk = np.array(a_Tk[::step][:len(Tc1)]) - 273.15
# a_Tr = np.array(a_Tr[::step][:len(Tc1)]) - 273.15
# a_Th = np.array(a_Th[::step][:len(Th1)]) - 273.15
# a_Te = np.array(a_Te[::step][:len(Tc1)]) - 273.15
# a_pc = a_pc[::step][:len(Tc1)]
# a_pk = a_pk[::step][:len(Tc1)]
# a_pr = a_pr[::step][:len(Tc1)]
# a_ph = a_ph[::step][:len(Tc1)]
# a_pe = a_pe[::step][:len(Tc1)]
# a_Dc = a_Dc[::step][:len(Tc1)]
# a_Dk = a_Dk[::step][:len(Tc1)]
# a_Dr = a_Dr[::step][:len(Tc1)]
# a_Dh = a_Dh[::step][:len(Tc1)]
# a_De = a_De[::step][:len(Tc1)]
# a_mc = a_mc[::step][:len(Tc1)]
# a_mk = a_mk[::step][:len(Tc1)]
# a_mr = a_mr[::step][:len(Tc1)]
# a_mh = a_mh[::step][:len(Tc1)]
# a_me = a_me[::step][:len(Tc1)]
# a_mint_dot = a_mint_dot[::step][:len(Tc1)]
# a_mout_dot = a_mout_dot[::step][:len(Tc1)]
# a_Vc = a_Vc[::step][:len(Tc1)]
# a_Ve = a_Ve[::step][:len(Tc1)]

# print(f"{'mdot_in[g/s]':<15}", np.mean(a_mint_dot), np.mean(mint_dot1))
# print(f"{'mdot_out[g/s]':<15}", np.mean(a_mout_dot), np.mean(mout_dot1))
# print(f"{'Tc [°C]':<15}", np.mean(a_Tc), np.mean(Tc1))
# print(f"{'Tk [°C]':<15}", np.mean(a_Tk), np.mean(Tk1))
# print(f"{'Tr [°C]':<15}", np.mean(a_Tr), np.mean(Tr1))
# print(f"{'Th [°C]':<15}", np.mean(a_Th), np.mean(Th1))
# print(f"{'Te [°C]':<15}", np.mean(a_Te), np.mean(Te1))
# print(f"{'pc [bar]':<15}", np.mean(a_pc), np.mean(pc1))
# print(f"{'pk [bar]':<15}", np.mean(a_pk), np.mean(pk1))
# print(f"{'pr [bar]':<15}", np.mean(a_pr), np.mean(pr1))
# print(f"{'ph [bar]':<15}", np.mean(a_ph), np.mean(ph1))
# print(f"{'pe [bar]':<15}", np.mean(a_pe), np.mean(pe1))
# print(f"{'Dc [kg/m^3]':<15}", np.mean(a_Dc), np.mean(Dc1))
# print(f"{'Dk [kg/m^3]':<15}", np.mean(a_Dk), np.mean(Dk1))
# print(f"{'Dr [kg/m^3]':<15}", np.mean(a_Dr), np.mean(Dr1))
# print(f"{'Dh [kg/m^3]':<15}", np.mean(a_Dh), np.mean(Dh1))
# print(f"{'De [kg/m^3]':<15}", np.mean(a_De), np.mean(De1))

# import matplotlib.pyplot as plt
# import os
# save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\CFD'
# os.makedirs(save_dir, exist_ok=True)

# plt.figure(1)
# plt.plot(a_Vc, pc1,color='b', label='Compression')
# plt.plot(a_Vc, a_pc, color='b', linestyle='--')
# plt.plot(a_Ve, pe1,color='r', label='Expansion')
# plt.plot(a_Ve, a_pe, color='r', linestyle='--')
# plt.xlabel("Volume [$cm^3$]", fontsize = 14)
# plt.ylabel("Pressure [bar]", fontsize = 14)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.legend()
# plt.grid()
# plt.savefig(os.path.join(save_dir, "TC_p-V.eps"), format='eps')

# # Temperature comparison
# plt.figure(2)
# plt.plot(theta1, Tc1, color='b', label="Compression")
# plt.plot(theta1, a_Tc,linestyle='--', color='b')
# plt.plot(theta1, Tk1, color='cyan', label="Cooler")
# plt.plot(theta1, a_Tk,linestyle='--', color='cyan')
# plt.plot(theta1, Tr1, color='g', label="Regenerator")
# plt.plot(theta1, a_Tr,linestyle='--', color='g')
# plt.plot(theta1, Th1, color='orange', label="Heater")
# plt.plot(theta1, a_Th,linestyle='--', color='orange')
# plt.plot(theta1, Te1, color='r', label="Expansion")
# plt.plot(theta1, a_Te,linestyle='--', color='r')
# plt.xlabel("theta [°]", fontsize = 14)
# plt.ylabel("Temperature [°C]", fontsize = 14)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.legend()
# plt.grid()
# plt.savefig(os.path.join(save_dir, "TC_T.eps"), format='eps')

# # Density comparison
# plt.figure(3)
# plt.plot(theta1, Dc1, color='b', label="Compression")
# plt.plot(theta1, a_Dc, linestyle = '--', color='b')
# plt.plot(theta1, Dk1, color='cyan', label="Cooler")
# plt.plot(theta1, a_Dk, linestyle = '--', color='cyan')
# plt.plot(theta1, Dr1, color='g', label="Regenerator")
# plt.plot(theta1, a_Dr, linestyle = '--', color='g')
# plt.plot(theta1, Dh1, color='orange', label="Heater")
# plt.plot(theta1, a_Dh, linestyle = '--', color='orange')
# plt.plot(theta1, De1, color='r', label="Expansion")
# plt.plot(theta1, a_De, linestyle = '--', color='r')
# plt.xlabel("Theta [°]", fontsize = 14)
# plt.ylabel("Density [kg/m³]", fontsize = 14)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# # plt.legend()
# plt.grid()
# plt.savefig(os.path.join(save_dir, "TC_D.eps"), format='eps')


# # Mass flow rate comparison
# plt.figure(4)
# plt.plot(theta1, mint_dot1, label="Suction")
# plt.plot(theta1, a_mint_dot, linestyle = '--')
# plt.plot(theta1, mout_dot1, label="Discharge")
# plt.plot(theta1, a_mout_dot, linestyle = '--')
# plt.xlabel("theta [°]", fontsize = 14)
# plt.ylabel("Mass flow rate [g/s]", fontsize = 14)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.legend()
# plt.grid()
# plt.savefig(os.path.join(save_dir, "TC_mdot.eps"), format='eps')

# # Pressure comparison
# plt.figure(5)
# plt.plot(theta1, pc1, color='b', label="Compression")
# plt.plot(theta1, a_pc, linestyle = '--', color='b')
# plt.plot(theta1, pk1, color='cyan', label="Cooler")
# plt.plot(theta1, a_pk, linestyle = '--', color='cyan')
# plt.plot(theta1, pr1, color='g', label="Regenerator")
# plt.plot(theta1, a_pr, linestyle = '--', color='g')
# plt.plot(theta1, ph1, color='orange', label="Heater")
# plt.plot(theta1, a_ph, linestyle = '--', color='orange')
# plt.plot(theta1, pe1, color='r', label="Expansion")
# plt.plot(theta1, a_pe, linestyle = '--', color='r')
# plt.xlabel("theta [°]", fontsize = 14)
# plt.ylabel("Pressure [bar]", fontsize = 14)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# plt.grid()
# plt.savefig(os.path.join(save_dir, "TC_p.eps"), format='eps')

# # Mass comparison
# plt.figure(6)
# plt.plot(theta1, mc1, color='b', label="Compression")
# plt.plot(theta1, a_mc, linestyle = '--', color='b')
# plt.plot(theta1, mk1, color='cyan', label="Cooler")
# plt.plot(theta1, a_mk, linestyle = '--', color='cyan')
# plt.plot(theta1, mr1, color='g', label="Regenerator")
# plt.plot(theta1, a_mr, linestyle = '--', color='g')
# plt.plot(theta1, mh1, color='orange', label="Heater")
# plt.plot(theta1, a_mh, linestyle = '--', color='orange')
# plt.plot(theta1, me1, color='r', label="Expansion")
# plt.plot(theta1, a_me, linestyle = '--', color='r')
# plt.xlabel("theta [°]", fontsize = 14)
# plt.ylabel("Mass [g]", fontsize = 14)
# plt.xticks(fontsize = 12)
# plt.yticks(fontsize = 12)
# # plt.legend()
# plt.grid()
# plt.savefig(os.path.join(save_dir, "TC_m.eps"), format='eps')

# plt.show()
