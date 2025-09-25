import csv
import numpy as np
from utils import (
	get_state,
)
import CoolProp.CoolProp as CP

# Initialize lists to hold the column data
t = []
d_Hpev = []
d_Lpev = []
d_mw_dot = []
d_mmpg_dot = []
d_Tw_in = []
d_Tmpg_in = []
d_Tw_out = []
d_Tmpg_out = []
d_pc = []
d_p25 = []
d_pi = []
d_pe = []
d_Tc_out = []
d_Te_out = []
d_omega1 = []
d_omega2 = []
d_omega3 = []
d_omegab = []
d_Theater1 = []
d_Theater2 = []
d_Pcomb = []
d_Pheat_out = []
d_Pevap = []
d_Pm1 = []
d_Pm2 = []
d_Pm3 = []
d_COP = []
d_SH = []
d_Ex_eff = []
i = 0
cp_w = 4186

mpg_percentage = 25
cp_mpg = (100 - mpg_percentage)/100*cp_w + mpg_percentage/100 * 0.6 * cp_w

step_counter = 0
# Read the CSV file
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\LPV PRBS.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\HPV PRBS.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\bf step.csv') as csv_file:
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\M1 step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\M2 step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\M3 step.csv') as csv_file:      
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\Tw step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\mw step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\Tmpg step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\mmpg step.csv') as csv_file: 
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        step_counter += 1  
        try:
            if row['omegab [rpm]']:
                d_omegab.append(float(row['omegab [rpm]']))
            if row['Tw_in [°C]']:
                d_Tw_in.append(float(row['Tw_in [°C]']))
            if row['Tmpg_in [°C]']:
                d_Tmpg_in.append(float(row['Tmpg_in [°C]']))
            if row['mmpg_dot [l/min]']:
                d_mmpg_dot.append(float(row['mmpg_dot [l/min]'])/60)
            if row['mw_dot [l/min]']:
                d_mw_dot.append(float(row['mw_dot [l/min]'])/60)
            if row['Hpev']:
                d_Hpev.append(float(row['Hpev']))
            if row['Lpev']:
                d_Lpev.append(float(row['Lpev']))
            if row['omega1 [rpm]']:
                d_omega1.append(float(row['omega1 [rpm]']))
            if row['omega2 [rpm]']:
                d_omega2.append(float(row['omega2 [rpm]']))
            if row['omega3 [rpm]']:
                d_omega3.append(float(row['omega3 [rpm]']))         
            if row['Theater1 [°C]']:
                d_Theater1.append(float(row['Theater1 [°C]']))
            if row['Theater2 [°C]']:
                d_Theater2.append(float(row['Theater2 [°C]']))
            if row['pc [bar]']:
                d_pc.append(float(row['pc [bar]']))
            if row['p25 [bar]']:
                d_p25.append(float(row['p25 [bar]']))
            if row['pi [bar]']:
                d_pi.append(float(row['pi [bar]']))
            if row['pe [bar]']:
                d_pe.append(float(row['pe [bar]']))
            if row['Tc_out [°C]']:
                d_Tc_out.append(float(row['Tc_out [°C]']))
            if row['Te_out [°C]']:
                d_Te_out.append(float(row['Te_out [°C]']))
            if row['Pheat_out [W]']:
                d_Pheat_out.append(float(row['Pheat_out [W]']))
            if row['Pm1']:
                d_Pm1.append(float(row['Pm1']))
            if row['Pm2']:
                d_Pm2.append(float(row['Pm2']))
            if row['Pm3']:
                d_Pm3.append(float(row['Pm3']))
            if row['Tw_out [°C]']:
                d_Tw_out.append(float(row['Tw_out [°C]']))
            if row['Tmpg_out [°C]']:
                d_Tmpg_out.append(float(row['Tmpg_out [°C]']))
        except ValueError as e:
            print(f"Skipping row {i} due to error: {e}")

d_Te_out = np.interp(np.linspace(0, len(d_Te_out) - 1, len(d_omegab)), np.arange(len(d_Te_out)), d_Te_out)
d_Tc_out = np.interp(np.linspace(0, len(d_Tc_out) - 1, len(d_omegab)), np.arange(len(d_Tc_out)), d_Tc_out)
d_Pm1 = np.interp(np.linspace(0, len(d_Pm1) - 1, len(d_omegab)), np.arange(len(d_Pm1)), d_Pm1)
d_Pm2 = np.interp(np.linspace(0, len(d_Pm2) - 1, len(d_omegab)), np.arange(len(d_Pm2)), d_Pm2)
d_Pm3 = np.interp(np.linspace(0, len(d_Pm3) - 1, len(d_omegab)), np.arange(len(d_Pm3)), d_Pm3)

LHV = 50e6
p0 = 101325
T0 = 275
h0 = CP.PropsSI('H', 'P', p0, 'T', T0, 'Air')
s0 = CP.PropsSI('S', 'P', p0, 'T', T0, 'Air')

for i in range(len(d_omegab)):
    mCH4_dot = (0.0022 * d_omegab[i] - 2.5965) * 0.657/60000
    d_Pcomb.append(LHV * mCH4_dot)
    d_COP.append(d_Pheat_out[i]/d_Pcomb[-1])
    d_Pevap.append(d_mmpg_dot[i] * cp_mpg * (d_Tmpg_in[i] - d_Tmpg_out[i]))
    state = get_state(CP.PQ_INPUTS, d_pe[i] * 1e5, 1)
    d_SH.append(d_Te_out[i] + 273.15 - state.T())

    sw_in = CP.PropsSI('S', 'P', p0, 'T', d_Tw_in[i] + 273.15, 'Water')
    hw_in = CP.PropsSI('H', 'P', p0, 'T', d_Tw_in[i] + 273.15, 'Water')

    sw_out = CP.PropsSI('S', 'P', p0, 'T', d_Tw_out[i] + 273.15, 'Water')
    hw_out = CP.PropsSI('H', 'P', p0, 'T', d_Tw_out[i] + 273.15, 'Water')
    psiw_in = (hw_in - h0) - T0*(sw_in - s0)
    psiw_out = (hw_out - h0) - T0*(sw_out - s0)

    d_Ex_eff.append(d_mw_dot[i] * (psiw_out - psiw_in)/(d_Pcomb[i] * (1 - T0/1400) + d_Pm1[i] + d_Pm2[i] + d_Pm3[i]))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# 1) Prepare data (edit here)
# ---------------------------
Ts_exp = 1.0  # [s] sampling time of experimental data (adjust if needed)
Ts_sim = 1.0  # [s] sampling time of simulation data (adjust if needed)

N_target = 7000
# Experimental (from your code above)
d_Pheat_out = np.asarray(d_Pheat_out[2500:N_target])          # [W]
d_Pevap     = np.asarray(d_Pevap[2500:N_target])              # [W]
d_COP       = np.asarray(d_COP[2500:N_target])                # [-]
d_Tw_out       = np.asarray(d_Tw_out[2500:N_target])                # [-]
d_Tc_out       = np.asarray(d_Tc_out[2500:N_target])                # [-]
d_SH       = np.asarray(d_SH[2500:N_target])                # [-]
d_pe       = np.asarray(d_pe[2500:N_target])                # [-]
d_pc       = np.asarray(d_pc[2500:N_target])                # [-]
d_Lpev       = np.asarray(d_Lpev[2500:N_target])                # [-]
d_Hpev       = np.asarray(d_Hpev[2500:N_target])                # [-]
d_omegab       = np.asarray(d_omegab[2500:N_target])                # [-]


# Time vectors in minutes
t1_ss = np.arange(len(d_Pheat_out)) * (Ts_exp/60.0)   # experimental time base [min]

# ========= LPV LOOP (controls superheat) =========
fig_lpv, (ax_lpv1, ax_lpv2, ax_lpv3) = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

# --- Low-side pressure (coupling output) ---
# Optional sim overlay:
# ax_lpv2.plot(t_ss, pe_ss, '--', lw=1.8, label='Sim: Low-side pressure', color='#3CB371')
ax_lpv1.plot(t1_ss, d_pe, '-', lw=2.3, label='Low-side', color='#3CB371')
ax_lpv1.set_ylabel(r"Pressure [bar]", fontsize=14)
ax_lpv1.legend(fontsize=11)
ax_lpv1.grid(True, alpha=0.3)

# --- Superheat (primary output) ---
# Optional sim overlay:
# ax_lpv1.plot(t_ss, SH_ss, '--', lw=1.8, label='Sim', color='#1E90FF')
ax_lpv2.plot(t1_ss, d_SH, '-', lw=2.3, color='#1E90FF')
ax_lpv2.set_ylabel(r"Superheat [K]", fontsize=14)
ax_lpv2.legend(fontsize=11)
ax_lpv2.grid(True, alpha=0.3)

# --- LPV input (manipulated variable) ---
ax_lpv3.plot(t1_ss, d_Lpev, lw=2.3, label='LPV', color="#B6CD5A")
ax_lpv3.set_xlabel("Time [min]", fontsize=14)
ax_lpv3.set_ylabel("Valve opening [%]", fontsize=14)
ax_lpv3.legend(fontsize=11)
ax_lpv3.grid(True, alpha=0.3)

# --- Increase tick label size only ---
for ax in (ax_lpv1, ax_lpv2, ax_lpv3):
    ax.tick_params(axis='both', which='major', labelsize=14)

fig_lpv.tight_layout()

plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\transient\LPV.eps", format='eps', bbox_inches='tight')
plt.show()