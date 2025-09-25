import csv
import numpy as np
from utils import (
	get_state,
)
import CoolProp.CoolProp as CP

# Initialize lists to hold the column data
t = []
Hpev = []
Lpev = []
mw_dot = []
mmpg_dot = []
Tw_in = []
Tmpg_in = []
Tw_out = []
Tmpg_out = []
pc = []
p25 = []
pi = []
pe = []
Tc_out = []
Te_out = []
omega1 = []
omega2 = []
omega3 = []
omegab = []
Theater1 = []
Theater2 = []
Pcomb = []
Pheat_out = []
Pevap = []
Pm1 = []
Pm2 = []
Pm3 = []
COP = []
SH = []
Ex_eff = []
i = 0
cp_w = 4186

mpg_percentage = 25
cp_mpg = (100 - mpg_percentage)/100*cp_w + mpg_percentage/100 * 0.6 * cp_w

step_counter = 0
# Read the CSV file
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\LPV step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\HPV step.csv') as csv_file:    
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\bf step.csv') as csv_file:
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\M1 step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\M2 step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\M3 step.csv') as csv_file:      
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\Tw step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\mw step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\Tmpg step.csv') as csv_file:    
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\mmpg step.csv') as csv_file: 
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        try:
            if row['omegab [rpm]']:
                omegab.append(float(row['omegab [rpm]']))
            if row['Tw_in [°C]']:
                Tw_in.append(float(row['Tw_in [°C]']))
            if row['Tmpg_in [°C]']:
                Tmpg_in.append(float(row['Tmpg_in [°C]']))
            if row['mmpg_dot [l/min]']:
                mmpg_dot.append(float(row['mmpg_dot [l/min]'])/60)
            if row['mw_dot [l/min]']:
                mw_dot.append(float(row['mw_dot [l/min]'])/60)
            if row['Hpev']:
                Hpev.append(float(row['Hpev']))
            if row['Lpev']:
                Lpev.append(float(row['Lpev']))
            if row['omega1 [rpm]']:
                omega1.append(float(row['omega1 [rpm]']))
            if row['omega2 [rpm]']:
                omega2.append(float(row['omega2 [rpm]']))
            if row['omega3 [rpm]']:
                omega3.append(float(row['omega3 [rpm]']))         
            if row['Theater1 [°C]']:
                Theater1.append(float(row['Theater1 [°C]']))
            if row['Theater2 [°C]']:
                Theater2.append(float(row['Theater2 [°C]']))
            if row['pc [bar]']:
                pc.append(float(row['pc [bar]']))
            if row['p25 [bar]']:
                p25.append(float(row['p25 [bar]']))
            if row['pi [bar]']:
                pi.append(float(row['pi [bar]']))
            if row['pe [bar]']:
                pe.append(float(row['pe [bar]']))
            if row['Tc_out [°C]']:
                Tc_out.append(float(row['Tc_out [°C]']))
            if row['Te_out [°C]']:
                Te_out.append(float(row['Te_out [°C]']))
            if row['Pheat_out [W]']:
                Pheat_out.append(float(row['Pheat_out [W]']))
            if row['Pm1']:
                Pm1.append(float(row['Pm1']))
            if row['Pm2']:
                Pm2.append(float(row['Pm2']))
            if row['Pm3']:
                Pm3.append(float(row['Pm3']))
            if row['Tw_out [°C]']:
                Tw_out.append(float(row['Tw_out [°C]']))
            if row['Tmpg_out [°C]']:
                Tmpg_out.append(float(row['Tmpg_out [°C]']))
        except ValueError as e:
            print(f"Skipping row {i} due to error: {e}")

        # step_counter += 1            

Te_out = np.interp(np.linspace(0, len(Te_out) - 1, len(omegab)), np.arange(len(Te_out)), Te_out)
Tc_out = np.interp(np.linspace(0, len(Tc_out) - 1, len(omegab)), np.arange(len(Tc_out)), Tc_out)
Pm1 = np.interp(np.linspace(0, len(Pm1) - 1, len(omegab)), np.arange(len(Pm1)), Pm1)
Pm2 = np.interp(np.linspace(0, len(Pm2) - 1, len(omegab)), np.arange(len(Pm2)), Pm2)
Pm3 = np.interp(np.linspace(0, len(Pm3) - 1, len(omegab)), np.arange(len(Pm3)), Pm3)
# mmpg_dot = np.interp(np.linspace(0, len(mmpg_dot) - 1, len(omegab)), np.arange(len(mmpg_dot)), mmpg_dot)

LHV = 50e6
p0 = 101325
T0 = 275
h0 = CP.PropsSI('H', 'P', p0, 'T', T0, 'Air')
s0 = CP.PropsSI('S', 'P', p0, 'T', T0, 'Air')

for i in range(len(omegab)):
    mCH4_dot = (0.0022 * omegab[i] - 2.5965) * 0.657/60000
    Pcomb.append(LHV * mCH4_dot)
    COP.append(Pheat_out[i]/Pcomb[-1])
    Pevap.append(mmpg_dot[i] * cp_mpg * (Tmpg_in[i] - Tmpg_out[i]))
    state = get_state(CP.PQ_INPUTS, pe[i] * 1e5, 1)
    SH.append(Te_out[i] - state.T())

    sw_in = CP.PropsSI('S', 'P', p0, 'T', Tw_in[i] + 273.15, 'Water')
    hw_in = CP.PropsSI('H', 'P', p0, 'T', Tw_in[i] + 273.15, 'Water')

    sw_out = CP.PropsSI('S', 'P', p0, 'T', Tw_out[i] + 273.15, 'Water')
    hw_out = CP.PropsSI('H', 'P', p0, 'T', Tw_out[i] + 273.15, 'Water')
    psiw_in = (hw_in - h0) - T0*(sw_in - s0)
    psiw_out = (hw_out - h0) - T0*(sw_out - s0)

    Ex_eff.append(mw_dot[i] * (psiw_out - psiw_in)/(Pcomb[i] * (1 - T0/1400) + Pm1[i] + Pm2[i] + Pm3[i]))

import matplotlib.pyplot as plt

# Create time vector
time = np.arange(len(omegab))

# Setup 5 subplots
fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# Set global font sizes
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 12

# 1. COP
axs[0].plot(time, COP, label='COP', color='tab:green')
axs[0].plot(time, Ex_eff, label='Ex_eff')

axs[0].set_ylabel("COP", fontsize=label_fontsize)
axs[0].tick_params(axis='both', labelsize=tick_fontsize)
axs[0].legend(fontsize=legend_fontsize)
axs[0].grid(True)

# 2. Powers
axs[1].plot(time, Pcomb, label='Combustion')
axs[1].plot(time, Pheat_out, label='Heat output')
axs[1].plot(time, Pevap, label='Evaporation')
axs[1].set_ylabel("Power [W]", fontsize=label_fontsize)
axs[1].tick_params(axis='both', labelsize=tick_fontsize)
axs[1].legend(fontsize=legend_fontsize)
axs[1].grid(True)

# 3. Temperatures (dual axis)
# 3. Heater Temperatures (single y-axis)
axs[2].plot(time, Theater1, label='Heater1', linestyle='-', color='tab:red')
axs[2].plot(time, Theater2, label='Heater2', linestyle='-', color='tab:orange')
axs[2].set_ylabel("Heater Temp [°C]", fontsize=label_fontsize)
axs[2].tick_params(axis='both', labelsize=tick_fontsize)
axs[2].legend(loc='upper right', fontsize=legend_fontsize)
axs[2].grid(True)


# 4. Pressure
axs[3].plot(time, pc, label='High', linestyle='-.')
axs[3].plot(time, pe, label='Low', linestyle='-.')
axs[3].plot(time, pi, label='Intermediary', linestyle='-.')
axs[3].set_ylabel("Pressure [Bar]", fontsize=label_fontsize)
axs[3].tick_params(axis='both', labelsize=tick_fontsize)
axs[3].legend(fontsize=legend_fontsize)
axs[3].grid(True)

# 5. Valve openings
axs[4].plot(time, Lpev, label='Low-pressure', color='tab:blue')
axs[4].plot(time, Hpev, label='High-pressure', color='tab:red')
axs[4].set_ylabel("Valve opening [%]", fontsize=label_fontsize)
axs[4].set_xlabel("Time [s]", fontsize=label_fontsize)
axs[4].tick_params(axis='both', labelsize=tick_fontsize)
axs[4].legend(fontsize=legend_fontsize)
axs[4].grid(True)

# # 4. Input: Lpev
# axs[4].plot(time, omegab, label='Burner fan', color='tab:brown')
# # axs[4].plot(time, omega1, label='Motor1', color='tab:blue')
# # axs[4].plot(time, omega2, label='Motor2', color='tab:red')
# # axs[4].plot(time, omega3, label='Motor3', color='tab:orange')
# axs[4].set_ylabel("Speed [rpm]")
# axs[4].set_xlabel("Time [s]")
# # axs[4].set_title("Input Signal")
# axs[4].legend()
# axs[4].grid(True)

# # 4. Input: Lpev
# axs[4].plot(time, Tw_in, label='Water inlet', color='tab:red')
# axs[4].plot(time, Tmpg_in, label='MPG inlet', color='tab:blue')
# # axs[4].plot(time, omega3, label='Motor3', color='tab:orange')
# axs[4].set_ylabel("Temperature [°C]")
# axs[4].set_xlabel("Time [s]")
# # axs[4].set_title("Input Signal")
# axs[4].legend()
# axs[4].grid(True)

# # 4. Input: Lpev
# axs[4].plot(time, mw_dot, label='Water', color='tab:red')
# axs[4].plot(time, mmpg_dot, label='MPG', color='tab:blue')
# # axs[4].plot(time, omega3, label='Motor3', color='tab:orange')
# axs[4].set_ylabel("Mass flow rate [Kg/s]")
# axs[4].set_xlabel("Time [s]")
# # axs[4].set_title("Input Signal")
# axs[4].legend()
# axs[4].grid(True)
plt.tight_layout()

fig2 = plt.figure()
plt.plot(time, Pm1)
plt.plot(time, Pm2)
plt.plot(time, Pm3)
plt.ylim([-20, 300])
plt.show()
