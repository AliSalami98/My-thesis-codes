import csv
import matplotlib.pyplot as plt
import numpy as np
from utils import get_state, CP, h0, T0, s0, p0
data = {
    "Tsuc [K]": [], "psuc [pa]": [], "pdis [pa]": [],
    "Theater [K]": [], "Tw_in [K]": [], "omega [rpm]": [], "omegab [rpm]": [],
    "mdot [kg/s]": [], "Pcomb [W]": [], "Pheating [W]": [],
    "Pcooling [W]": [], "Pmotor [W]": [], "Tdis [K]": [],
    "Pcomp [W]": [], "eff [%]": [], "Ploss [W]": [], "Pmech [W]": []
}

eff_Ex = []
i = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\TC experiments\p60.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        data['Tsuc [K]'].append(float(row['Tsuc [°C]']) + 273.15)
        data['psuc [pa]'].append(float(row['psuc [bar]']) * 1e5)
        data['pdis [pa]'].append(float(row['pdis [bar]']) * 1e5)
        data['Theater [K]'].append(float(row['Theater [°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        # data['omegab [rpm]'].append(float(row['omegab [rpm]']))
        data['mdot [kg/s]'].append(float(row['mdot [g/s]']) * 1e-3)
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Pmech [W]'].append(float(row['Pmotor [W]']) * 0.9)
        data['Tdis [K]'].append(float(row['Tdis [°C]']) + 273.15)
        data['Pcomp [W]'].append(float(row['Pcomp [W]']))
        # data['eff [%]'].append(100 * float(row['eff [%]']))
        data['Ploss [W]'].append(float(row['Ploss [W]']))


        mw_dot = 10/60
        Theater = data['Theater [K]'][-1]
        omega = data['omega [rpm]'][-1]
        Twi = data['Tw_in [K]'][-1]
        T2 = data['Tdis [K]'][-1]
        p2 = data['pdis [pa]'][-1]
        mdot = data['mdot [kg/s]'][-1]
        state = get_state(CP.PT_INPUTS, p2, T2)
        h2, s2 = state.hmass(), state.smass()

        T1 = data['Tsuc [K]'][-1]
        p1 = data['psuc [pa]'][-1]
        state = get_state(CP.PT_INPUTS, p1, T1)
        h1, s1 = state.hmass(), state.smass()
        # --- Compression power and exergy output ---
        Pcomp = data['Pcomp [W]'][-1]
        Pcool  = data['Pcooling [W]'][-1]
        Pheat  = data['Pheating [W]'][-1]
        Pmech  = data['Pmech [W]'][-1]
        psi1 = (h1 - h0) - T0 * (s1 - s0)
        psi2 = (h2 - h0) - T0 * (s2 - s0)
        Ex_comp = mdot * (psi2 - psi1)

        # --- Water side (cooling) ---
        hwi = CP.PropsSI('H', 'P', p0, 'T', Twi, 'Water')
        swi = CP.PropsSI('S', 'P', p0, 'T', Twi, 'Water')
        hwo = hwi + Pcool / mw_dot
        swo = CP.PropsSI('S', 'P', p0, 'H', hwo, 'Water')

        psi_wi = (hwi - h0) - T0 * (swi - s0)
        psi_wo = (hwo - h0) - T0 * (swo - s0)
        Ex_cooler = mw_dot * (psi_wo - psi_wi)

        # --- Exergy terms ---
        Ex_heater = (1 - T0 / Theater) * Pheat
        X_flow   = mdot * T0 * (s2 - s1)
        X_transfer  = -T0 / Theater * Pheat + T0 / Twi * Pcool
        X_heater    = T0 / Theater * Pheat
        X_cooler    = T0 / Twi * Pcool
        X_total     = Ex_heater + Pmech - Ex_comp - Ex_cooler

        # --- Efficiencies ---
        eff_Ex.append(100 * (1 - X_total / (Ex_heater + Pmech)))
        # i+= 1
        # if i > 26:
        #     break
Pr =[s/r for s,r in zip(data['pout [pa]'], data['pin [pa]'])] 

import os
save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_data\omega_Pr'
os.makedirs(save_dir, exist_ok=True)


# Convert temperature and mass flow rate
data['Tout [°C]'] = np.array(data['Tout [K]']) - 273.15
# data['mdot [kg/s]'] = np.array(data['mdot [g/s]']) / 1000

# Define colors and labels for all figures
colors = ['g', 'b', 'r', 'k', 'gray']
labels_exp = ['150 rpm', '180', '210 rpm', '240 rpm', 'max']

# Figure 1: Pheating
plt.figure()
for i in range(5):
    start, end = i * 5, (i + 1) * 5
    plt.plot(Pr[start:end], eff_Ex[start:end], color=colors[i])
    plt.scatter(Pr[start:end], eff_Ex[start:end], color=colors[i], label=labels_exp[i])
plt.xlabel('Pressure Ratio $p_r$ [-]', fontsize=14)
plt.ylabel(r'Exergy Efficiency $\eta_\text{TC, ex}$ [%]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Ex_eff_omega_Pr.eps"), format='eps')

plt.show()





# Figure 1: Pheating
# plt.figure()
# for i in range(3):
#     start, end = i * 5, (i + 1) * 5
#     plt.plot(Pr[start:end], data['Pheating [kW]'][start:end], color=colors[i])
#     plt.scatter(Pr[start:end], data['Pheating [kW]'][start:end], color=colors[i], label=labels_exp[i])
# plt.xlabel('Pressure Ratio $p_r$ [-]', fontsize=14)
# plt.ylabel('Heating Power $\dot{Q}_{heater}$ [kW]', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.grid(True)
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "Qheater_omega_Pr.eps"), format='eps')

# # Figure 2: Pcooling
# plt.figure()
# for i in range(3):
#     start, end = i * 5, (i + 1) * 5
#     plt.plot(Pr[start:end], data['Pcooling [kW]'][start:end], color=colors[i])
#     plt.scatter(Pr[start:end], data['Pcooling [kW]'][start:end], color=colors[i])
# plt.xlabel('Pressure Ratio $p_r$ [-]', fontsize=14)
# plt.ylabel('Cooling Power $\dot{Q}_{cooler}$ [kW]', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "Qcooler_omega_Pr.eps"), format='eps')
# # Figure 3: Pmotor
# plt.figure()
# for i in range(3):
#     start, end = i * 5, (i + 1) * 5
#     plt.plot(Pr[start:end], data['Pmech [kW]'][start:end], color=colors[i])
#     plt.scatter(Pr[start:end], data['Pmech [kW]'][start:end], color=colors[i])
# plt.xlabel('Pressure Ratio $p_r$ [-]', fontsize=14)
# plt.ylabel('Mechanical Power $\dot{W}_{mech}$ [kW]', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "Pmech_omega_Pr.eps"), format='eps')
# # Figure 4: Tout (converted to °C)
# plt.figure()
# for i in range(3):
#     start, end = i * 5, (i + 1) * 5
#     plt.plot(Pr[start:end], data['Tout [°C]'][start:end], color=colors[i])
#     plt.scatter(Pr[start:end], data['Tout [°C]'][start:end], color=colors[i])
# plt.xlabel('Pressure Ratio $p_r$ [-]', fontsize=14)
# plt.ylabel('Discharge Temperature $T_{dis}$ [°C]', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "Tout_omega_Pr.eps"), format='eps')
# # Figure 5: mdot (converted to kg/s)
# plt.figure()
# for i in range(3):
#     start, end = i * 5, (i + 1) * 5
#     plt.plot(Pr[start:end], data['mdot [g/s]'][start:end], color=colors[i])
#     plt.scatter(Pr[start:end], data['mdot [g/s]'][start:end], color=colors[i])
# plt.xlabel('Pressure Ratio $p_r$ [-]', fontsize=14)
# plt.ylabel('Mass Flow Rate $\dot{m}_f$ [g/s]', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "mdot_omega_Pr.eps"), format='eps')

# # Figure 6: alpha
# plt.figure()
# for i in range(3):
#     start, end = i * 5, (i + 1) * 5
#     plt.plot(Pr[start:end], data['Ploss [kW]'][start:end], color=colors[i])
#     plt.scatter(Pr[start:end], data['Ploss [kW]'][start:end], color=colors[i], s=60, marker=['o', '^', 's'][i])
# plt.xlabel('Pressure Ratio $p_r$ [-]', fontsize=14)
# plt.ylabel('Heat loss $\dot{Q}_{loss}$ [kW]', fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "Qloss_omega_Pr.eps"), format='eps')

# plt.show()
