import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from data_filling_ss import data, a_Pr

# Load data
zabri = 'omega'
csv_path = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\Exergy\first patch\Edest_results_omega.csv'
df = pd.read_csv(csv_path, sep=';')

save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\Exergy'
os.makedirs(save_dir, exist_ok=True)

# Destroyed exergy components
Pheat_ss = df['Pheat [kW]']
Pcool_ss = df['Pcool [kW]']
W_ss = df['W [kW]']
mdot_ss = np.array(df['mdot [kg/s]']) * 1e3
Tout_ss = df['Tout [C]']
Ex_eff_ss = df['Ex_eff [%]']

Pheat_real = np.array(data['Pheating [W]'])/1e3
Pcool_real = np.array(data['Pcooling [W]'])/1e3
Pmech_real = np.array(data['Pmech [W]'])/1e3
Pout_real = np.array(data['Pcomp [W]'])/1e3
Tdis_real = np.array(data['Tdis [K]']) - 273.15
mdot_real = np.array(data['mdot [g/s]'])
Ex_eff_real = np.array(data['Ex_eff [%]'])

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
MAPE_Ex_eff = [100 * np.abs(x - y)/np.abs(y) for x, y in zip(Ex_eff_ss, Ex_eff_real)]
print('MAPE of Ex_eff', np.mean(MAPE_Ex_eff))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

Pheating_uncertainty = [0.05*x for x in Pheat_real]

fig1 = plt.figure(1)
plt.scatter(a_Pr[0:5], Pheat_real[0:5], c='g', marker='o', s=60)
plt.plot(a_Pr[0:5], Pheat_ss[0:5], linestyle=':', c='g', linewidth=2.5)

plt.scatter(a_Pr[5:10], Pheat_real[5:10], c='b', marker='^', s=60)
plt.plot(a_Pr[5:10], Pheat_ss[5:10], linestyle='--', c='b', linewidth=2.5)

plt.scatter(a_Pr[10:15], Pheat_real[10:15], c='r', marker='s', s=60)
plt.plot(a_Pr[10:15], Pheat_ss[10:15], linestyle='-', c='r', linewidth=2.5)

plt.xlabel(r'Pressure ratio $r_\text{p}$ [-]', fontsize=14)
plt.ylabel(r'Heater heat transfer rate $\dot{Q}_\text{h}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.grid(True)
# Define custom legend
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"Qheater_{zabri}.eps"), format='eps')


Pcooling_uncertainty = [0.05*x for x in Pcool_real]
fig2 = plt.figure(2)
plt.scatter(a_Pr[0:5], Pcool_real[0:5], c='g', marker='o', s=60)
plt.plot(a_Pr[0:5], Pcool_ss[0:5], linestyle=':', c='g', linewidth=2.5)

plt.scatter(a_Pr[5:10], Pcool_real[5:10], c='b', marker='^', s=60)
plt.plot(a_Pr[5:10], Pcool_ss[5:10], linestyle='--', c='b', linewidth=2.5)

plt.scatter(a_Pr[10:15], Pcool_real[10:15], marker='s', c='r')
plt.plot(a_Pr[10:15], Pcool_ss[10:15], c='r', linewidth=2.5)

plt.xlabel(r'Pressure ratio $r_\text{p}$ [-]', fontsize=14)
plt.ylabel(r'Cooler heat transfer rate $\dot{Q}_\text{k}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"Qcooler_{zabri}.eps"), format='eps')

# plt.legend()

Pmech_uncertainty = [0.05*x for x in Pmech_real]
fig3 = plt.figure(3)
plt.scatter(a_Pr[0:5], Pmech_real[0:5], c='g', marker='o', s=60)
plt.plot(a_Pr[0:5], W_ss[0:5], linestyle=':', c='g', linewidth=2.5)

plt.scatter(a_Pr[5:10], Pmech_real[5:10], c='b', marker='^', s=60)
plt.plot(a_Pr[5:10], W_ss[5:10], linestyle='--', c='b', linewidth=2.5)

plt.scatter(a_Pr[10:15], Pmech_real[10:15], marker='s', c='r')
plt.plot(a_Pr[10:15], W_ss[10:15], c='r', linewidth=2.5)

plt.xlabel(r'Pressure ratio $r_\text{p}$ [-]', fontsize=14)
plt.ylabel(r'Mechanical power $\dot{W}_\text{mech}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"Wmech_{zabri}.eps"), format='eps')


# plt.legend()

T_error = 1
fig4 = plt.figure(4)
plt.scatter(a_Pr[0:5], Tdis_real[0:5], c='g', marker='o', s=60)
plt.plot(a_Pr[0:5], Tout_ss[0:5], linestyle=':', c='g', linewidth=2.5)

plt.scatter(a_Pr[5:10], Tdis_real[5:10], c='b', marker='^', s=60)
plt.plot(a_Pr[5:10], Tout_ss[5:10], linestyle='--', c='b', linewidth=2.5)

plt.scatter(a_Pr[10:15], Tdis_real[10:15], marker='s', c='r')
plt.plot(a_Pr[10:15], Tout_ss[10:15], c='r', linewidth=2.5)

plt.xlabel(r'Pressure ratio $r_\text{p}$ [-]', fontsize=14)
plt.ylabel(r'Discharge temperature $T_\text{dis}$ [°C]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"Tdis_{zabri}.eps"), format='eps')


mdot_uncertainty = [0.05*x for x in mdot_real]
fig5 = plt.figure(5)
plt.scatter(a_Pr[0:5], mdot_real[0:5], c='g', marker='o', s=60)
plt.plot(a_Pr[0:5], mdot_ss[0:5], linestyle=':', c='g', linewidth=2.5)

plt.scatter(a_Pr[5:10], mdot_real[5:10], c='b', marker='^', s=60)
plt.plot(a_Pr[5:10], mdot_ss[5:10], linestyle='--', c='b', linewidth=2.5)

plt.scatter(a_Pr[10:15], mdot_real[10:15], marker='s', c='r')
plt.plot(a_Pr[10:15], mdot_ss[10:15], c='r', linewidth=2.5)

plt.xlabel(r'Pressure ratio $r_\text{p}$ [-]', fontsize=14)
plt.ylabel(r'Mass flow rate $\dot{m}_\text{f}$ [g/s]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.grid(True)
if zabri == 'pcharged':
    custom_lines = [
        Line2D([0], [0], color='g', marker='o', linestyle='None', markersize=8, label='30 bar exp'),
        Line2D([0], [0], color='g', linestyle=':', linewidth=2.5, label='30 bar sim'),
        Line2D([0], [0], color='b', marker='^', linestyle='None', markersize=8, label='40 bar exp'),
        Line2D([0], [0], color='b', linestyle='--', linewidth=2.5, label='40 bar sim'),
        Line2D([0], [0], color='r', marker='s', linestyle='None', markersize=8, label='56 bar exp'),
        Line2D([0], [0], color='r', linestyle='-', linewidth=2.5, label='56 bar sim'),
    ]
else:
    custom_lines = [
    Line2D([0], [0], color='g', marker='o', linestyle='None', markersize=8, label='150 rpm exp'),
    Line2D([0], [0], color='g', linestyle=':', linewidth=2.5, label='150 rpm sim'),
    Line2D([0], [0], color='b', marker='^', linestyle='None', markersize=8, label='200 rpm exp'),
    Line2D([0], [0], color='b', linestyle='--', linewidth=2.5, label='200 rpm sim'),
    Line2D([0], [0], color='r', marker='s', linestyle='None', markersize=8, label='240 rpm exp'),
    Line2D([0], [0], color='r', linestyle='-', linewidth=2.5, label='240 rpm sim'),
    ]
plt.legend(handles=custom_lines, loc='best')
plt.tight_layout()

plt.savefig(os.path.join(save_dir, f"mdot_{zabri}.eps"), format='eps')


fig6 = plt.figure(6)
plt.scatter(a_Pr[0:5], Ex_eff_real[0:5], c='g', marker='o', s=60)
plt.plot(a_Pr[0:5], Ex_eff_ss[0:5], linestyle=':', c='g', linewidth=2.5)

plt.scatter(a_Pr[5:10], Ex_eff_real[5:10], c='b', marker='^', s=60)
plt.plot(a_Pr[5:10], Ex_eff_ss[5:10], linestyle='--', c='b', linewidth=2.5)

plt.scatter(a_Pr[10:15], Ex_eff_real[10:15], marker='s', c='r')
plt.plot(a_Pr[10:15], Ex_eff_ss[10:15], c='r', linewidth=2.5)

plt.xlabel(r'Pressure ratio $r_\text{p}$ [-]', fontsize=14)
plt.ylabel(r'Exergy efficiency $\eta_\text{ex}$ [%]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"Ex_eff_{zabri}.eps"), format='eps')

plt.show()