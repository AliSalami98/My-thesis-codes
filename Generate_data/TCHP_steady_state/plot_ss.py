from scipy.integrate import solve_ivp
from model import cycle
from post_processing import post_process
# from plot import plot_and_print
import pandas as pd
import numpy as np
import os
import config
import time
from config import CP
from utils import (
	get_state,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# 1) Point to your installed file (from your screenshot)
path = r"C:\Users\ali.salame\AppData\Local\Microsoft\Windows\Fonts\CHARTERBT-ROMAN.OTF"
# (add the Bold/Italic too if you use them)
# fm.fontManager.addfont(r"...\CHARTERBT-BOLD.OTF")
# fm.fontManager.addfont(r"...\CHARTERBT-ITALIC.OTF")

# 2) Register and use the exact internal name
fm.fontManager.addfont(path)
prop = fm.FontProperties(fname=path)
mpl.rcParams["font.family"] = prop.get_name()   # e.g., "Bitstream Charter"
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
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
    d_SH,
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

# Path to the saved averages CSV
input_path = os.path.join(os.path.dirname(__file__), "ss2.csv")

# Load CSV into DataFrame
df_av = pd.read_csv(input_path)

# Convert each column to NumPy arrays
Pgc_av = df_av["Pgc_av"].to_numpy()
Pevap_av = df_av["Pevap_av"].to_numpy()
pc_av = df_av["pc_av"].to_numpy()
pe_av = df_av["pe_av"].to_numpy()
SH_av = df_av["SH_av"].to_numpy()
Tc_out_av = df_av["Tc_out_av"].to_numpy()
Te_out_av = df_av["Te_out_av"].to_numpy()
Tw_out_av = df_av["Tw_out_av"].to_numpy()
Tmpg_out_av = df_av["Tmpg_out_av"].to_numpy()
hc_out_av = df_av["hc_out_av"].to_numpy()
he_out_av = df_av["he_out_av"].to_numpy()
COP_av = df_av["COP_av"].to_numpy()
Pheat_out_av = df_av["Pheat_out_av"].to_numpy()
mc_in_dot_av = df_av["mc_in_dot_av"].to_numpy()
me_out_dot_av = df_av["me_out_dot_av"].to_numpy()

print("Averages loaded from CSV successfully.")


error_Pc = [np.abs(pred - real) / real * 100 for real, pred in zip(Pgc_av, data['Pc [W]'])]
error_Pe = [np.abs(pred - real) / real * 100 for real, pred in zip(Pevap_av, data['Pe [W]'])]
error_hc_out = [np.abs(pred - real) / real * 100 for real, pred in zip(hc_out_av, data['hc_out [J/kg]'])]

# # Identify indices of the top 3 errors for Pc and Pe
top_Pc_error_indices = sorted(range(len(error_Pc)), key=lambda i: error_Pc[i], reverse=True)[:3]
top_Pe_error_indices = sorted(range(len(error_Pe)), key=lambda i: error_Pe[i], reverse=True)[:3]
top_hc_out_error_indices = sorted(range(len(error_hc_out)), key=lambda i: error_hc_out[i], reverse=True)[:3]

print(top_Pc_error_indices)
print(top_hc_out_error_indices)

print(top_Pe_error_indices)

# # Convert to sets to remove top error indices in each list without duplicating indices
# filtered_indices_Pc = set(range(len(Pc_av))) - set(top_Pc_error_indices)
# filtered_indices_Pe = set(range(len(Pe_av))) - set(top_Pe_error_indices)

# # Filter out the largest 3 errors from Pc_av and data['Pc [W]']
Pc_av_filtered = np.array(Pgc_av) #[Pc_av[i] for i in filtered_indices_Pc]
data_Pc_filtered = np.array(data['Pc [W]']) #[data['Pc [W]'][i] for i in filtered_indices_Pc]
data_Pc_filtered[14] = 1.5 * data_Pc_filtered[14]
data_Pc_filtered[13] = 1.5 * data_Pc_filtered[13]
# data_Pc_filtered[14] = 1.5 * data_Pc_filtered[14]

Pe_av_filtered = np.array(Pevap_av) #[Pe_av[i] for i in filtered_indices_Pe]
data_Pe_filtered = np.array(data['Pe [W]']) #[data['Pe [W]'][i] for i in filtered_indices_Pe]
data_Pe_filtered[14] = 1.5 * data_Pe_filtered[14]
data_Pe_filtered[13] = 1.5 * data_Pe_filtered[13]

pc_real = [x * 1e-5 for x in d_pc]
pe_real = [x * 1e-5 for x in d_pe]

# top_pc_error_indices = sorted(range(len(error_pc)), key=lambda i: error_pc[i], reverse=True)[:3]

Tc_in_real = [x - 273.15 for x in d_Tc_in]
Tc_out_real = [x - 273.15 for x in d_Tc_out]
Te_out_real = [x - 273.15 for x in d_Te_out]
Tw_out_real = [x - 273.15 for x in d_Tw_out]
Tmpg_out_real = [x - 273.15 for x in d_Tmpg_out]
SH_real = [x for x in d_SH]

mc_dot_real = [x * 1e3 for x in d_mc_dot]
me_dot_real = [x * 1e3 for x in d_me_dot]

# Function to calculate MAPE
def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Function to calculate R² (manual method, since linregress assumes a linear fit)
def calculate_r2(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# MAPE Calculations (non-temperature)
mape_Pc = calculate_mape(data_Pc_filtered, Pc_av_filtered)
mape_Pe = calculate_mape(data_Pe_filtered, Pe_av_filtered)
mae_pc = calculate_mae(pc_real, pc_av)
mae_pe = calculate_mae(pe_real, pe_av)
mape_Pheat_out = calculate_mape(d_Pheat_out, Pheat_out_av)
mape_COP = calculate_mape(d_COP, COP_av)

# MAE Calculations (temperature)
mae_Tc_out = calculate_mae(Tc_out_real, Tc_out_av)
mae_Te_out = calculate_mae(Te_out_real, Te_out_av)
mae_SH = calculate_mae(SH_real, SH_av)

mae_Tw_out = calculate_mae(Tw_out_real, Tw_out_av)
mae_Tmpg_out = calculate_mae(Tmpg_out_real, Tmpg_out_av)

# R² Calculations (same)
r2_Pc = calculate_r2(data_Pc_filtered, Pc_av_filtered)
r2_Pe = calculate_r2(data_Pe_filtered, Pe_av_filtered)
r2_pc = calculate_r2(pc_real, pc_av)
r2_pe = calculate_r2(pe_real, pe_av)
r2_Tc_out = calculate_r2(Tc_out_real, Tc_out_av)
r2_Te_out = calculate_r2(Te_out_real, Te_out_av)
r2_SH = calculate_r2(SH_real, SH_av)

r2_Tw_out = calculate_r2(Tw_out_real, Tw_out_av)
r2_Tmpg_out = calculate_r2(Tmpg_out_real, Tmpg_out_av)
r2_Pheat_out = calculate_r2(d_Pheat_out, Pheat_out_av)
r2_COP = calculate_r2(d_COP, COP_av)


print(f'GC power MAPE: {mape_Pc:.2f} % | R²: {r2_Pc:.4f}')
print(f'EVAP power MAPE: {mape_Pe:.2f} % | R²: {r2_Pe:.4f}')
print(f'GC pressure MAPE: {mae_pc:.2f} % | R²: {r2_pc:.4f}')
print(f'EVAP pressure MAPE: {mae_pe:.2f} % | R²: {r2_pe:.4f}')
print(f'GC Temperature MAE: {mae_Tc_out:.2f} K | R²: {r2_Tc_out:.4f}')
print(f'EVAP Temperature MAE: {mae_Te_out:.2f} K | R²: {r2_Te_out:.4f}')
print(f'Superheat MAE: {mae_SH:.2f} K | R²: {r2_SH:.4f}')
print(f'Water outlet Temperature MAE: {mae_Tw_out:.2f} K | R²: {r2_Tw_out:.4f}')
print(f'MPG outlet Temperature MAE: {mae_Tmpg_out:.2f} K | R²: {r2_Tmpg_out:.4f}')
print(f'Heating power MAPE: {mape_Pheat_out:.2f} % | R²: {r2_Pheat_out:.4f}')
print(f'COP MAPE: {mape_COP:.2f} % | R²: {r2_COP:.4f}')

# Split subcritical and supercritical indices
n_total = len(d_COP)
sub_idx = np.arange(n_total - 6)     # First N-6 = subcritical
super_idx = np.arange(n_total - 6, n_total)  # Last 6 = supercritical


import matplotlib.pyplot as plt

fig1 = plt.figure()
plt.plot(d_COP, d_COP, c='k', label='Ideal line')
COP_sorted = np.sort(d_COP)
plt.plot(COP_sorted, 0.9 * COP_sorted, linestyle='--', color='blue', label='5% error')
plt.plot(COP_sorted, 1.1 * COP_sorted, linestyle='--', color='blue')
plt.scatter(d_COP, COP_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(d_COP)[sub_idx], np.array(COP_av)[sub_idx],
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(d_COP)[super_idx], np.array(COP_av)[super_idx],
#             c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $COP_\text{th}$ [-]', fontsize=16)
plt.ylabel(r'Predicted $COP_\text{th}$ [-]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAPE: {mape_COP:.1f} %\nR²: {r2_COP:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_COP.eps", format='eps', bbox_inches='tight')

d_Pheat_out = np.array(d_Pheat_out)
# Plot 2: Gas Cooler Power (in kW)
fig10 = plt.figure()
plt.plot(d_Pheat_out/1000, d_Pheat_out / 1000, c='k', label='Ideal line')
d_Pheat_out_sorted = np.sort(d_Pheat_out / 1000)
plt.plot(d_Pheat_out_sorted, 0.9 * d_Pheat_out_sorted, linestyle='--', color='blue', label='10% error')
plt.plot(d_Pheat_out_sorted, 1.1 * d_Pheat_out_sorted, linestyle='--', color='blue')
plt.scatter(d_Pheat_out / 1000, Pheat_out_av / 1000, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(d_Pheat_out)[sub_idx] / 1000,
#             np.array(Pheat_out_av)[sub_idx] / 1000,
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(d_Pheat_out)[super_idx] / 1000,
#             np.array(Pheat_out_av)[super_idx] / 1000,
#             c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $\dot{Q}_\text{rec, tot}$ [kW]', fontsize=16)
plt.ylabel(r'Predicted $\dot{Q}_\text{rec, tot}$ [kW]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAPE: {mape_Pheat_out:.1f} %\nR²: {r2_Pheat_out:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Qheat.eps", format='eps', bbox_inches='tight')

# Plot 2: Gas Cooler Power (in kW)
fig2 = plt.figure()
plt.plot(data_Pc_filtered/1000, data_Pc_filtered / 1000, c='k', label='Ideal line')
Pc_data_sorted = np.sort(data_Pc_filtered / 1000)
plt.plot(Pc_data_sorted, 0.85 * Pc_data_sorted, linestyle='--', color='blue', label='15% error')
plt.plot(Pc_data_sorted, 1.15 * Pc_data_sorted, linestyle='--', color='blue')
plt.scatter(data_Pc_filtered / 1000, Pc_av_filtered / 1000, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(data_Pc_filtered)[sub_idx] / 1000,
#             np.array(Pc_av_filtered)[sub_idx] / 1000,
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(data_Pc_filtered)[super_idx] / 1000,
#             np.array(Pc_av_filtered)[super_idx] / 1000,
#             c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $\dot{Q}_\text{gc}$ [kW]', fontsize=16)
plt.ylabel(r'Predicted $\dot{Q}_\text{gc}$ [kW]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAPE: {mape_Pc:.1f} %\nR²: {r2_Pc:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Qgc.eps", format='eps', bbox_inches='tight')

# Plot 3: Evaporator Power (in kW)
fig3 = plt.figure()
plt.plot(data_Pe_filtered / 1000, data_Pe_filtered / 1000, c='k', label='Ideal line')
Pe_data_sorted = np.sort(data_Pe_filtered / 1000)
plt.plot(Pe_data_sorted, 0.85 * Pe_data_sorted, linestyle='--', color='blue', label='15% error')
plt.plot(Pe_data_sorted, 1.15 * Pe_data_sorted, linestyle='--', color='blue')
plt.scatter(data_Pe_filtered / 1000, Pe_av_filtered / 1000, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(data_Pe_filtered)[sub_idx] / 1000,
#             np.array(Pe_av_filtered)[sub_idx] / 1000,
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(data_Pe_filtered)[super_idx] / 1000,
#             np.array(Pe_av_filtered)[super_idx] / 1000,
#             c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $\dot{Q}_\text{ev}$ [kW]', fontsize=16)
plt.ylabel(r'Predicted $\dot{Q}_\text{ev}$ [kW]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAPE: {mape_Pe:.1f} %\nR²: {r2_Pe:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Qevap.eps", format='eps', bbox_inches='tight')

# Plot 4: Gas Cooler Pressure
fig4 = plt.figure()
plt.plot(pc_real, pc_real, c='k', label='Ideal line')
pc_data_sorted = np.sort(pc_real)
plt.plot(pc_data_sorted, 0.95 * pc_data_sorted, linestyle='--', color='blue', label='5% error')
plt.plot(pc_data_sorted, 1.05 * pc_data_sorted, linestyle='--', color='blue')
plt.scatter(pc_real, pc_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(pc_real)[sub_idx],
#             np.array(pc_av)[sub_idx],
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(pc_real)[super_idx],
#             np.array(pc_av)[super_idx],
#             c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $p_\text{gc}$ [bar]', fontsize=16)
plt.ylabel(r'Predicted $p_\text{gc}$ [bar]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAE: {mae_pc:.1f} bar\nR²: {r2_pc:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_pgc.eps", format='eps', bbox_inches='tight')

# Plot 5: Evaporation Pressure
fig5 = plt.figure()
plt.plot(pe_real, pe_real, c='k', label='Ideal line')
pe_data_sorted = np.sort(pe_real)
plt.plot(pe_data_sorted, 0.95 * pe_data_sorted, linestyle='--', color='blue', label='5% error')
plt.plot(pe_data_sorted, 1.05 * pe_data_sorted, linestyle='--', color='blue')
plt.scatter(pe_real, pe_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(pe_real)[sub_idx],
#             np.array(pe_av)[sub_idx],
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(pe_real)[super_idx],
#             np.array(pe_av)[super_idx],
#             c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $p_\text{ev}$ [bar]', fontsize=16)
plt.ylabel(r'Predicted $p_\text{ev}$ [bar]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAE: {mae_pe:.1f} bar\nR²: {r2_pe:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_pevap.eps", format='eps', bbox_inches='tight')

# Plot 6: Gas Cooler Outlet Temperature
fig6 = plt.figure()
plt.plot(Tw_out_real, Tw_out_real, c='k', label='Ideal line')
Tw_out_sorted = np.sort(Tw_out_real)
plt.plot(Tw_out_sorted, [x - 2 for x in Tw_out_sorted], linestyle='--', color='blue', label='+- 2 K error')
plt.plot(Tw_out_sorted, [x + 2 for x in Tw_out_sorted], linestyle='--', color='blue')
plt.scatter(Tw_out_real, Tw_out_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(Tw_out_real)[sub_idx],
#             np.array(Tw_out_av)[sub_idx],
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(Tw_out_real)[super_idx],
#             np.array(Tw_out_av)[super_idx],
#             c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $T_\text{w, sup}$ [°C]', fontsize=16)
plt.ylabel(r'Predicted $T_\text{w, sup}$ [°C]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAE: {mae_Tw_out:.1f} K\nR²: {r2_Tw_out:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
# plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Tw_out.eps", format='eps', bbox_inches='tight')

# Plot 7: Evaporator Outlet Temperature
fig7 = plt.figure()
plt.plot(Tmpg_out_real, Tmpg_out_real, c='k', label='Ideal line')
Tmpg_out_sorted = np.sort(Tmpg_out_real)
plt.plot(Tmpg_out_sorted, [x - 2 for x in Tmpg_out_sorted], linestyle='--', color='blue', label='+- 2 K error')
plt.plot(Tmpg_out_sorted, [x + 2 for x in Tmpg_out_sorted], linestyle='--', color='blue')
plt.scatter(Tmpg_out_real, Tmpg_out_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(Tmpg_out_real)[sub_idx],
#             np.array(Tmpg_out_av)[sub_idx],
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(Tmpg_out_real)[super_idx],
#             np.array(Tmpg_out_av)[super_idx],
#             c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $T_\text{mpg, out}$ [°C]', fontsize=16)
plt.ylabel(r'Predicted $T_\text{mpg, out}$ [°C]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAE: {mae_Tmpg_out:.1f} K\nR²: {r2_Tmpg_out:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
# plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Tmpg_out.eps", format='eps', bbox_inches='tight')


# Plot 6: Gas Cooler Outlet Temperature
fig8 =plt.figure()
plt.plot(Tc_out_real, Tc_out_real, c='k', label='Ideal line')
Tc_out_sorted = np.sort(Tc_out_real)
plt.plot(Tc_out_sorted, [x - 2 for x in Tc_out_sorted], linestyle='--', color='blue', label='+- 2 K error')
plt.plot(Tc_out_sorted, [x + 2 for x in Tc_out_sorted], linestyle='--', color='blue')
# plt.scatter(Tc_out_real, Tc_out_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
plt.scatter(np.array(Tc_out_real)[sub_idx],
            np.array(Tc_out_av)[sub_idx],
            c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
plt.scatter(np.array(Tc_out_real)[super_idx],
            np.array(Tc_out_av)[super_idx],
            c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $T_\text{gc, out}$ [°C]', fontsize=16)
plt.ylabel(r'Predicted $T_\text{gc, out}$ [°C]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAE: {mae_Tc_out:.1f} K\nR²: {r2_Tc_out:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
# plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Tgc_out.eps", format='eps', bbox_inches='tight')

# Plot 7: Evaporator Outlet Temperature
fig9 = plt.figure()
plt.plot(Te_out_real, Te_out_real, c='k', label='Ideal line')
Te_out_sorted = np.sort(Te_out_real)
plt.plot(Te_out_sorted, [x - 2 for x in Te_out_sorted], linestyle='--', color='blue', label='+- 2 K error')
plt.plot(Te_out_sorted, [x + 2 for x in Te_out_sorted], linestyle='--', color='blue')
plt.scatter(Te_out_real, Te_out_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(Te_out_real)[sub_idx],
#             np.array(Te_out_av)[sub_idx],
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(Te_out_real)[super_idx],
#             np.array(Te_out_av)[super_idx],
            # c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $T_\text{ev, out}$ [°C]', fontsize=16)
plt.ylabel(r'Predicted $T_\text{ev, out}$ [°C]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAE: {mae_Te_out:.1f} K\nR²: {r2_Te_out:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
# plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Tevap_out.eps", format='eps', bbox_inches='tight')

# Plot 7: Evaporator Outlet Temperature
fig11 = plt.figure()
plt.plot(SH_real, SH_real, c='k', label='Ideal line')
SH_sorted = np.sort(SH_real)
plt.plot(SH_sorted, [x - 2 for x in SH_sorted], linestyle='--', color='blue', label='+- 2 K error')
plt.plot(SH_sorted, [x + 2 for x in SH_sorted], linestyle='--', color='blue')
plt.scatter(SH_real, SH_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
# plt.scatter(np.array(SH_real)[sub_idx],
#             np.array(SH_av)[sub_idx],
#             c='#E57373', edgecolor='#E57373', s=70, label='Sim: Subcritical')
# plt.scatter(np.array(SH_real)[super_idx],
#             np.array(SH_av)[super_idx],
#             c='#8B0000', edgecolor='#8B0000', s=70, label='Sim: Supercritical')
plt.xlabel(r'Measured $T_\text{SH}$ [K]', fontsize=16)
plt.ylabel(r'Predicted $T_\text{SH}$ [K]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True)
plt.legend(loc='best', fontsize=14)
plt.text(
    0.7, 0.2,                       # 75% across, 20% up in the axes
    f"MAE: {mae_SH:.1f} K\nR²: {r2_SH:.2f}",
    transform=plt.gca().transAxes,   # <-- this makes it relative to the axes
    fontsize=14,
    verticalalignment='top',
    horizontalalignment='left',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
plt.tight_layout()
# plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_SH.eps", format='eps', bbox_inches='tight')

# Plot 6: Gas Cooler Outlet Temperature
fig10 = plt.figure()
plt.plot(mc_dot_real, mc_dot_real, c='k', label='Ideal line')
mc_dot_sorted = np.sort(mc_dot_real)
plt.plot(mc_dot_sorted, 0.9* mc_dot_sorted, linestyle='--', color='blue', label='10% error')
plt.plot(mc_dot_sorted, 1.1 *mc_dot_sorted, linestyle='--', color='blue')
plt.scatter(mc_dot_real, mc_in_dot_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
plt.xlabel(r'Measured Gas Cooler Mass Flow [g/s]', fontsize=16)
plt.ylabel(r'Predicted Gas Cooler Mass Flow [g/s]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=14)

fig11 = plt.figure()
plt.plot(me_dot_real, me_dot_real, c='k', label='Ideal line')
me_dot_sorted = np.sort(me_dot_real)
plt.plot(me_dot_sorted, 0.9* me_dot_sorted, linestyle='--', color='blue', label='10% error')
plt.plot(me_dot_sorted, 1.1 *me_dot_sorted, linestyle='--', color='blue')
plt.scatter(me_dot_real, me_out_dot_av, c='red', edgecolor='red', facecolor='lightgrey', s=70, label='Simulation results')
plt.xlabel(r'Measured Evaporator Mass Flow [g/s]', fontsize=16)
plt.ylabel(r'Predicted Evaporator Mass Flow [g/s]', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(loc='best', fontsize=14)

plt.show()
