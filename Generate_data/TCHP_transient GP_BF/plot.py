import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from config import CP, total_time_steps
from utils import (
	get_state,
    keyed_output
)
from sklearn.metrics import r2_score, mean_absolute_error
import config
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
def calculate_r2_mape(simulated, experimental):
    r2 = r2_score(experimental, simulated)
    mape = np.mean(np.abs((experimental - simulated) / experimental)) * 100
    return r2, mape

def calculate_r2_mae(simulated, experimental):
    r2 = r2_score(experimental, simulated)
    mae = mean_absolute_error(experimental, simulated)
    return r2, mae

# def average_consecutive_values(data, n=4):
#     averaged_list = []
#     # Loop through the list in steps of 4 (or n)
#     for i in range(0, len(data), n):
#         # Take a slice of 4 (or n) elements
#         subset = data[i:i+n]
#         # Calculate the average and append it to the averaged list
#         averaged_list.append(sum(subset) / len(subset))
#     return averaged_list
# ===== Load saved CSV =====
input_path = os.path.join(os.path.dirname(__file__), "BF.csv")
# Load CSV into DataFrame
df = pd.read_csv(input_path)


# Assign columns to NumPy arrays
t_ss = df["t_ss"].to_numpy()
Tmpg_out_ss = df["Tmpg_out_ss"].to_numpy()
Tw_out_ss = df["Tw_out_ss"].to_numpy()
pc_ss = df["pc_ss"].to_numpy()
hc_out_ss = df["hc_out_ss"].to_numpy()
hihx1_out_ss = df["hihx1_out_ss"].to_numpy()
hft_ss = df["hft_ss"].to_numpy()
hihx2_out_ss = df["hihx2_out_ss"].to_numpy()
he_out_ss = df["he_out_ss"].to_numpy()
pft_ss = df["pft_ss"].to_numpy()
pbuff1_ss = df["pbuff1_ss"].to_numpy()
pbuff2_ss = df["pbuff2_ss"].to_numpy()
pe_ss = df["pe_ss"].to_numpy()
Tc_out_ss = df["Tc_out_ss"].to_numpy()
Te_out_ss = df["Te_out_ss"].to_numpy()
mc_in_dot_ss = df["mc_in_dot_ss"].to_numpy()
mc_out_dot_ss = df["mc_out_dot_ss"].to_numpy()
mft_in_dot_ss = df["mft_in_dot_ss"].to_numpy()
mft_out_dot_ss = df["mft_out_dot_ss"].to_numpy()
mbuff1_in_dot_ss = df["mbuff1_in_dot_ss"].to_numpy()
mbuff1_out_dot_ss = df["mbuff1_out_dot_ss"].to_numpy()
me_in_dot_ss = df["me_in_dot_ss"].to_numpy()
me_out_dot_ss = df["me_out_dot_ss"].to_numpy()
Pheat_out_ss = df["Pheat_out_ss"].to_numpy()
Pgc_ss = df["Pgc_ss"].to_numpy()
Pevap_ss = df["Pevap_ss"].to_numpy()
Prec_total_ss = df["Prec_total_ss"].to_numpy()
COP_ss = df["COP_ss"].to_numpy()
SH_ss = df["SH_ss"].to_numpy()

from data_filling_tr import (
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
    d_Tmpg_in,
    d_Tmpg_out,
    d_Lpev,
    d_Hpev,
    d_pc,
    d_p25,
    d_pi,
    d_pe,
    d_Te_out,
    d_Tc_out,
    d_Pheat_out,
    d_COP,
    d_Pcomb,
    d_Pevap,
    d_SH,
    d_error

)
t_init = 30
d_pc = np.array([x * 10**(-5) for x in d_pc[t_init:total_time_steps]])
d_p25 = np.array([x * 10**(-5) for x in d_p25[t_init:total_time_steps]])
d_pi = np.array([x * 10**(-5) for x in d_pi[t_init:total_time_steps]])
d_pe = np.array([x * 10**(-5) for x in d_pe[t_init:total_time_steps]])
d_Te_out = np.array([x - 273.15 for x in d_Te_out[t_init:total_time_steps]])
d_Tc_out = np.array([x - 273.15 for x in d_Tc_out[t_init:total_time_steps]])
d_Tw_in = np.array([x - 273.15 for x in d_Tw_in[t_init:total_time_steps]])
d_Tw_out = np.array([x - 273.15 for x in d_Tw_out[t_init:total_time_steps]])
d_Tmpg_in = np.array([x - 273.15 for x in d_Tmpg_in[t_init:total_time_steps]])
d_Tmpg_out = np.array([x - 273.15 for x in d_Tmpg_out[t_init:total_time_steps]])
d_mw_dot = np.array([x for x in d_mw_dot[t_init:total_time_steps]]) * 1e3
d_mmpg_dot = np.array([x for x in d_mmpg_dot[t_init:total_time_steps]])
d_Hpev = np.array([x for x in d_Hpev[t_init:total_time_steps]])
d_Lpev = np.array([x for x in d_Lpev[t_init:total_time_steps]])
d_omegab = np.array([x for x in d_omegab[t_init:total_time_steps]])
d_SH = np.array([x for x in d_SH[t_init:total_time_steps]])
d_Theater2 = np.array([x - 273.15 for x in d_Theater2[t_init:total_time_steps]])
d_Theater1 = np.array([x - 273.15 for x in d_Theater1[t_init:total_time_steps]])
d_Pheat_out = np.array([x for x in d_Pheat_out[t_init:total_time_steps]]) * 1e-3
d_COP = np.array([x for x in d_COP[t_init:total_time_steps]])
d_Pevap = np.array([x for x in d_Pevap[t_init:total_time_steps]]) * 1e-3
d_error = np.array([x for x in d_error[t_init:total_time_steps]])

def block_reduce(a: np.ndarray, w: int = 2, agg: str = "mean") -> np.ndarray:
    a = np.asarray(a, float)
    n = len(a)
    if n == 0 or w <= 1:
        return a
    m = (n // w) * w
    core = a[:m].reshape(-1, w)
    if agg == "median":
        out = np.median(core, axis=1)
    else:
        out = np.mean(core, axis=1)
    # handle remainder
    if m < n:
        tail = a[m:]
        tail_val = np.median(tail) if agg == "median" else np.mean(tail)
        out = np.concatenate([out, [tail_val]])
    return out

# === Apply window method (w=2 halves the length) ===
W = 1  # change to any integer window size you want

d_pc        = block_reduce(d_pc,        w=W)
d_p25       = block_reduce(d_p25,       w=W)
d_pi        = block_reduce(d_pi,        w=W)
d_pe        = block_reduce(d_pe,        w=W)

d_Te_out    = block_reduce(d_Te_out,    w=W)
d_Tc_out    = block_reduce(d_Tc_out,    w=W)
d_Tw_in     = block_reduce(d_Tw_in,     w=W)
d_Tw_out    = block_reduce(d_Tw_out,    w=W)
d_Tmpg_in   = block_reduce(d_Tmpg_in,   w=W)
d_Tmpg_out  = block_reduce(d_Tmpg_out,  w=W)

d_mw_dot    = block_reduce(d_mw_dot,    w=W)
d_mmpg_dot  = block_reduce(d_mmpg_dot,  w=W)

d_Hpev      = block_reduce(d_Hpev,      w=W)
d_Lpev      = block_reduce(d_Lpev,      w=W)
d_omegab    = block_reduce(d_omegab,    w=W)

d_SH        = block_reduce(d_SH,        w=W)
d_Theater2  = block_reduce(d_Theater2,  w=W)
d_Theater1  = block_reduce(d_Theater1,  w=W)

d_Pheat_out = block_reduce(d_Pheat_out, w=W)
d_COP       = block_reduce(d_COP,       w=W)
d_Pevap     = block_reduce(d_Pevap,     w=W)
d_error     = block_reduce(d_error,     w=W)

pc_ss = np.array(pc_ss[t_init:total_time_steps])
# pbuff2_ss = np.array(pbuff2_ss[t_init:total_time_steps])
pe_ss = np.array(pe_ss[t_init:total_time_steps])
Tc_out_ss = np.array(Tc_out_ss[t_init:total_time_steps])
Te_out_ss = np.array(Te_out_ss[t_init:total_time_steps])
Tw_out_ss = np.array(Tw_out_ss[t_init:total_time_steps])
Tmpg_out_ss = np.array(Tmpg_out_ss[t_init:total_time_steps])
SH_ss = np.array(SH_ss[t_init:total_time_steps])
COP_ss = np.array(COP_ss[t_init:total_time_steps])
Pgc_ss = np.array(Pgc_ss[t_init:total_time_steps])
Prec_total_ss = np.array(Prec_total_ss[t_init:total_time_steps])
Pheat_out_ss = np.array(Pheat_out_ss[t_init:total_time_steps]) * 1e-3
Pevap_ss = np.array(Pevap_ss[t_init:total_time_steps]) * 1e-3
t_ss = np.array([x - t_init for x in t_ss[t_init:total_time_steps]])

# --- Simulated series ---
pc_ss         = block_reduce(pc_ss,         w=W)
pe_ss         = block_reduce(pe_ss,         w=W)
Tc_out_ss     = block_reduce(Tc_out_ss,     w=W)
hc_out_ss     = block_reduce(hc_out_ss,     w=W)

Te_out_ss     = block_reduce(Te_out_ss,     w=W)
Tw_out_ss     = block_reduce(Tw_out_ss,     w=W)
Tmpg_out_ss   = block_reduce(Tmpg_out_ss,   w=W)
SH_ss         = block_reduce(SH_ss,         w=W)
COP_ss        = block_reduce(COP_ss,        w=W)
Pgc_ss        = block_reduce(Pgc_ss,        w=W)
Prec_total_ss = block_reduce(Prec_total_ss, w=W)
Pheat_out_ss  = block_reduce(Pheat_out_ss,  w=W)
Pevap_ss      = block_reduce(Pevap_ss,      w=W)

# --- Time vector ---
t_ss = block_reduce(t_ss, w=W)


# print(len(Tc_out), len(Tc_out_ss))
r2_pc, mae_pc = calculate_r2_mae(pc_ss, d_pc)
print(f'High Pressure: R² = {r2_pc:.4f}, mae = {mae_pc:.2f}%')

# R² and mae for low pressure
r2_pe, mae_pe = calculate_r2_mae(pe_ss, d_pe)
print(f'Low Pressure: R² = {r2_pe:.4f}, mae = {mae_pe:.2f}%')

# R² and MAPE for GC output
r2_Tc_out, mae_Tc_out = calculate_r2_mae(Tc_out_ss, d_Tc_out)
print(f'GC Output: R² = {r2_Tc_out:.4f}, MAPE = {mae_Tc_out:.2f}K')

# R² and MAPE for EVAP output
r2_Te_out, mae_Te_out = calculate_r2_mae(Te_out_ss, d_Te_out)
print(f'EVAP Output: R² = {r2_Te_out:.4f}, MAE = {mae_Te_out:.2f}K')

r2_SH, mae_SH = calculate_r2_mae(SH_ss, d_SH)
print(f'Superheat: R² = {r2_SH:.4f}, MAPE = {mae_SH:.2f}K')

r2_Tw_out, mae_Tw_out = calculate_r2_mae(Tw_out_ss, d_Tw_out)
print(f'Water Output: R² = {r2_Tw_out:.4f}, MAE = {mae_Tw_out:.2f}K')

r2_Tmpg_out, mae_Tmpg_out = calculate_r2_mae(Tmpg_out_ss, d_Tmpg_out)
print(f'mpg Output: R² = {r2_Tmpg_out:.4f}, MAE = {mae_Tmpg_out:.2f}K')

r2_Pevap, mape_Pevap = calculate_r2_mape(Pevap_ss, d_Pevap)
print(f'Pevap: R² = {r2_Pevap:.4f}, MAPE = {mape_Pevap:.2f}%')

r2_Pheat, mape_Pheat = calculate_r2_mape(Pheat_out_ss, d_Pheat_out)
print(f'Pheat: R² = {r2_Pheat:.4f}, MAPE = {mape_Pheat:.2f}%')

r2_COP, mape_COP = calculate_r2_mape(COP_ss, d_COP)
print(f'COP: R² = {r2_COP:.4f}, MAPE = {mape_COP:.2f}%')
t1_ss = np.linspace(t_ss[0], t_ss[-1], len(t_ss))

# --- Plot 1: Pressures, Temperatures, Water & MPG, Valve ---
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

# --- Pressure Plot ---
ax1.plot(t_ss, pc_ss, '--', lw=1.8, color='#006400')
ax1.plot(t_ss, d_pc, '-', lw=2.3, label='High-Pressure', color='#006400')
ax1.plot(t_ss, pe_ss, '--', lw=1.8, color='#3CB371')
ax1.plot(t_ss, d_pe, '-', lw=2.3, label='Low-Pressure', color='#3CB371')
ax1.set_ylabel("Pressure [bar]", fontsize=16)
# ax1.set_title("High & Low Side Pressures", fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=12)
# ax1.grid(True, alpha=0.3)
ax1.tick_params(labelbottom=False)

# --- GC & EVAP Temps ---
# ax2.plot(t_ss, Tc_out_ss, '--', lw=1.8, label='Sim: GC Out', color='#800080')
# ax2.plot(t_ss, d_Tc_out, '-', lw=2.3, label='Exp: GC Out', color='#800080')
# ax2.plot(t_ss, Te_out_ss, '--', lw=1.8, label='Sim: EVAP Out', color='#4682B4')
# ax2.plot(t_ss, d_Te_out, '-', lw=2.3, label='Exp: EVAP Out', color='#4682B4')
# ax2.set_ylabel("Temperature [°C]", fontsize=16)
# ax2.set_title("GC & EVAP Outlet Temperatures", fontsize=18)
# ax2.legend(fontsize=12)
# ax2.grid(True, alpha=0.3)
# ax2.tick_params(labelsize=14)

# # --- Water Temps ---
# ax2.plot(t_ss, d_Tw_in, '-', lw=2.3, label='Water return', color='#FF6347')
# ax2.plot(t_ss, Tw_out_ss, '--', lw=1.8, label='Sim: Water supply', color='#B22222')
# ax2.plot(t_ss, d_Tw_out, '-', lw=2.3, label='Exp: Water supply', color='#B22222')
# # ax2.plot(t_ss, Tmpg_out_ss, '--', lw=1.8, label='Sim: MPG Out', color='#1E90FF')
# # ax2.plot(t_ss, d_Tmpg_out, '-', lw=2.3, label='Exp: MPG Out', color='#1E90FF')
# ax2.set_ylabel("Temperature [°C]", fontsize=16)
# ax2.tick_params(axis='both', which='major', labelsize=14)
# # ax2.set_title("Water & MPG Outlet Temperatures", fontsize=18)
# ax2.legend(fontsize=12)
# # ax2.grid(True, alpha=0.3)
# ax2.tick_params(labelbottom=False)

# # ax3.plot(t_ss, d_Tmpg_in, '-', lw=2.3, label='MPG in', color='#00008B')
# ax3.plot(t_ss, SH_ss, '--', lw=1.8, label='Sim', color='#1E90FF')
# ax3.plot(t_ss, d_SH, '-', lw=2.3, label='Exp', color='#1E90FF')
# ax3.set_ylabel("Superheat [K]", fontsize=16)
# ax3.tick_params(axis='both', which='major', labelsize=14)
# # ax3.set_title("Water & MPG Outlet Temperatures", fontsize=18)
# ax3.legend(fontsize=12)
# # ax3.grid(True, alpha=0.3)
# ax3.tick_params(labelbottom=False)

# --- MPG Temps ---
# ax3.plot(t_ss, d_Tmpg_in, '-', lw=2.3, label='MPG in', color='#00008B')
# ax3.plot(t_ss, Tmpg_out_ss, '--', lw=1.8, label='Sim: MPG Out', color='#1E90FF')
# ax3.plot(t_ss, d_Tmpg_out, '-', lw=2.3, label='Exp: MPG Out', color='#1E90FF')
# ax3.set_ylabel("Temperature [°C]", fontsize=16)
# ax3.tick_params(axis='both', which='major', labelsize=14)
# # ax3.set_title("Water & MPG Outlet Temperatures", fontsize=18)
# ax3.legend(fontsize=12)
# # ax3.grid(True, alpha=0.3)
# ax3.tick_params(labelbottom=False)


# --- Heating & Evaporation Power ---
ax2.plot(t_ss, Pheat_out_ss, '--', lw=1.8, color='#FF4500')
ax2.plot(t1_ss, d_Pheat_out, '-', lw=2.3, label='Total recovered', color='#FF4500')
ax2.plot(t_ss, Pevap_ss, '--', lw=1.8, color='#00CED1')
ax2.plot(t_ss, d_Pevap, '-', lw=2.3, label='Evaporator', color='#00CED1')
ax2.set_ylabel("Heat transfer rate [kW]", fontsize=14)
# ax2.set_title("Heating & Evaporation Power", fontsize=18)
ax2.legend(fontsize=12)
# ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=14)

# --- COP Plot ---
ax3.plot(t_ss, COP_ss, '--', lw=1.8, color='#008000')
ax3.plot(t1_ss, d_COP, '-', lw=2.3, color='#008000')
ax3.set_ylabel("Thermal COP [-]", fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.legend(fontsize=12)
# ax3.grid(True, alpha=0.3)

ax4.plot(t_ss, d_omegab, label='Burner fan', color='#FFD700')
ax4.set_xlabel("Time [min]", fontsize=16)
ax4.set_ylabel("Speed [rpm]", fontsize=14)
# ax4.set_title("Burner Speed & Heaters Temperatures", fontsize=18)
ax4.tick_params(axis='both', which='major', labelsize=14)
ax4.legend(loc='lower left', fontsize=14)
# ax4.grid(True)

ax5 = ax4.twinx()
ax5.plot(t_ss, d_Theater1, label='Heater 1', color='#8A2BE2')
ax5.plot(t_ss, d_Theater2, label='Heater 2', color='#DA70D6')
ax5.set_ylabel("Temperature [°C]", fontsize=14)
ax5.legend(loc='upper right', fontsize=14)
ax5.tick_params(axis='both', which='major', labelsize=14)
# ax5.grid(True)

# from matplotlib.lines import Line2D
# # -------- Figure-level legend for style (top, black dashed/solid) --------
# fig1.tight_layout(rect=[0, 0, 1, 0.96])   # leaves ~12% at top

# # 2) add the figure-level legend inside the figure
# style_handles = [
#     Line2D([0], [0], color='k', lw=2.3, ls='-',  label='Experimental'),
#     Line2D([0], [0], color='k', lw=1.8, ls='--', label='Simulated'),
# ]
# fig1.legend(handles=style_handles,
#             loc='upper center', bbox_to_anchor=(0.5, 0.997),
#             ncol=2, frameon=True, fancybox=True, framealpha=0.9,
#             edgecolor='0.3', fontsize=12)
fig1.tight_layout()
fig1.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\transient\TCHP_BF.eps", 
            format='eps', bbox_inches='tight')

# # --- Plot 2: Power & COP ---
# fig2, (ax5, ax6, ax7) = plt.subplots(3, 1, figsize=(10, 6))

# # Heating & Evaporation Power
# ax5.plot(t_ss, Pheat_out_ss, '--', lw=1.8, label='Sim: Total recover', color='#FF4500')
# ax5.plot(t1_ss, d_Pheat_out, '-', lw=2.3, label='Exp: Total recover', color='#FF4500')
# ax5.plot(t_ss, Pevap_ss, '--', lw=1.8, label='Sim: Evaporator', color='#00CED1')
# ax5.plot(t_ss, d_Pevap, '-', lw=2.3, label='Exp: Evaporator', color='#00CED1')
# ax5.set_ylabel("Heat Transfer Rate [kW]", fontsize=16)
# ax5.legend(fontsize=12)
# ax5.tick_params(labelsize=14)

# # COP Plot
# ax6.plot(t_ss, COP_ss, '--', lw=1.8, label='Sim: COP', color='#008000')
# ax6.plot(t1_ss, d_COP, '-', lw=2.3, label='Exp: COP', color='#008000')
# ax6.set_xlabel("Time [min]", fontsize=16)
# ax6.set_ylabel("COP [-]", fontsize=16)
# ax6.legend(fontsize=12)
# ax6.tick_params(labelsize=14)

# ax7.plot(t1_ss, d_error, '-', lw=2.3, label='Exp: Error', color='#008000')
# ax7.set_xlabel("Time [min]", fontsize=16)
# ax7.set_ylabel("Error [W]", fontsize=16)
# ax7.legend(fontsize=12)
# ax7.tick_params(labelsize=14)

# fig2.tight_layout()
# fig2.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\transient\TCHP_LPV2.eps", 
#             format='eps', bbox_inches='tight')

# # Plot 3: Heat Recovery
# plt.figure(3)
# plt.plot(t_ss, Prec_total_ss, '--', label='Sim: Recovered Heat', color='k')
# plt.plot(t_ss, Pgc_ss, '--', label='Sim: GC Power', color='b')
# plt.ylabel("Power [W]", fontsize=14)
# plt.xlabel("Time [s]", fontsize=14)
# # plt.title("Recovered vs GC Power", fontsize=18)
# plt.legend(fontsize=12)
# plt.grid(True, alpha=0.3)
# plt.tick_params(labelsize=12)
# plt.tight_layout()
# # plt.show()

# plt.figure(4)
# plt.plot(t_ss, hc_out_ss, label='GC', color='r')
# plt.plot(t_ss, hihx1_out_ss, label='IHX1', color='orange')
# plt.plot(t_ss, hft_ss, label='FT', color='green')
# plt.plot(t_ss, hihx2_out_ss, label='IHX2', color='k')
# plt.plot(t_ss, he_out_ss, label='EVAP', color='b')
# plt.ylabel("Enthalpy [J/kg]", fontsize=14)
# plt.legend(loc='best', fontsize=14)
# plt.grid(True)

# plt.figure(5)
# plt.plot(t_ss, pc_ss, linestyle = '--', label='Sim high pressure', color='r')
# plt.plot(t_ss, d_pc, label='Exp high pressure', color='r')
# plt.plot(t_ss, pbuff2_ss, label='Sim intermediary pressure', color='y')
# plt.plot(t_ss, d_pi, label='Exp intermediary pressure', color='y')
# plt.plot(t_ss, pft_ss, label='pft sim', color='gray')
# plt.plot(t_ss, d_p25, label='Sim MLP', color='green')
# plt.plot(t_ss, pbuff1_ss, linestyle = '--', label='Exp MLP', color='green')
# plt.plot(t_ss, pe_ss, linestyle = '--', label='Sim low pressure', color='b')
# plt.plot(t_ss, d_pe, label='Exp low pressure', color='b')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylabel("Pressure [bar]", fontsize=14)
# plt.legend(loc='best', fontsize=14)
# plt.grid(True)

# plt.figure(6)
# plt.plot(t_ss, d_Hpev, label='HPV', color='r')
# plt.plot(t_ss, d_Lpev, label='LPV', color='b')    
# plt.xlabel("time [s]", fontsize=14)
# plt.ylabel("Valve opening [%]", fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend(loc='best', fontsize=14)
# plt.grid(True)

# # plt.figure(3)
# # plt.plot(t_ss, Tw_in, label='Water', color='r')
# # plt.plot(t_ss, Tmpg_in, label='MPG', color='b')
# # plt.xlabel("time [s]", fontsize=14)
# # plt.ylabel("Temperature [K]", fontsize=14)
# # plt.legend(loc='best', fontsize=14)
# # plt.grid(True)

# # plt.figure(3)
# # plt.plot(t_ss, omegab, label='Burner fan', color='gray')
# # plt.xlabel("time [s]", fontsize=14)
# # plt.ylabel("Speed [rpm]", fontsize=14)
# # plt.xticks(fontsize=12)
# # plt.yticks(fontsize=12)
# # plt.legend(loc='best', fontsize=14)
# # plt.grid(True)

# # plt.figure(3)
# # plt.plot(t_ss, mw_dot, label='mw_dot', color='r')
# # plt.plot(t_ss, mmpg_dot, label='mmpg_dot', color='b')
# # plt.xlabel("time [s]", fontsize=14)
# # plt.ylabel("mass flow rate [kg/s]", fontsize=14)
# # # plt.set_ylabel("$u_{HPV} [\%]$", fontsize=14)
# # # plt.legend(loc='best', fontsize=14)
# # plt.grid(True)

# plt.figure(7)
# plt.plot(t_ss, Tc_out_ss, linestyle = '--', label='Sim GC out', color='r')
# # plt.plot(t_ss, Tc_out, label='Exp GC out', color='r')
# plt.plot(t_ss, Te_out_ss, linestyle = '--', label='Sim EVAP out', color='b')
# # plt.plot(t_ss, Te_out, label='Exp EVAP out', color='b')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("t [s]", fontsize=14)
# plt.ylabel("Temperature [°C]", fontsize=14)
# plt.legend(loc='best', fontsize=12)

# plt.figure(8)
# plt.plot(t_ss, mc_in_dot_ss, label='GC in', linestyle='-', color='blue', linewidth=2, markersize=5)
# plt.plot(t_ss, mc_out_dot_ss, label='GC out', linestyle='--', color='orange', linewidth=2, marker='x', markersize=5)
# # plt.plot(t_ss, mft_in_dot_ss, label='FT in', linestyle='-.', color='green', linewidth=2, marker='s', markersize=5)
# # plt.plot(t_ss, mft_out_dot_ss, label='FT out', linestyle=':', color='red', linewidth=2, marker='^', markersize=5)
# # plt.plot(t_ss, mbuff1_in_dot_ss, label='buff1 in', linestyle='-', color='purple', linewidth=2, marker='D', markersize=5)
# # plt.plot(t_ss, mbuff1_out_dot_ss, label='buff1 out', linestyle='--', color='brown', linewidth=2, marker='v', markersize=5)
# plt.plot(t_ss, me_in_dot_ss, label='EVAP in', linestyle='-.', color='pink', linewidth=2, marker='>', markersize=5)
# plt.plot(t_ss, me_out_dot_ss, label='EVAP out', linestyle=':', color='cyan', linewidth=2, marker='<', markersize=5)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("t [s]", fontsize=14)
# plt.ylabel("mass flow rate [kg/s]", fontsize=14)
# plt.legend(loc='best', fontsize=12)

# plt.figure(9)
# plt.plot(t_ss, COP_ss, linestyle = '--', label = 'Sim', color = 'green')
# plt.plot(t_ss, d_COP, label = 'Exp', color = 'green')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel("t [s]", fontsize=14)
# plt.ylabel("COP [-]", fontsize=14)
# plt.legend(loc='best', fontsize=14)

plt.show()


