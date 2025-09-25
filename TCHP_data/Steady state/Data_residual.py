import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


from cycle_model import(
    a_T22,
    a_T9,
    a_Tevap_out,
    a_T20,
    a_T1,
    a_T2,
    a_T3,
    a_T4,
    a_T5,
    a_T6,
    a_pevap,
    a_p25,
    a_pi,
    a_pgc,
    a_h11,
    a_h22,
    a_h8,
    a_h23,
    a_h24,
    a_mf1_dot,
    a_mf_dot,
    a_hevap_out,
    a_hgc_out,
    a_Theater,
    a_Pcomb,
    a_Pcooler1,
    a_Pcooler2,
    a_Pcooler3,
    a_Pgc,
    a_Pevap,
    a_PIHX,
    a_Pbuffer1,
    a_Pbuffer2,
    a_Pbuffer3,
    a_PHXs_t,
    a_Text,
    Pheat_out,
    a_Pout_comp,
    SLE,
    GUE,
    a_Pcomp3,
    Res_HP,
    Res_water,
    Res_HP_min,
    Res_HP_max,
    Res_TCs,
    Res_ratio
)

Res_HP = np.array(Res_HP)
Res_TCs = np.array(Res_TCs)
Res_water = np.array(Res_water)

points = np.linspace(0, len(Res_HP) - 1, len(Res_HP))

# Calculate zero error line
zero_error = np.zeros_like(Res_HP)

# Calculate the error as ±5.65% of Res_HP
error = 0.0565 * np.abs(Res_HP)
error = 0.0565 * np.abs(Res_TCs)


# # Create the plot
# plt.figure()

# # First plot: Scatter plot of residuals
# plt.scatter(points, Res_ratio, label='Thermal Compressor Residuals', color='yellow')
# plt.xlabel('Data Points', fontsize = 12)
# plt.ylabel('Residuel [-]', fontsize = 12)
# # plt.title('Global Balance Residue')
# plt.grid(True)


# Calculate averages
avg_tc = np.mean(Res_TCs)
avg_hp = np.mean(Res_HP)
avg_water = np.mean(Res_water)

# Plot setup
plt.figure(figsize=(8, 6))

# Scatter plots
plt.scatter(points, Res_TCs, label='Thermal Compressors', color='darkorange', s=35, edgecolors='black', alpha=0.8)
plt.scatter(points, Res_HP, label='Heat Pump Cycle', color='royalblue', s=35, edgecolors='black', alpha=0.8)
plt.scatter(points, Res_water, label='Water Circuit', color='crimson', s=35, edgecolors='black', alpha=0.8)

# Add average lines
plt.axhline(avg_tc, color='darkorange', linestyle='--', linewidth=1.5, label=f'Avg TC: {avg_tc:.2f} kW')
plt.axhline(avg_hp, color='royalblue', linestyle='--', linewidth=1.5, label=f'Avg HP: {avg_hp:.2f} kW')
plt.axhline(avg_water, color='crimson', linestyle='--', linewidth=1.5, label=f'Avg Water: {avg_water:.2f} kW')

# Axis labels
plt.xlabel('Samples', fontsize=12)
plt.ylabel('Residual [kW]', fontsize=12)

# Grid and ticks
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))

# Horizontal zero line
plt.axhline(0, color='gray', linewidth=1, linestyle='--')

# Legend
plt.legend(loc='best', fontsize=10, frameon=False)

# Layout and show
plt.tight_layout()
plt.show()