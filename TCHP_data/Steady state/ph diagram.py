import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import csv

from cycle_model import (
    a_pevap,
    a_p25,
    a_pi,
    a_pgc,
    a_h1,
    a_h2,
    a_h3,
    a_h4,
    a_h5,
    a_h6,
    a_h7,
    a_h11,
    a_h22,
    a_h23,
    a_h8,
    a_h24,
    a_hevap_out,
    a_hbuffer1_out,
    a_Text,
)

pressure = np.zeros((18, 15))
enthalpy = np.zeros((18, 15))
pressure_min = np.zeros((18, 15))
enthalpy_min = np.zeros((18, 15))
pressure_max = np.zeros((18, 15))
enthalpy_max = np.zeros((18, 15))

pressure_est1 = np.zeros((18, 1))
enthalpy_est1 = np.zeros((18, 1))

pressure_est2 = np.zeros((18, 1))
enthalpy_est2 = np.zeros((18, 1))
for i in range(len(a_Text)):
        pressure[i][:] = [a_p25[i], a_p25[i], a_p25[i], a_pi[i], a_pi[i], a_pgc[i], a_pgc[i], a_pgc[i], a_pgc[i], a_p25[i], a_p25[i], a_pevap[i], a_pevap[i], a_p25[i], a_p25[i]]        
        pressure[i][:] = [x*10**(-5) for x in pressure[i][:]]
        pressure_est1[i][:] = [a_p25[i]*10**(-5)]
        pressure_est2[i][:] = [a_pevap[i]*10**(-5)]
        pressure_min[i][:] = [0.97*x for x in pressure[i][:]]
        pressure_max[i][:] = [1.03*x for x in pressure[i][:]]

        enthalpy[i][:] = [a_h8[i], a_h7[i], a_h3[i], a_h4[i], a_h5[i], a_h6[i], a_h11[i], a_h22[i], a_h23[i], a_h8[i], a_h24[i], a_h24[i], a_h1[i], a_h2[i], a_h3[i]]
        enthalpy[i][:] = [x*10**(-3) for x in enthalpy[i][:]]
        enthalpy_est1[i][:] = [a_hbuffer1_out[i]*10**(-3)]
        enthalpy_est2[i][:] = [a_hevap_out[i]*10**(-3)]
        enthalpy_min[i][:] = [0.95*x for x in enthalpy[i][:]]
        enthalpy_max[i][:] = [1.05*x for x in enthalpy[i][:]]

import matplotlib.pyplot as plt

temperature_range = np.linspace(CP.PropsSI('Ttriple', 'CO2'), CP.PropsSI('Tcrit', 'CO2'), 1000)

# Initialize lists to hold saturation properties
pressure_liquid = []
pressure_vapor = []
enthalpy_liquid = []
enthalpy_vapor = []

# Loop through the temperature range and calculate saturation properties
for T in temperature_range:
    P = CP.PropsSI('P', 'T', T, 'Q', 0, 'CO2')  # Saturation pressure for liquid
    h_l = CP.PropsSI('H', 'T', T, 'Q', 0, 'CO2')  # Enthalpy for liquid
    h_v = CP.PropsSI('H', 'T', T, 'Q', 1, 'CO2')  # Enthalpy for vapor

    pressure_liquid.append(P)
    pressure_vapor.append(P)
    enthalpy_liquid.append(h_l*10**(-3))
    enthalpy_vapor.append(h_v*10**(-3))

pressure_liquid = np.array(pressure_liquid) / 1e5
pressure_vapor = np.array(pressure_vapor) / 1e5

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
fig.tight_layout(pad=4.0)

labels = ['D', 'G', 'C', 'H', 'B', 'F', 'I', 'A', 'Z']
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

# -------- FIRST 9 POINTS (0–8) --------
fig1, axes1 = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), sharex=True, sharey=True)
fig1.tight_layout(pad=4.0)

labels = ['D', 'G', 'C', 'H', 'B', 'F', 'I', 'A', 'Z']
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']

for index in range(9):
    ax = axes1[index // 3, index % 3]
    ax.plot(enthalpy_liquid, pressure_liquid, enthalpy_vapor, pressure_vapor, color='k')
    ax.plot(enthalpy[index], pressure[index], marker='o', color=colors[index])
    ax.plot(enthalpy_est1[index], pressure_est1[index], marker='x', color=colors[index])
    ax.plot(enthalpy_est2[index], pressure_est2[index], marker='x', color=colors[index])
    ax.fill_betweenx(pressure[index], enthalpy_min[index], enthalpy_max[index], color=colors[index], alpha=0.3)
    ax.fill_between(enthalpy[index], pressure_min[index], pressure_max[index], color=colors[index], alpha=0.3)
    ax.set_title(f'Point: {labels[index]}, Text: {int(a_Text[index])}') 
    ax.set_xlabel('Enthalpy [kJ/kg]')
    ax.set_ylabel('Pressure [bar]')
    ax.grid(True)

# -------- REMAINING POINTS (index 9 onwards) --------
remaining = len(a_Text) - 9
ncols = 3
nrows = (remaining + ncols - 1) // ncols

fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows), sharex=True, sharey=True)
axes2 = axes2.flatten()
fig2.tight_layout(pad=4.0)

new_labels = labels[9:] if len(labels) > 9 else [chr(65 + i + 9) for i in range(remaining)]
new_colors = colors[9:] if len(colors) > 9 else ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:purple'] * 3

for idx, index in enumerate(range(9, len(a_Text))):
    ax = axes2[idx]
    ax.plot(enthalpy_liquid, pressure_liquid, enthalpy_vapor, pressure_vapor, color='k')
    ax.plot(enthalpy[index], pressure[index], marker='o', color=new_colors[idx])
    ax.plot(enthalpy_est1[index], pressure_est1[index], marker='x', color=new_colors[idx])
    ax.plot(enthalpy_est2[index], pressure_est2[index], marker='x', color=new_colors[idx])
    ax.fill_betweenx(pressure[index], enthalpy_min[index], enthalpy_max[index], color=new_colors[idx], alpha=0.3)
    ax.fill_between(enthalpy[index], pressure_min[index], pressure_max[index], color=new_colors[idx], alpha=0.3)
    ax.set_title(f'Point: {new_labels[idx]}, Text: {int(a_Text[index])}')
    ax.set_xlabel('Enthalpy [kJ/kg]')
    ax.set_ylabel('Pressure [bar]')
    ax.grid(True)

# Hide unused axes if any
for ax in axes2[remaining:]:
    ax.axis('off')

plt.show()
