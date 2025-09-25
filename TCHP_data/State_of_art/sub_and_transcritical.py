import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState

# ---------------------------------------
# 1. Create and configure CO₂ state
# ---------------------------------------
fluid = "CO2"
state = AbstractState("HEOS", fluid)

T_triple = state.keyed_output(CP.iT_triple)
T_critical = state.keyed_output(CP.iT_critical)
P_triple = state.keyed_output(CP.iP_triple)
P_critical = state.keyed_output(CP.iP_critical)

# ---------------------------------------
# 2. Generate saturation curves (liq & vap)
# ---------------------------------------
T_range = np.linspace(T_triple, T_critical, 1000)
h_liq, h_vap, P_sat = [], [], []

for T in T_range:
    # Liquid
    state.update(CP.QT_INPUTS, 0, T)
    h_liq.append(state.hmass() / 1e3)  # kJ/kg
    P_sat.append(state.p() / 1e5)      # bar

    # Vapor
    state.update(CP.QT_INPUTS, 1, T)
    h_vap.append(state.hmass() / 1e3)  # kJ/kg

# ---------------------------------------
# 3. Define cycle points
# ---------------------------------------
# Start point = saturated vapor enthalpy at 40 bar
p1_bar = 40
state.update(CP.PQ_INPUTS, p1_bar * 1e5, 1.0)
h1_kJkg = state.hmass() / 1e3

# Manually define rest of the cycle
pressure_cycle = [p1_bar, 80, 80, 40, p1_bar]  # bar
enthalpy_cycle = [h1_kJkg, 480, 300, 300, h1_kJkg]  # kJ/kg

# ---------------------------------------
# 4. Plot P-h diagram with cycle
# ---------------------------------------
plt.figure(figsize=(8,6))
plt.plot(h_liq, P_sat, 'k-', label='Saturation curve')
plt.plot(h_vap, P_sat, 'k-')
plt.plot(enthalpy_cycle, pressure_cycle, 'r-o', label='CO₂ Cycle')

# Label the cycle points
labels = ['1', '2', '3', '4']
for i, label in enumerate(labels):
    plt.text(enthalpy_cycle[i], pressure_cycle[i] + 2, label, ha='center', fontsize=12, weight='bold')

# Add process labels between points
process_labels = ['Compression', 'Heat Rejection', 'Expansion', 'Evaporation']
for i in range(4):
    x = (enthalpy_cycle[i] + enthalpy_cycle[i+1]) / 2
    y = (pressure_cycle[i] + pressure_cycle[i+1]) / 2
    plt.text(x, y + 3, process_labels[i], fontsize=11, ha='center', color='darkred')

# ---------------------------------------
# 5. Add phase region annotations
# ---------------------------------------
mid_h_liq = h_liq[100]
mid_h_vap = h_vap[-100]
mid_h_mix = (h_liq[500] + h_vap[500]) / 2
mid_p = P_sat[500]

plt.text(mid_h_liq - 10, mid_p, 'Subcooled Liquid', rotation=90, fontsize=12, color='blue')
plt.text(mid_h_mix - 5, mid_p + 10, 'Two-phase Region', rotation=0, fontsize=12, color='blue')
plt.text(mid_h_vap + 10, mid_p, 'Superheated Vapor', rotation=90, fontsize=12, color='blue')

# Final plot settings
plt.xlabel("Enthalpy [kJ/kg]", fontsize=16)
plt.ylabel("Pressure [bar]", fontsize=16)
plt.ylim([1, 100])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
