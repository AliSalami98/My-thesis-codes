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
pressure_cycle1 = [p1_bar, 80, 80, 40, p1_bar]  # bar
enthalpy_cycle1 = [h1_kJkg, 480, 340, 340, h1_kJkg]  # kJ/kg

pressure_cycle2 = [p1_bar, 90, 90, 40, p1_bar]  # bar
enthalpy_cycle2 = [h1_kJkg, 493, 297, 297, h1_kJkg]  # kJ/kg
# Optionally compute temperatures
temperature_cycle1 = []
for h, p in zip(enthalpy_cycle1, pressure_cycle1):
    state.update(CP.HmassP_INPUTS, h * 1e3, p * 1e5)
    temperature_cycle1.append(state.T() - 273.15)

temperature_cycle2 = []
for h, p in zip(enthalpy_cycle2, pressure_cycle2):
    state.update(CP.HmassP_INPUTS, h * 1e3, p * 1e5)
    temperature_cycle2.append(state.T() - 273.15)
# ---------------------------------------
# 4. Plot P-h diagram with cycle
# ---------------------------------------
plt.figure(figsize=(8,6))
plt.plot(h_liq, P_sat, 'k-', label='Saturation curve')
plt.plot(h_vap, P_sat, 'k-')
plt.plot(enthalpy_cycle1, pressure_cycle1, 'r-')
plt.plot(enthalpy_cycle2, pressure_cycle2, 'b-')

labels = ['1', '2', '3', '4']
for i, label in enumerate(labels):
    if label == '3':
        plt.text(enthalpy_cycle1[i], pressure_cycle1[i]+2, label, ha='center', fontsize=14)
    elif label == '2':
        plt.text(enthalpy_cycle1[i] + 10, pressure_cycle1[i], label, ha='center', fontsize=14)
    elif label == '1':
        plt.text(enthalpy_cycle1[i] + 15, pressure_cycle1[i]-4, label, ha='center', fontsize=14)
    else:
        plt.text(enthalpy_cycle1[i] - 10, pressure_cycle1[i]-4, label, ha='center', fontsize=14)

plt.text(enthalpy_cycle2[1] + 10, pressure_cycle2[1], "2'", ha='center', fontsize=14)
plt.text(enthalpy_cycle2[2] + 2, pressure_cycle2[2]+2, "3'", ha='center', fontsize=14)
plt.text(enthalpy_cycle2[3], pressure_cycle2[3]-4, "4'", ha='center', fontsize=14)

# ---------------------------------------
# 4a. Plot constant temperature curves (for points 3 and 4)
# ---------------------------------------
isotherm_pressures = np.logspace(np.log10(1e5), np.log10(100e5), 300)  # 1 to 100 bar in Pa

for idx in [2, 3]:  # Points 3 and 4
    T_iso_C = temperature_cycle1[idx]             # °C
    T_iso_K = T_iso_C + 273.15                   # Convert to K
    h_iso = []
    p_iso = []

    for p in isotherm_pressures:
        try:
            state.update(CP.PT_INPUTS, p, T_iso_K)
            h = state.hmass() / 1e3  # in kJ/kg
            h_iso.append(h)
            p_iso.append(p / 1e5)    # in bar
        except:
            continue  # Skip invalid states

    # Plot the isotherm line (no label)
    plt.plot(h_iso, p_iso, 'gray', lw=1)

    # Annotate temperature on the middle of the curve
    if idx == 2:
        plt.text(490, 25, f'{int(T_iso_C)}°C',
                 fontsize=14, color='k', ha='left', va='center')
    else:
        plt.text(460, 25, f'{int(T_iso_C)}°C',
                 fontsize=14, color='k', ha='left', va='center')
        


# # ---------------------------------------
# # 4b. Add isotherm for point 3′ (cycle2 index 2)
# # ---------------------------------------
# T_iso_C_3p = temperature_cycle2[2]              # °C
# T_iso_K_3p = T_iso_C_3p + 273.15                # K
# h_iso_3p = []
# p_iso_3p = []

# for p in isotherm_pressures:
#     try:
#         state.update(CP.PT_INPUTS, p, T_iso_K_3p)
#         h = state.hmass() / 1e3  # kJ/kg
#         h_iso_3p.append(h)
#         p_iso_3p.append(p / 1e5)  # bar
#     except:
#         continue

# plt.plot(h_iso_3p, p_iso_3p, 'gray', lw=1)

# if len(h_iso_3p) > 0:
#     mid_idx = len(h_iso_3p) // 2
#     plt.text(h_iso_3p[mid_idx], p_iso_3p[mid_idx], f"{int(T_iso_C_3p)}°C",
#              fontsize=10, color='gray', ha='left', va='center')

# ---------------------------------------
# 4b. Add Δh2 and Δh3 markers
# ---------------------------------------

# # Δh2: between points 2 and 2'
# plt.annotate('', xy=(enthalpy_cycle2[1], pressure_cycle2[1]), 
#              xytext=(enthalpy_cycle1[1], pressure_cycle1[1]),
#              arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
# mid_h2 = (enthalpy_cycle1[1] + enthalpy_cycle2[1]) / 2
# plt.text(mid_h2, pressure_cycle1[1]+2, r'$\Delta h_2$', color='red', fontsize=12, ha='center')

# # Δh3: between points 3 and 3'
# plt.annotate('', xy=(enthalpy_cycle2[2], pressure_cycle2[2]), 
#              xytext=(enthalpy_cycle1[2], pressure_cycle1[2]),
#              arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
# mid_h3 = (enthalpy_cycle1[2] + enthalpy_cycle2[2]) / 2
# plt.text(mid_h3, pressure_cycle1[2]+2, r'$\Delta h_3$', color='red', fontsize=12, ha='center')


plt.xlabel("Specific enthalpy [kJ/kg]", fontsize=16)
plt.ylabel("Pressure [bar]", fontsize=16)
plt.xlim([180, 520])
plt.ylim([20, 100])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\State_of_art\CO2_cycle_ph2.eps", format='eps')

# plt.title("CO₂ Compression Cycle on P-h Diagram")
plt.show()
