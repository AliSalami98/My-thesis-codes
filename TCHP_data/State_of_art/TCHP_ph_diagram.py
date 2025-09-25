import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState

# ---------------------
# 1. Setup fluid and state
# ---------------------
fluid = "CO2"
state = AbstractState("HEOS", fluid)

T_triple = state.keyed_output(CP.iT_triple)
T_critical = state.keyed_output(CP.iT_critical)
P_triple = state.keyed_output(CP.iP_triple)
P_critical = state.keyed_output(CP.iP_critical)

# ---------------------
# 2. Saturation curve
# ---------------------
T_range = np.linspace(T_triple, T_critical, 1000)
h_liq, h_vap, P_sat = [], [], []

for T in T_range:
    state.update(CP.QT_INPUTS, 0, T)
    h_liq.append(state.hmass() / 1e3)
    P_sat.append(state.p() / 1e5)
    
    state.update(CP.QT_INPUTS, 1, T)
    h_vap.append(state.hmass() / 1e3)

# ---------------------
# 3. Define cycle points using HmassP_INPUTS
# ---------------------
cycle_points = []

# Point 1: 
p1 = 30e5  # Pa
h1 = 470e3  # J/kg
state.update(CP.HmassP_INPUTS, h1, p1)
cycle_points.append((h1 / 1e3, p1 / 1e5))  # (kJ/kg, bar)

# Point 2:
p2 = 45e5  # Pa
h2 = h1 + 20e3
state.update(CP.HmassP_INPUTS, h2, p2)
cycle_points.append((h2 / 1e3, p2 / 1e5))

# Point 3:
p3 = p2  # Pa
h3 = h2 - 30e3 # J/kg
state.update(CP.HmassP_INPUTS, h3, p3)
cycle_points.append((h3 / 1e3, p3 / 1e5))

# Point 4: 
p4 = 68e5  # Pa
h4 = h3 + 30e3  # J/kg
state.update(CP.HmassP_INPUTS, h4, p4)
cycle_points.append((h4 / 1e3, p4 / 1e5))

# Point 5: 
p5 = p4  # Pa
h5 = h4 - 20e3  # J/kg
state.update(CP.HmassP_INPUTS, h5, p5)
cycle_points.append((h5 / 1e3, p5 / 1e5))

# Point 6: 
p6 = 90e5  # Pa
h6 = h1 + 30e3  # J/kg
state.update(CP.HmassP_INPUTS, h6, p6)
cycle_points.append((h6 / 1e3, p6 / 1e5))

# Point 7: 
p76 = p6
h76 = h6 - 10e3
state.update(CP.HmassP_INPUTS, h76, p76)
cycle_points.append((h76 / 1e3, p76 / 1e5))

# Point 7: 
p7 = p6
h7 = h6 - 150e3
state.update(CP.HmassP_INPUTS, h7, p7)
cycle_points.append((h7 / 1e3, p7 / 1e5))

# Point 8:
p8 = p6
h8 = h7 - 20e3
state.update(CP.HmassP_INPUTS, h8, p8)
cycle_points.append((h8 / 1e3, p8 / 1e5))

# Point 9:
p9 = p3
h9 = h8
state.update(CP.HmassP_INPUTS, h9, p9)
cycle_points.append((h9 / 1e3, p9 / 1e5))

# Point 10:
p10 = p3
state.update(CP.PQ_INPUTS, p10, 0)
h10 = state.hmass()
state.update(CP.HmassP_INPUTS, h10, p10)
cycle_points.append((h10 / 1e3, p10 / 1e5))


# Point 11:
p11 = p3
state.update(CP.PQ_INPUTS, p11, 1)
h11 = state.hmass()
state.update(CP.HmassP_INPUTS, h11, p11)
cycle_points.append((h11 / 1e3, p11 / 1e5))

# Point 12:
p12 = p3
h12 = h3
state.update(CP.HmassP_INPUTS, h12, p12)
cycle_points.append((h12 / 1e3, p12 / 1e5))

cycle_points.append((h10 / 1e3, p10 / 1e5))

# Point 13:
p13 = p1
h13 = h10
state.update(CP.HmassP_INPUTS, h13, p13)
cycle_points.append((h13 / 1e3, p13 / 1e5))

# Point 14:
p14 = p1
h14 = h1 - 30e3
state.update(CP.HmassP_INPUTS, h14, p14)
cycle_points.append((h14 / 1e3, p14 / 1e5))

# Point 15:
p15 = p1
h15 = h1
state.update(CP.HmassP_INPUTS, h15, p15)
cycle_points.append((h15 / 1e3, p15 / 1e5))
# Split into H and P arrays for plotting
enthalpy_cycle, pressure_cycle = zip(*cycle_points)
# ---------------------
# 4. Plot
# ---------------------
plt.figure(figsize=(9, 6))
plt.plot(h_liq, P_sat, 'k-', label="Saturation Curve")
plt.plot(h_vap, P_sat, 'k-')

# Cycle plot
plt.plot(enthalpy_cycle, pressure_cycle, 'g-o', linewidth=2.5, label='CO$_2$ Cycle')

# Manually label each point with chosen offsets (edit these as needed)
plt.text(enthalpy_cycle[0] + 10,  pressure_cycle[0] + 1,  "1",  fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[1] + 10,  pressure_cycle[1] + 2,  "2",  fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[2] - 10,  pressure_cycle[2] - 3,  "3",  fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[3] + 5,   pressure_cycle[3] + 2,  "4",  fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[4] - 10,  pressure_cycle[4] - 2,  "5",  fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[5] + 8,   pressure_cycle[5] + 1,  "6",  fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[6] - 5,   pressure_cycle[6] - 2,  "7",  fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[7] + 5,  pressure_cycle[7] - 2,  "8",  fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[8] - 12,  pressure_cycle[8] + 2,  "9",  fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[9] - 10,  pressure_cycle[9] + 2,  "10", fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[10] - 10, pressure_cycle[10], "12", fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[11] - 10, pressure_cycle[11] - 2, "11", fontsize=14, ha='center', va='center')
# plt.text(enthalpy_cycle[12] + 10, pressure_cycle[12] + 1, "12", fontsize=14, ha='center', va='center')
# plt.text(enthalpy_cycle[13] + 10, pressure_cycle[13] - 2, "12", fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[14] - 10, pressure_cycle[14] - 2, "13", fontsize=14, ha='center', va='center')
plt.text(enthalpy_cycle[15] - 15, pressure_cycle[15] - 2, "14", fontsize=14, ha='center', va='center')


# # Gas cooler water temperature profile (30°C to 60°C)
# T_water_gc = np.linspace(30, 60, 50)  # in °C
# P_gc_line = np.full_like(T_water_gc, 90)  # Gas cooler pressure in bar
# h_gc_water = np.linspace(h7 / 1e3, h6 / 1e3, len(T_water_gc))  # From Point 7 to 6 (kJ/kg)

# plt.plot(h_gc_water, P_gc_line, 'b--', label='Water T (GC)')
# for i in range(0, len(T_water_gc), 10):
#     plt.text(h_gc_water[i], P_gc_line[i] + 2, f"{int(T_water_gc[i])}°C",
#              fontsize=9, color='blue', ha='center')

# ---------------------
# 5. Aesthetics
# ---------------------
plt.xlabel("Specific Enthalpy [kJ/kg]", fontsize=18)
plt.ylabel("Pressure [bar]", fontsize=18)
plt.yscale("linear")
plt.ylim(20, 100)
plt.xlim(140, 540)
# plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper left', fontsize = 18)
plt.tight_layout()

plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\State_of_art\TCHP_ph.eps", format='eps')
plt.show()
