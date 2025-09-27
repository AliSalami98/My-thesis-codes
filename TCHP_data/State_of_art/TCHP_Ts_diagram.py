import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
def interpolate_curve(h_start, p_start, h_end, p_end, n):
    """Return lists of entropy and temperature interpolated between two h-p points"""
    s_list, T_list = [], []
    for i in np.linspace(0, 1, n):
        h = h_start + i * (h_end - h_start)
        p = p_start + i * (p_end - p_start)
        if 73.5e5 < p < 73.8e5:
            p = 73.85e5
        state.update(CP.HmassP_INPUTS, h, p)
        s_list.append(state.smass() / 1e3)
        T_list.append(state.T() - 273.15)
    return s_list, T_list
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
# ---------------------
# 1. Setup fluid and state
# ---------------------
fluid = "CO2"
state = AbstractState("HEOS", fluid)

T_triple = state.keyed_output(CP.iT_triple)
T_critical = state.keyed_output(CP.iT_critical)

# ---------------------
# 2. Saturation curve (T vs s)
# ---------------------
T_range = np.linspace(T_triple, T_critical, 1000)
s_liq, s_vap = [], []

for T in T_range:
    state.update(CP.QT_INPUTS, 0, T)
    s_liq.append(state.smass() / 1e3)

    state.update(CP.QT_INPUTS, 1, T)
    s_vap.append(state.smass() / 1e3)

T_C = T_range - 273.15  # Convert to Celsius

# ---------------------
# 3. Define cycle points using HmassP_INPUTS and extract T-s points
# ---------------------
cycle_points_hp = []

# Define points as in original code
p1 = 30e5
state.update(CP.PQ_INPUTS, p1, 1)
h1 = state.hmass() + 30e3
# h1 = 470e3
p2 = 45e5
h2 = h1 + 20e3
p3 = p2
h3 = h2 - 30e3
p4 = 68e5
h4 = h3 + 30e3
p5 = p4
h5 = h4 - 20e3
p6 = 90e5
h6 = h1 + 30e3
p76 = p6
h76 = h6 - 10e3
p7 = p6
h7 = h6 - 150e3
p8 = p6
h8 = h7 - 20e3
p9 = p3
h9 = h8
p10 = p3
state.update(CP.PQ_INPUTS, p10, 0)
h10 = state.hmass()
p11 = p3
state.update(CP.PQ_INPUTS, p11, 1)
h11 = state.hmass()
p12 = p1
h12 = h10
p13 = p1
h13 = h1 - 20e3

# List of (h, p) points
cycle_points_hp = [
    (h9, p9), (h11, p11), (h3, p3), (h4, p4), (h5, p5),
    (h6, p6), (h76, p76), (h7, p7), (h8, p8), (h9, p9),
    (h10, p10), (h12, p12), (h13, p13), (h1, p1),
      (h2, p2), (h3, p3), (h4, p4), 
]


Water_points = []
# Convert to T-s pairs
Ts_cycle = []
for h, p in cycle_points_hp:
    state.update(CP.HmassP_INPUTS, h, p)
    T = state.T() - 273.15  # °C
    s = state.smass() / 1e3  # kJ/kg·K
    Ts_cycle.append((s, T))

# entropy_cycle, temperature_cycle = [], []
# for i in range(len(cycle_points_hp) - 1):
#     h1, p1 = cycle_points_hp[i]
#     h2, p2 = cycle_points_hp[i + 1]
#     s_interp, T_interp = interpolate_curve(h1, p1, h2, p2, n=10)
#     entropy_cycle.extend(s_interp)
#     temperature_cycle.extend(T_interp)

entropy_cycle, temperature_cycle = zip(*Ts_cycle)


# ---------------------
# 4. Add Water Line (25 °C to 34 °C)
# ---------------------
# T_water = np.linspace(40, 50, 10)  # in Celsius
# T_water_K = T_water + 273.15
# s_water = [CP.PropsSI("S", "T", T, "P", 2e5, "Water") / 1e3 for T in T_water_K]  # kJ/kg.K


# 5. Plot T-s diagram
plt.figure(figsize=(8, 5))

# Plot saturation curve
plt.plot(s_liq, T_C, 'k-', linewidth=1, label='Saturation curve', zorder=3)
plt.plot(s_vap, T_C, 'k-', linewidth=1, zorder=3)

# Plot interpolated CO2 cycle lines
s_interp, T_interp = [], []
for i in range(len(cycle_points_hp) - 1):
    h1, p1 = cycle_points_hp[i]
    h2, p2 = cycle_points_hp[i + 1]
    s_seg, T_seg = interpolate_curve(h1, p1, h2, p2, n=40)
    s_interp.extend(s_seg)
    T_interp.extend(T_seg)

plt.plot(s_interp, T_interp, 'g-', linewidth=2, label=r'TCHP CO$_2$ cycle', zorder=3)

# Plot white-filled, green-edged markers at cycle points
for s, T in Ts_cycle:
    plt.plot(s, T, marker='o', markerfacecolor='white', markeredgecolor='green', markeredgewidth=2, markersize=6, zorder=3)

# Optional: Water heating line
# plt.plot(s_water, T_water, 'b--', linewidth=2, label='Water Heating')

# 6. Aesthetics
plt.xlabel(r"Specific entropy [kJ·kg$^{-1}$·K$^{-1}$]", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)
plt.xlim(1, 2.1)
plt.ylim(-10, 90)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which='both', alpha=0.3, zorder=0)
plt.legend(loc='upper left', fontsize=14)
plt.tight_layout()

# Save figure
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\State_of_art\TCHP_Ts.eps", format='eps')
plt.show()
