import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import matplotlib.patches as patches
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
# ---------------------------------------
# 1. Create and configure CO₂ state
# ---------------------------------------
fluid = "CO2"
state = AbstractState("HEOS", fluid)

T_triple = state.keyed_output(CP.iT_triple)
T_crit = state.keyed_output(CP.iT_critical)
p_triple = state.keyed_output(CP.iP_triple)
p_crit = state.keyed_output(CP.iP_critical)

# ---------------------------------------
# 2. Generate saturation curves (liq & vap)
# ---------------------------------------
T_range = np.linspace(T_triple, T_crit, 1000)
h_liq, h_vap, P_sat = [], [], []

for T in T_range:
    # Liquid
    state.update(CP.QT_INPUTS, 0, T)
    h_liq.append(state.hmass() / 1e3)  # kJ/kg
    P_sat.append(state.p() / 1e5)      # bar

    # Vapor
    state.update(CP.QT_INPUTS, 1, T)
    h_vap.append(state.hmass() / 1e3)  # kJ/kg

# Append critical point to both curves to close dome
state.update(CP.PT_INPUTS, p_crit, T_crit)
h_crit = state.hmass() / 1e3  # kJ/kg
p_crit_bar = p_crit / 1e5     # bar

h_liq.append(h_crit)
h_vap.append(h_crit)
P_sat.append(p_crit_bar)

# ---------------------------------------
# 3. Plot
# ---------------------------------------
h_min, h_max = 150, 600
transition_lower = p_crit_bar - 5
transition_upper = p_crit_bar + 5


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
# Colored regions
ax.add_patch(patches.Rectangle((h_min, 10), h_max - h_min, transition_lower, color='skyblue', alpha=0.4))
ax.add_patch(patches.Rectangle((h_min, transition_lower), h_max - h_min, 10, color='yellowgreen', alpha=0.4))
ax.add_patch(patches.Rectangle((h_min, transition_upper), h_max - h_min, 100 - transition_upper, color='sandybrown', alpha=0.4))

# Saturation dome
ax.plot(h_liq, P_sat, 'k', lw=2)
ax.plot(h_vap, P_sat, 'k', lw=2, label='Saturation curve')

# Critical point
ax.plot(h_crit, p_crit_bar, 'ko')
ax.text(h_crit + 5, p_crit_bar + 1, "Critical point", fontsize=12)

# Region labels
ax.text(490, 40, "Subcritical", fontsize=14, color='black')
ax.text(490, 73, "Transition", fontsize=14, color='black')
ax.text(490, 90, "Transcritical", fontsize=14, color='black')

# Axis labels
ax.set_xlabel(r"Specific enthalpy [kJ.kg$^{-1}$]", fontsize=16)
ax.set_ylabel("Pressure [bar]", fontsize=16)
ax.set_xlim(h_min, h_max)
ax.set_ylim(20, 100)
ax.set_yticks([10, 30, 50, 70, 90])
ax.set_xticks(np.arange(150, 625, 75))
ax.tick_params(axis='both', labelsize=14)
# ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\State_of_art\CO2_ph.eps", format='eps')

plt.show()
