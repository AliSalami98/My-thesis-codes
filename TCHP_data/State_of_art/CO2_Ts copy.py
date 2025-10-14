import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState

# --- Font setup (unchanged) ---
path = r"C:\Users\ali.salame\AppData\Local\Microsoft\Windows\Fonts\CHARTERBT-ROMAN.OTF"
fm.fontManager.addfont(path)
prop = fm.FontProperties(fname=path)
mpl.rcParams["font.family"] = prop.get_name()
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10

# --- Fluid / state ---
fluid = "CO2"
st = AbstractState("HEOS", fluid)

# Critical & triple points
T_triple = st.keyed_output(CP.iT_triple)
T_crit   = st.keyed_output(CP.iT_critical)
p_crit   = st.keyed_output(CP.iP_critical)
p_crit_bar = p_crit / 1e5  # bar

# --- Saturation dome in p–h space ---
T_vals = np.linspace(T_triple, T_crit, 1000)
hL, hV, Psat_bar = [], [], []
for T in T_vals:
    st.update(CP.QT_INPUTS, 0.0, T)        # saturated liquid
    hL.append(st.hmass()/1e3)              # kJ/kg
    Psat_bar.append(st.p()/1e5)            # bar
    st.update(CP.QT_INPUTS, 1.0, T)        # saturated vapor
    hV.append(st.hmass()/1e3)

# Add critical point to close the dome visually
st.update(CP.PT_INPUTS, p_crit, T_crit)
h_crit = st.hmass()/1e3                   # kJ/kg
hL.append(h_crit); hV.append(h_crit); Psat_bar.append(p_crit_bar)

# --- Plot ---
h_min, h_max = 200, 500
p_min, p_max = 20, 100

fig, ax = plt.subplots(figsize=(8,6))

# Saturation dome curves
ax.plot(hL, Psat_bar, 'k', lw=2)
ax.plot(hV, Psat_bar, 'k', lw=2, label='Saturation curve')

# Critical point
ax.plot(h_crit, p_crit_bar, 'g^', ms=9, label='Critical point')

# Dashed line at critical pressure
ax.axhline(p_crit_bar, color='k', linestyle='--', lw=1, label=r'$p_\mathrm{crit}$')

# Region labels
ax.text(220, 55, "Liquid", fontsize=12, color='black')
ax.text(330, 55, "Two-phase",
        ha='center', va='center', fontsize=12, color='black')
ax.text(330, 90, "Supercritical", fontsize=12, color='black')
ax.text(450, 55, "Vapor", fontsize=12, color='black')

# Axes & styling
ax.set_xlabel(r"Specific enthalpy [kJ·kg$^{-1}$]", fontsize=16)
ax.set_ylabel("Pressure [bar]", fontsize=16)
ax.set_xlim(h_min, h_max)
ax.set_ylim(p_min, p_max)
# ax.set_yticks([20, 30, 50, 70, 90, 100])
# ax.set_xticks(np.arange(200, 500, 75))
ax.tick_params(axis='both', labelsize=14)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(loc='upper right')
plt.tight_layout()

plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\State_of_art\CO2_ph.eps", format='eps')
plt.show()
