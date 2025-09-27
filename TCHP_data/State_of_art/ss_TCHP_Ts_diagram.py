import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
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
# Helper: interpolate between two h-p points for smooth plotting
# ---------------------
def interp_hp_to_Ts(state, h_start, p_start, h_end, p_end, n=40):
    s_list, T_list = [], []
    for lam in np.linspace(0, 1, n):
        h = h_start + lam * (h_end - h_start)
        p = p_start + lam * (p_end - p_start)
        state.update(CP.HmassP_INPUTS, h, p)
        s_list.append(state.smass() / 1e3)         # kJ/kg-K
        T_list.append(state.T() - 273.15)          # °C
    return s_list, T_list

# ---------------------
# 1) Fluid and saturation curve
# ---------------------
fluid = "CO2"
state = AbstractState("HEOS", fluid)

T_triple = state.keyed_output(CP.iT_triple)
T_critical = state.keyed_output(CP.iT_critical)

T_range = np.linspace(T_triple, T_critical, 800)
s_liq, s_vap = [], []
for T in T_range:
    state.update(CP.QT_INPUTS, 0, T); s_liq.append(state.smass()/1e3)
    state.update(CP.QT_INPUTS, 1, T); s_vap.append(state.smass()/1e3)
T_C = T_range - 273.15

# ---------------------
# 2) Single-stage cycle: p_low=45 bar, p_high=60 bar
#    Points: 1(suction, sat.vapor+ΔT) -> 2(comp) -> 3(condensed) -> 4(throttle) -> back to 1 (evap)
# ---------------------
p_low  = 45e5   # Pa
p_high = 60e5   # Pa
dT_sh  = 5.0    # K superheat at suction
eta_is = 0.4   # isentropic efficiency (set to 1.0 if you want ideal)

# 1) Suction: saturated vapor at p_low + small superheat
state.update(CP.PQ_INPUTS, p_low, 1)     # sat. vapor
T1 = state.T() + dT_sh                   # add superheat
state.update(CP.PT_INPUTS, p_low, T1)
h1 = state.hmass(); s1 = state.smass()

# 2s) Isentropic to p_high
state.update(CP.PSmass_INPUTS, p_high, s1)
h2s = state.hmass()

# 2) Actual compression with efficiency
h2 = h1 + (h2s - h1) / eta_is
state.update(CP.HmassP_INPUTS, h2, p_high)
s2 = state.smass()

# 3) Condense (or gas-cool if you choose) at p_high to saturated liquid
state.update(CP.PQ_INPUTS, p_high, 0)
h3 = state.hmass(); s3 = state.smass()

# 4) Throttle to p_low (isenthalpic)
h4 = h3
state.update(CP.HmassP_INPUTS, h4, p_low)
s4 = state.smass()

# Back to 1 through evaporation to sat. vapor + superheat (shown as two segments)
# Evaporation to sat. vapor (Q=1) at p_low:
state.update(CP.PQ_INPUTS, p_low, 1)
h1_sat = state.hmass(); s1_sat = state.smass()
# Then small superheat to point 1:
# (already defined by h1,s1 at p_low,T1)

# ---------------------
# Build smooth segments for plotting
# ---------------------
segments = []
# 1 -> 2 (compression)
segments += [interp_hp_to_Ts(state, h1, p_low, h2, p_high)]
# 2 -> 3 (condensing at p_high)
segments += [interp_hp_to_Ts(state, h2, p_high, h3, p_high)]
# 3 -> 4 (throttle: isenthalpic)
segments += [interp_hp_to_Ts(state, h3, p_high, h4, p_low)]
# 4 -> 1sat (evap at p_low)
segments += [interp_hp_to_Ts(state, h4, p_low, h1_sat, p_low)]
# 1sat -> 1 (superheat at p_low)
segments += [interp_hp_to_Ts(state, h1_sat, p_low, h1, p_low)]

# Collect cycle points for markers
cycle_pts = []
for (h,p) in [(h1,p_low),(h2,p_high),(h3,p_high),(h4,p_low)]:
    state.update(CP.HmassP_INPUTS, h, p)
    cycle_pts.append((state.smass()/1e3, state.T()-273.15))

# --- inside plotting section ---

plt.figure(figsize=(8,5))
# Saturation dome
plt.plot(s_liq, T_C, 'k-', linewidth=1, label='Saturation curve', zorder=2)
plt.plot(s_vap, T_C, 'k-', linewidth=1, zorder=2)

# Cycle segments in green
for s_seg, T_seg in segments:
    plt.plot(s_seg, T_seg, color='green', linewidth=2,
             label='CO$_2$ cycle' if 'plotted' not in locals() else None,
             zorder=3)
    plotted = True

for s, T in cycle_pts:
    plt.plot(s, T, marker='o', markerfacecolor='white',
             markeredgecolor='green', markeredgewidth=2,
             markersize=6, zorder=4)

plt.text(cycle_pts[0][0] + 0.02, cycle_pts[0][1] - 2, "1", 
         color='black', fontsize=12, weight='bold')
plt.text(cycle_pts[1][0] + 0.02, cycle_pts[1][1] + 2, "2", 
         color='black', fontsize=12, weight='bold')
plt.text(cycle_pts[2][0] - 0.03, cycle_pts[2][1] + 1, "3", 
         color='black', fontsize=12, weight='bold')
plt.text(cycle_pts[3][0] - 0.03, cycle_pts[3][1] + 1, "4", 
         color='black', fontsize=12, weight='bold')

# Labels & aesthetics
plt.xlabel(r"Specific entropy [kJ·kg$^{-1}$·K$^{-1}$]", fontsize=14)
plt.ylabel("Temperature [°C]", fontsize=14)
plt.xlim(1, 2)
plt.ylim(0, 50)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which='both', alpha=0.3, zorder=0)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()

plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\State_of_art\ss_TCHP_Ts_with_water.eps", format='eps')

plt.show()
