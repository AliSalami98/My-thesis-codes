import matplotlib
import numpy as np
import CoolProp as CP
import matplotlib.pyplot as plt

# Initialize CO2 state
CO2 = CP.AbstractState("HEOS", "CO2")
pc = CO2.keyed_output(CP.iP_critical) / 1e5       # Pa to bar
Tc = CO2.keyed_output(CP.iT_critical) - 273.15     # K to °C
Tmin = -73.15  # 200 K in °C
Tmax = 176.85  # 450 K in °C
pmax = CO2.keyed_output(CP.iP_max) / 1e5           # Pa to bar
pt = CO2.keyed_output(CP.iP_triple) / 1e5          # Pa to bar
Tt = CO2.keyed_output(CP.iT_triple) - 273.15       # K to °C

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
lw = 3

# ----------------
# Melting curve
# ----------------
melt_args = dict(lw=lw, solid_capstyle='round')
TT = []
PP = list(np.logspace(np.log10(pt*1e5), np.log10(pmax*1e5), 1000))  # back to Pa for API call
for p in PP:
    try:
        TT.append(CO2.melting_line(CP.iT, CP.iP, p) - 273.15)  # K to °C
    except:
        TT.append(np.nan)

TT = np.array(TT)
PP = np.array(PP) / 1e5  # Pa to bar
valid = ~np.isnan(TT)
plt.plot(TT[valid], PP[valid], 'darkblue', **melt_args)

# ----------------
# Saturation curve
# ----------------
Ts_K = np.linspace(Tt + 273.15, Tc + 273.15, 1000)
ps = CP.CoolProp.PropsSI('P', 'T', Ts_K, 'Q', 0, 'CO2') / 1e5
Ts = Ts_K - 273.15  # to °C

plt.plot(Ts, ps, 'orange', lw=lw, solid_capstyle='round')

# ----------------
# Critical lines
# ----------------
plt.axvline(Tc, dashes=[2, 2], color='gray', lw=1.5)
plt.axhline(pc, dashes=[2, 2], color='gray', lw=1.5)

# ----------------
# Labels
# ----------------
plt.text(126.85, 500, 'Supercritical', ha='center', fontsize=14)
plt.text(66.85, 10, 'Supercritical\nvapor', rotation=0, fontsize=14)
plt.text(-3.15, 500, 'Supercritical\nliquid', rotation=0, ha='center', fontsize=14)
plt.text(-43.15, 20, 'Liquid', rotation=45, fontsize=14)
plt.text(-3.15, 10, 'Vapor', rotation=45, fontsize=14)

# Axes formatting
plt.ylim(pt, pmax)
plt.xlim(Tmin, Tmax)
plt.yscale('log')
plt.ylabel('Pressure [bar]', fontsize=16)
plt.xlabel('Temperature [°C]', fontsize=16)
plt.grid(True, which="both", ls="--", lw=0.5)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\State_of_art\CO2_pT.eps", format='eps')

plt.show()
