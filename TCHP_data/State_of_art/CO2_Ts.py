import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

fluid = "CO2"

# Critical properties
T_crit = PropsSI("TCRIT", fluid) - 273.15  # °C
p_crit = PropsSI("PCRIT", fluid)

# Temperature range for isobar at critical pressure (in Kelvin)
T_vals = np.linspace(220, 440, 500)  # Kelvin
s_vals = []

for T in T_vals:
    try:
        s = PropsSI("S", "T", T, "P", p_crit, fluid) / 1000  # Convert J/kg·K to kJ/kg·K
        s_vals.append(s)
    except:
        s_vals.append(np.nan)

# Saturation curve
Tsat = np.linspace(PropsSI("Ttriple", fluid), PropsSI("Tcrit", fluid), 500)
sL = [PropsSI("S", "T", T, "Q", 0, fluid)/1000 for T in Tsat]
sV = [PropsSI("S", "T", T, "Q", 1, fluid)/1000 for T in Tsat]
Tsat_C = Tsat - 273.15  # °C

# Critical point
s_crit = PropsSI("S", "T", PropsSI("Tcrit", fluid), "P", PropsSI("Pcrit", fluid), fluid)/1000
T_crit_C = PropsSI("Tcrit", fluid) - 273.15

# Plotting
plt.figure(figsize=(8,6))

# Saturation dome
plt.plot(sL, Tsat_C, 'k', lw=2)
plt.plot(sV, Tsat_C, 'k', lw=2)

# Isobar at critical pressure
plt.plot(s_vals, T_vals - 273.15, 'gray', lw=2, label = 'Critical Pressure') #, label="$p = p_{cr}$")

# Critical point
plt.plot(s_crit, T_crit_C, 'g^', markersize=10, label="Critical point")

# Labels
plt.xlabel("Entropy [kJ/kg·K]", fontsize=16)
plt.ylabel("Temperature [°C]", fontsize=16)

# Text annotations
plt.text(0.8, -30, "Liquid", fontsize=14)
plt.text(1.3, -20, "Two phase", fontsize=14)
plt.text(2.05, 0, "Vapor", fontsize=14)
plt.text(0.75, 10, "Supercritical\nliquid", fontsize=14)
plt.text(2.0, 50, "Supercritical\nvapor", fontsize=14)
plt.text(1.4, 100, "Supercritical", fontsize=14)

plt.axhline(y=T_crit_C, color='k', linestyle='--', label = 'Critical Temperature')
# plt.text(2.25, T_crit_C + 2, r"$T = T_{cr}$", fontsize=12)

# plt.text(2.25, 130, r"$p = p_{cr}$", fontsize=12)

plt.legend()
plt.grid(True)
plt.xlim(0.5, 2.5)
plt.ylim(-50, 150)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\State_of_art\CO2_Ts.eps", format='eps')

plt.show()
