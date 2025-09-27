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
enthalpy_cycle = [h1_kJkg, 480, 340, 340, h1_kJkg]  # kJ/kg

# Optionally compute temperatures
temperature_cycle = []
for h, p in zip(enthalpy_cycle, pressure_cycle):
    state.update(CP.HmassP_INPUTS, h * 1e3, p * 1e5)
    temperature_cycle.append(state.T() - 273.15)

# ---------------------------------------
# 4. Plot P-h diagram with cycle
# ---------------------------------------
plt.figure(figsize=(8,6))
plt.plot(h_liq, P_sat, 'k-', label='Saturation curve')
plt.plot(h_vap, P_sat, 'k-')
plt.plot(enthalpy_cycle, pressure_cycle, 'r-o', label=r'CO$_2$ cycle')

labels = ['1', '2', '3', '4']
for i, label in enumerate(labels):
    if label == '2' or label == '3':
        plt.text(enthalpy_cycle[i], pressure_cycle[i]+2, label, ha='center', fontsize=14)
    elif label == '1':
        plt.text(enthalpy_cycle[i] + 15, pressure_cycle[i]-4, label, ha='center', fontsize=14)
    else:
        plt.text(enthalpy_cycle[i] - 10, pressure_cycle[i]-4, label, ha='center', fontsize=14)

# ---------------------------------------
# 4a. Plot constant temperature curves (for points 3 and 4)
# ---------------------------------------
isotherm_pressures = np.logspace(np.log10(20e5), np.log10(100e5), 300)  # 1 to 100 bar in Pa

for idx in [2, 3]:  # Points 3 and 4
    T_iso_C = temperature_cycle[idx]             # °C
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
    # if len(h_iso) > 0:
    #     mid_idx = len(h_iso) // 2
    if idx == 2:
        plt.text(490, 25, f'{int(T_iso_C)}°C',
                 fontsize=14, color='k', ha='left', va='center')
    else:
        plt.text(460, 25, f'{int(T_iso_C)}°C',
                 fontsize=14, color='k', ha='left', va='center')
        
plt.xlabel(r"Specific enthalpy [kJ.kg$^{-1}$]", fontsize=16)
plt.ylabel("Pressure [bar]", fontsize=16)
plt.xlim([180, 520])
plt.ylim([20, 100])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\State_of_art\CO2_cycle_ph.eps", format='eps')

plt.show()
