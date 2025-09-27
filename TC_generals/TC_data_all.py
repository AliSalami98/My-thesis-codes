import csv
import matplotlib.pyplot as plt
import numpy as np
from utils import get_state, CP, h0, T0, s0, p0
import os
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
data = {
    "Tsuc [K]": [], "psuc [pa]": [], "pdis [pa]": [],
    "Theater [K]": [], "Tw_in [K]": [], "omega [rpm]": [], "omegab [rpm]": [],
    "mdot [kg/s]": [], "Pcomb [W]": [], "Pheating [W]": [],
    "Pcooling [W]": [], "Pmotor [W]": [], "Tdis [K]": [],
    "Pcomp [W]": [], "eff [%]": [], "Ploss [W]": [], "Pmech [W]": []
}


a_Pr = []
a_Theater = []
a_pcharged = []
a_omega = []
eff_Ex = []
i = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\TC experiments\all_filtered_2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        data['Tsuc [K]'].append(float(row['Tsuc [°C]']) + 273.15)
        data['psuc [pa]'].append(float(row['psuc [bar]']) * 1e5)
        data['pdis [pa]'].append(float(row['pdis [bar]']) * 1e5)
        data['Theater [K]'].append(float(row['Theater [°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        # data['omegab [rpm]'].append(float(row['omegab [rpm]']))
        data['mdot [kg/s]'].append(float(row['mdot [g/s]']) * 1e-3)
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Pmech [W]'].append(float(row['Pmech [W]']))
        data['Tdis [K]'].append(float(row['Tdis [°C]']) + 273.15)
        data['Pcomp [W]'].append(float(row['Pcomp [W]']))
        # data['eff [%]'].append(100 * float(row['eff [%]']))
        data['Ploss [W]'].append(float(row['Ploss [W]']))


        mw_dot = 10/60
        Theater = data['Theater [K]'][-1]
        omega = data['omega [rpm]'][-1]
        Twi = data['Tw_in [K]'][-1]
        T2 = data['Tdis [K]'][-1]
        p2 = data['pdis [pa]'][-1]
        mdot = data['mdot [kg/s]'][-1]
        state = get_state(CP.PT_INPUTS, p2, T2)
        h2, s2 = state.hmass(), state.smass()

        T1 = data['Tsuc [K]'][-1]
        p1 = data['psuc [pa]'][-1]
        state = get_state(CP.PT_INPUTS, p1, T1)
        h1, s1 = state.hmass(), state.smass()
        # --- Compression power and exergy output ---
        Pcomp = data['Pcomp [W]'][-1]
        Pcool  = data['Pcooling [W]'][-1]
        Pheat  = data['Pheating [W]'][-1]
        Pmech  = data['Pmech [W]'][-1]
        psi1 = (h1 - h0) - T0 * (s1 - s0)
        psi2 = (h2 - h0) - T0 * (s2 - s0)
        Ex_comp = mdot * (psi2 - psi1)

        # --- Water side (cooling) ---
        hwi = CP.PropsSI('H', 'P', p0, 'T', Twi, 'Water')
        swi = CP.PropsSI('S', 'P', p0, 'T', Twi, 'Water')
        hwo = hwi + Pcool / mw_dot
        swo = CP.PropsSI('S', 'P', p0, 'H', hwo, 'Water')

        psi_wi = (hwi - h0) - T0 * (swi - s0)
        psi_wo = (hwo - h0) - T0 * (swo - s0)
        Ex_cooler = mw_dot * (psi_wo - psi_wi)

        # --- Exergy terms ---
        Ex_heater = (1 - T0 / Theater) * Pheat
        X_flow   = mdot * T0 * (s2 - s1)
        X_transfer  = -T0 / Theater * Pheat + T0 / Twi * Pcool
        X_heater    = T0 / Theater * Pheat
        X_cooler    = T0 / Twi * Pcool
        X_total     = Ex_heater + Pmech - Ex_comp - Ex_cooler

        # --- Efficiencies ---
        eff_Ex.append(100 * (1 - X_total / (Ex_heater + Pmech)))
        a_Pr.append(p2/p1)
        a_Theater.append(Theater)
        a_omega.append(omega)
        a_pcharged.append(np.sqrt(p1 * p2))
        # i+= 1
        # if i > 26:
        #     break
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_data'
os.makedirs(save_dir, exist_ok=True)

# Convert lists to NumPy arrays
a_Theater = np.array(a_Theater)
a_omega = np.array(a_omega)
a_Pr = np.array(a_Pr)
eff_Ex = np.array(eff_Ex)
a_pcharged = np.array(a_pcharged) * 1e-5

# Define masks for 700°C and 800°C (in Kelvin)
mask_700 = (a_Theater >= 695 + 273.15) & (a_Theater <= 705 + 273.15)
mask_800 = (a_Theater >= 795 + 273.15) & (a_Theater <= 805 + 273.15)

# Font sizes
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 12

# --- Figure 1: Exergy Efficiency vs. Omega ---
plt.figure(figsize=(7, 5))
sc1 = plt.scatter(a_omega[mask_700], eff_Ex[mask_700], c=a_pcharged[mask_700],
                  cmap='viridis', marker='x', label=r'$T_\text{h} = 700\,^\circ$C',
                  s=70, edgecolors='k')

sc2 = plt.scatter(a_omega[mask_800], eff_Ex[mask_800], c=a_pcharged[mask_800],
                  cmap='viridis', marker='o', label=r'$T_\text{h} = 800\,^\circ$C',
                  s=70, edgecolors='k')

plt.xlabel(r"Rotational speed $\omega_\text{m}$ [rpm]", fontsize=label_fontsize)
plt.ylabel(r'Exergy efficiency $\eta_\text{tc, ex}$ [%]', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
cbar = plt.colorbar(sc2)
cbar.set_label(r"Charged pressure $p_\text{charged}$ [bar]", fontsize=14)
cbar.ax.tick_params(labelsize=12)
# plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "exergy_omega.eps"), format='eps')

# --- Figure 2: Exergy Efficiency vs. Pressure Ratio ---
plt.figure(figsize=(7, 5))
sc3 = plt.scatter(a_Pr[mask_700], eff_Ex[mask_700], c=a_pcharged[mask_700],
                  cmap='viridis', marker='x', label=r'$T_\text{h} = 700\,^\circ$C',
                  s=70, edgecolors='k')

sc4 = plt.scatter(a_Pr[mask_800], eff_Ex[mask_800], c=a_pcharged[mask_800],
                  cmap='viridis', marker='o', label=r'$T_\text{h} = 800\,^\circ$C',
                  s=70, edgecolors='k')

plt.xlabel(r'Pressure ratio $r_\text{p}$ [-]', fontsize=label_fontsize)
plt.ylabel(r'Exergy efficiency $\eta_\text{tc, ex}$ [%]', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)
cbar = plt.colorbar(sc4)
cbar.set_label(r"Charged pressure $p_\text{charged}$ [bar]", fontsize=14)
cbar.ax.tick_params(labelsize=12)
plt.legend(fontsize=legend_fontsize)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "exergy_rp.eps"), format='eps')

plt.show()