import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import numpy as np

from data_filling_CFD import (
    Tc1,
    Tk1, 
    Tr1, 
    Th1,
    Te1, 
    pc1,
    pk1,
    pr1,
    ph1,
    pe1,
    Dc1,
    Dk1,
    Dr1,
    Dh1,
    De1,
    Ve1,
    Vc1,
    mc1,
    mk1,
    mr1,
    mh1, 
    me1, 
    theta1,
    mint_dot1, 
    mout_dot1, 
)
# Load data
csv_path = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\CFD\a_variables.csv'
df = pd.read_csv(csv_path, sep=',')


# ---- Extract all variables from CSV ----
a_theta = df['theta'].values
a_Tc = df['Tc [K]'].values - 273.15
a_Tk = df['Tk [K]'].values - 273.15
a_Tr = df['Tr [K]'].values - 273.15
a_Th = df['Th [K]'].values - 273.15
a_Te = df['Te [K]'].values - 273.15
a_Tk_wall = df['Tk_wall [K]'].values - 273.15
a_Tr_wall = df['Tr_wall [K]'].values - 273.15
a_Th_wall = df['Th_wall [K]'].values - 273.15

a_pc = df['pc [pa]'].values
a_pk = df['pk [pa]'].values
a_pr = df['pr [pa]'].values
a_ph = df['ph [pa]'].values
a_pe = df['pe [pa]'].values

a_Dc = df['Dc [kg/m3]'].values
a_Dk = df['Dk [kg/m3]'].values
a_Dr = df['Dr [kg/m3]'].values
a_Dh = df['Dh [kg/m3]'].values
a_De = df['De [kg/m3]'].values

a_mc = df['mc [g]'].values
a_mk = df['mk [g]'].values
a_mkr = df['mkr [g]'].values
a_mr = df['mr [g]'].values
a_mhr = df['mhr [g]'].values
a_mh = df['mh [g]'].values
a_me = df['me [g]'].values

a_mint_dot = df['mint_dot [kg/s]'].values
a_mout_dot = df['mout_dot [kg/s]'].values

a_Vc = df['Vc [m3]'].values
a_Ve = df['Ve [m3]'].values

# Calculate the expression at each time/angle step
expr = (a_pe * a_Ve - a_pc * a_Vc) * 1e-6

# Summation
total_sum = np.sum(expr)

print("Summation of pe*Ve - pc*Vc =", total_sum)

# ---- Resample everything to theta_ref ----
theta_model = np.linspace(0, 360, len(a_Tc))

# def resample(var):
#     return interp1d(theta_model, var, kind='linear', fill_value="extrapolate")(theta_ref)

# # Resample temperatures and convert to °C
# a_Tc = resample(a_Tc) - 273.15
# a_Tk = resample(a_Tk) - 273.15
# a_Tr = resample(a_Tr) - 273.15
# a_Th = resample(a_Th) - 273.15
# a_Te = resample(a_Te) - 273.15
# a_Tk_wall = resample(a_Tk_wall) - 273.15
# a_Tr_wall = resample(a_Tr_wall) - 273.15
# a_Th_wall = resample(a_Th_wall) - 273.15

# # Resample remaining variables
# a_theta = resample(a_theta)
# a_pc = resample(a_pc)
# a_pk = resample(a_pk)
# a_pr = resample(a_pr)
# a_ph = resample(a_ph)
# a_pe = resample(a_pe)
# a_Dc = resample(a_Dc)
# a_Dk = resample(a_Dk)
# a_Dr = resample(a_Dr)
# a_Dh = resample(a_Dh)
# a_De = resample(a_De)
# a_mc = resample(a_mc)
# a_mk = resample(a_mk)
# a_mkr = resample(a_mkr)
# a_mr = resample(a_mr)
# a_mhr = resample(a_mhr)

# a_mh = resample(a_mh)
# a_me = resample(a_me)
# a_mint_dot = resample(a_mint_dot)
# a_mout_dot = resample(a_mout_dot)
# a_Vc = resample(a_Vc)
# a_Ve = resample(a_Ve)


# Calculate the expression at each time/angle step
expr_cfd = (pe1 * Ve1 - pc1 * Vc1) * 1e-6

# Summation
total_sum = np.sum(expr_cfd)

print("Summation of pe*Ve - pc*Vc =", total_sum)


print(f"{'mdot_in[g/s]':<15}", np.mean(a_mint_dot), np.mean(mint_dot1))
print(f"{'mdot_out[g/s]':<15}", np.mean(a_mout_dot), np.mean(mout_dot1))
print(f"{'Tc [°C]':<15}", np.mean(a_Tc), np.mean(Tc1))
print(f"{'Tk [°C]':<15}", np.mean(a_Tk), np.mean(Tk1))
print(f"{'Tr [°C]':<15}", np.mean(a_Tr), np.mean(Tr1))
print(f"{'Th [°C]':<15}", np.mean(a_Th), np.mean(Th1))
print(f"{'Te [°C]':<15}", np.mean(a_Te), np.mean(Te1))
print(f"{'pc [bar]':<15}", np.mean(a_pc), np.mean(pc1))
print(f"{'pk [bar]':<15}", np.mean(a_pk), np.mean(pk1))
print(f"{'pr [bar]':<15}", np.mean(a_pr), np.mean(pr1))
print(f"{'ph [bar]':<15}", np.mean(a_ph), np.mean(ph1))
print(f"{'pe [bar]':<15}", np.mean(a_pe), np.mean(pe1))
print(f"{'Dc [kg/m^3]':<15}", np.mean(a_Dc), np.mean(Dc1))
print(f"{'Dk [kg/m^3]':<15}", np.mean(a_Dk), np.mean(Dk1))
print(f"{'Dr [kg/m^3]':<15}", np.mean(a_Dr), np.mean(Dr1))
print(f"{'Dh [kg/m^3]':<15}", np.mean(a_Dh), np.mean(Dh1))
print(f"{'De [kg/m^3]':<15}", np.mean(a_De), np.mean(De1))

import matplotlib.pyplot as plt
import os
save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_slow\CFD'
os.makedirs(save_dir, exist_ok=True)

plt.figure(1)
plt.plot(Vc1, pc1,color='b', label='Compression')
plt.plot(a_Vc, a_pc, color='b', linestyle='--')
plt.plot(Ve1, pe1,color='r', label='Expansion')
plt.plot(a_Ve, a_pe, color='r', linestyle='--')
plt.xlabel("Volume [cm$^3$]", fontsize = 14)
plt.ylabel("Pressure [bar]", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_dir, "TC_p-V.eps"), format='eps')

# Temperature comparison
plt.figure(2)
plt.plot(theta1, Tc1, color='b', label="Cold cavity")
plt.plot(a_theta, a_Tc,linestyle='--', color='b')
plt.plot(theta1, Tk1, color='cyan', label="Cooler")
plt.plot(a_theta, a_Tk,linestyle='--', color='cyan')
plt.plot(theta1, Tr1, color='g', label="Regenerator")
plt.plot(a_theta, a_Tr,linestyle='--', color='g')
plt.plot(theta1, Th1, color='orange', label="Heater")
plt.plot(a_theta, a_Th,linestyle='--', color='orange')
plt.plot(theta1, Te1, color='r', label="Hot cavity")
plt.plot(a_theta, a_Te,linestyle='--', color='r')
plt.xlabel("theta [°]", fontsize = 14)
plt.ylabel("Temperature [°C]", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(loc = 'best')
# plt.grid()
plt.savefig(os.path.join(save_dir, "TC_T.eps"), format='eps')

# Density comparison
plt.figure(3)
plt.plot(theta1, Dc1, color='b', label="Cold cavity")
plt.plot(a_theta, a_Dc, linestyle = '--', color='b')
plt.plot(theta1, Dk1, color='cyan', label="Cooler")
plt.plot(a_theta, a_Dk, linestyle = '--', color='cyan')
plt.plot(theta1, Dr1, color='g', label="Regenerator")
plt.plot(a_theta, a_Dr, linestyle = '--', color='g')
plt.plot(theta1, Dh1, color='orange', label="Heater")
plt.plot(a_theta, a_Dh, linestyle = '--', color='orange')
plt.plot(theta1, De1, color='r', label="Hot cavity")
plt.plot(a_theta, a_De, linestyle = '--', color='r')
plt.xlabel("Theta [°]", fontsize = 14)
plt.ylabel("Density [kg/m³]", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_dir, "TC_D.eps"), format='eps')


# Mass flow rate comparison
plt.figure(4)
# Use specified gray shades
color_suction = "#000dff"   # Standard matplotlib blue
color_discharge = "#FF6600FF" # Light blue
plt.plot(theta1, mint_dot1, label="Suction", color=color_suction)
plt.plot(a_theta, a_mint_dot, linestyle='--', color=color_suction)
plt.plot(theta1, mout_dot1, label="Discharge", color=color_discharge)
plt.plot(a_theta, a_mout_dot, linestyle='--', color=color_discharge)
plt.xlabel("theta [°]", fontsize=14)
plt.ylabel("Mass flow rate [g/s]", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid()
plt.savefig(os.path.join(save_dir, "TC_mdot.eps"), format='eps')

# Pressure comparison
plt.figure(5)
plt.plot(theta1, pc1, color='b', label="Cold cavity")
# plt.plot(a_theta, a_pc, linestyle = '--', color='b')
# plt.plot(theta1, pk1, color='cyan', label="Cooler")
# plt.plot(a_theta, a_pk, linestyle = '--', color='cyan')
# plt.plot(theta1, pr1, color='g', label="Regenerator")
# plt.plot(a_theta, a_pr, linestyle = '--', color='g')
# plt.plot(theta1, ph1, color='orange', label="Heater")
# plt.plot(a_theta, a_ph, linestyle = '--', color='orange')
plt.plot(theta1, pe1, color='r', label="Hot cavity")
# plt.plot(a_theta, a_pe, linestyle = '--', color='r')
plt.xlabel("theta [°]", fontsize = 14)
plt.ylabel("Pressure [bar]", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend()
# plt.grid()
# plt.savefig(os.path.join(save_dir, "TC_p.eps"), format='eps')

# Mass comparison
plt.figure(6)
plt.plot(theta1, mc1, color='b', label="Cold cavity")
plt.plot(a_theta, a_mc, linestyle = '--', color='b')
plt.plot(theta1, mk1, color='cyan', label="Cooler")
plt.plot(a_theta, [x + y for x,y in zip(a_mk, a_mkr)], linestyle = '--', color='cyan')
plt.plot(theta1, mr1, color='g', label="Regenerator")
plt.plot(a_theta, a_mr, linestyle = '--', color='g')
plt.plot(theta1, mh1, color='orange', label="Heater")
plt.plot(a_theta, [x + y for x,y in zip(a_mh, a_mhr)], linestyle = '--', color='orange')
plt.plot(theta1, me1, color='r', label="Hot cavity")
plt.plot(a_theta, a_me, linestyle = '--', color='r')
plt.xlabel("theta [°]", fontsize = 14)
plt.ylabel("Mass [g]", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
# plt.legend()
# plt.grid()
plt.savefig(os.path.join(save_dir, "TC_m.eps"), format='eps')



from matplotlib.patches import FancyArrowPatch, Arc

# ---- Data Preparation ----
a_Vt = [x + y for x, y in zip(a_Vc, a_Ve)]
a_pt = a_pe  # If no total pressure calculated

# Downsample arrays
step = 10
Vc = np.array(a_Vc)[::step]
Ve = np.array(a_Ve)[::step]
Vt = np.array(a_Vt)[::step]
pc = np.array(a_pc)[::step]
pe = np.array(a_pe)[::step]
pt = np.array(a_pt)[::step]

# ---- Plot Setup ----
plt.figure(7)

# Compression space
plt.plot(Vc, pc, 'b-', linewidth=2, label='Cold cavity')

# Expansion space
plt.plot(Ve, pe, 'r-', linewidth=2, label='Hot cavity')

# Whole system
plt.plot(Vt, pt, color='k', linewidth=1.5, label='Total working space')
plt.fill(Vt, pt, color='gray', alpha=0.2, zorder=0)

# ---- Add offset arrows ----

# Example arrow beside compression curve
arrow_c = FancyArrowPatch(
    posA=(235 + 10, 52),
    posB=(235 - 15, 52 + 2),
    arrowstyle='->',
    color='blue',
    mutation_scale=30,
    linewidth=1
)
arrow_c.set_clip_on(False)
plt.gca().add_patch(arrow_c)


arrow_e = FancyArrowPatch(
    posA=(160 - 10, 52),
    posB=(160 + 15, 52 + 2),
    arrowstyle='->',
    color='red',
    mutation_scale=30,
    linewidth=1
)
arrow_e.set_clip_on(False)
plt.gca().add_patch(arrow_e)

# # Whole system arrow (offset horizontally to the right)

arrow_t = FancyArrowPatch(
    posA=(390 - 8, 52 + 2),
    posB=(390 - 10, 52),
    arrowstyle='->',
    color='black',
    mutation_scale=30,
    linewidth=1
)
arrow_t.set_clip_on(False)
plt.gca().add_patch(arrow_t)


# # ---- Mechanical Power Label ----
# mid_idx = len(Vt) // 2
# plt.text(Vt[mid_idx] - 10, pt[mid_idx], 'Mechanical power', fontsize=12)

# ---- Axes and Formatting ----
plt.xlabel("Volume [cm$^3$]", fontsize=14)
plt.ylabel("Pressure [bar]", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim([44, 66])
plt.legend(fontsize=12, loc='best')
# plt.grid(True)
plt.tight_layout()

# ---- Save and Show ----
plt.savefig(os.path.join(save_dir, "TC_p-V_tot.eps"), format='eps', dpi=300)

plt.figure(8)
plt.plot(theta1, Tk1, color='cyan', label="Cooler")
plt.plot(a_theta, a_Tk,linestyle='--', color='cyan')
plt.plot(a_theta, a_Tk_wall, linestyle='-.', color='cyan')

plt.plot(theta1, Tr1, color='g', label="Regenerator")
plt.plot(a_theta, a_Tr,linestyle='--', color='g')
plt.plot(a_theta, a_Tr_wall,linestyle='-.', color='g')

plt.plot(theta1, Th1, color='orange', label="Heater")
plt.plot(a_theta, a_Th,linestyle='--', color='orange')
plt.plot(a_theta, a_Th_wall,linestyle='-.', color='orange')

plt.xlabel("theta [°]", fontsize = 14)
plt.ylabel("Temperature [°C]", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(loc = 'best')

Dp_cfd = [(x - y) for x,y in zip(pc1, pe1)]
Dp_model = [(x - y) for x,y in zip(a_pc, a_pe)]

plt.figure(9)
plt.plot(theta1, Dp_cfd, color='purple')
plt.plot(a_theta, Dp_model,linestyle='--', color='purple')

plt.xlabel("theta [°]", fontsize = 14)
plt.ylabel("Pressure [bar]", fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.legend(loc = 'best')
plt.show()