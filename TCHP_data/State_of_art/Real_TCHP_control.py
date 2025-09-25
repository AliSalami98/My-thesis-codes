import csv
import numpy as np
import matplotlib.pyplot as plt

# import pandas as pd

# # Define file paths
# excel_file = r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\cycle measurements\test.xlsx'
# csv_file = r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\cycle measurements\test.csv'

# # Read the Excel file
# df = pd.read_excel(excel_file)

# # Save the dataframe to CSV
# df.to_csv(csv_file, index=False)
a_Two = []
a_Two_sp = []
a_Hpev = []
a_Lpev = []
a_omegab = []
a_omega1 = []
a_omega2 = []
a_omega3 = []
a_Theater1 = []
a_Theater2 = []
a_Theater_sp = []
a_pc = []
a_pc_sp = []

with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\cycle measurements\test.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        a_Two.append(float(row['Tw_out [°C]'])+ 273.15)
        a_Two_sp.append(float(row['Tw_out_sp [°C]'])+ 273.15)
        a_Hpev.append(float(row['Hpev']))
        a_Lpev.append(float(row['Lpev']))
        a_Theater1.append(float(row['Theater1 [°C]'])+ 273.15)
        a_Theater_sp.append(float(row['Theater_sp [°C]'])+ 273.15)
        a_Theater2.append(float(row['Theater2 [°C]'])+ 273.15)
        a_pc.append(float(row['pc [bar]'])* 1e5)
        a_pc_sp.append(float(row['pc_sp [bar]'])* 1e5)
        a_omegab.append(float(row['omegab [rpm]']))
        a_omega1.append(float(row['omega1 [rpm]']))
        a_omega2.append(float(row['omega2 [rpm]']))
        a_omega3.append(float(row['omega3 [rpm]']))


t = np.linspace(0, len(a_Two), len(a_Two))

import matplotlib.pyplot as plt

a_Two = np.array(a_Two) - 273.15
a_Two_sp = np.array(a_Two_sp) - 273.15
a_Theater1 = np.array(a_Theater1) - 273.15
a_Theater2 = np.array(a_Theater2) - 273.15
a_Theater_sp = np.array(a_Theater_sp) - 273.15

# Pressure: Pascal to bar
a_pc = np.array(a_pc) / 1e5
a_pc_sp = np.array(a_pc_sp) / 1e5
# ------------------------------
# Figure 1: Temperatures & Pressure
# ------------------------------
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))

# Supply Temperature
ax1.plot(t, a_Two,  linewidth=2.3,  color='#B22222', label='Measured', zorder = 3)       # firebrick
ax1.plot(t, a_Two_sp, linestyle=':', color='k', linewidth=2, label='Setpoint', zorder = 2) 
ax1.set_ylabel(r"$T_\text{w, sup}$ [°C]", fontsize=16)
ax1.grid(True, zorder = 0)
ax1.legend(fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=14)

# Heater Temperatures
ax2.plot(t, a_Theater1, linewidth=2.3,  color='#8A2BE2', label='Heater1', zorder = 3)   # darkorange
ax2.plot(t, a_Theater2,  linewidth=2.3, color='#DA70D6', label='Heater2', zorder = 3)  # chocolate
ax2.plot(t, a_Theater_sp, linestyle='-.', color='k', linewidth=2, label='Setpoint', zorder = 2) 
ax2.set_ylabel(r"$T_\text{h}$ [°C]", fontsize=16)
ax2.grid(True, zorder = 0)
ax2.legend(fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=14)

# Pressure
ax3.plot(t, a_pc, linewidth=2.3, color='#006400', label='Measured', zorder = 3)      # teal
ax3.plot(t, a_pc_sp, linestyle=':', color='k', linewidth=2, label='Setpoint', zorder = 2) 
ax3.set_ylabel(r"$p_\text{gc}$ [bar]", fontsize=16)
ax3.set_xlabel("Time [s]", fontsize=16)
ax3.grid(True, zorder = 0)
ax3.legend(fontsize=12)
ax3.tick_params(axis='both', which='major', labelsize=14)

fig1.tight_layout()

# Figure 2: Control Inputs
fig2, (ax4, ax5) = plt.subplots(2, 1, figsize=(12, 6))

# Fan Speed
ax4.plot(t, a_omegab, linewidth=2.3, color="#FFD700", label='Burner fan', zorder = 3)   # slateblue
ax4.set_ylabel(r"$\omega_\text{bf}$ [rpm]", fontsize=16)
ax4.grid(True, zorder = 0)
ax4.legend(fontsize=12)
ax4.tick_params(axis='both', which='major', labelsize=14)

# HPV Opening
ax5.plot(t, a_Hpev, linewidth=2.3, color='#6A5ACD', label='HPV', zorder = 3)         # gray
ax5.set_ylabel(r"$\varphi_\text{hpv}$ [%]", fontsize=16)
ax5.set_xlabel("Time [s]", fontsize=16)
ax5.grid(True, zorder = 0)
ax5.legend(fontsize=12)
ax5.tick_params(axis='both', which='major', labelsize=14)

fig2.tight_layout()

fig1.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\control\real_TCHP_control_y.eps", format='eps', bbox_inches='tight')
fig2.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\control\real_TCHP_control_u.eps", format='eps', bbox_inches='tight')

plt.show()