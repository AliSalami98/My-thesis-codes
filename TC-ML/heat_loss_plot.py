import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt

# Create an AbstractState object using the HEOS backend and CO2
T0 = 298.15
p0 = 101325
state = AbstractState("HEOS", "CO2")
state.update(CP.PT_INPUTS, p0, T0)
h0 = state.hmass()
s0 = state.smass()
CP.set_reference_state('CO2',T0, p0, h0, s0)

data = {
    "Tin [K]": [],
    "pin [pa]": [],
    "pout [pa]": [],
    "Th_wall [K]": [],
    "Tw_in [K]": [],
    "omega [rpm]": [],
    "mdot [kg/s]": [],
    "Pcomb [W]": [],
    "Pheating [W]": [],
    "Pcooling [W]": [],
    "Pmotor [W]": [],
    "Pout [W]": [],
    "Tout [K]": [],
    'sin [J/kg]': [],
    'sout [J/kg]': [],
    'hin [J/kg]': [],
    'hout [J/kg]': [],

}
psi_in = []
psi_out = []
Pout = []
Ex_h = []
Ex_k = []
Ex_out = []
eff_ind = []
eff_Ex = []
Pr = []
Res = []
Res_ratio = []
counter = 0
# with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\I-O data4.csv') as csv_file:    
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\TC experiments\all.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        counter += 1
        # data['Tin [K]'].append(float(row['Tin [°C]']) + 273.15)
        data['pin [pa]'].append(float(row['pin [bar]'])*10**5)
        # state.update(CP.PT_INPUTS, float(row['pin [bar]'])*10**5, float(row['Tin [°C]']) + 273.15)
        # data['sin [J/kg]'].append(state.smass())
        # data['hin [J/kg]'].append(state.hmass())
        data['pout [pa]'].append(float(row['pout [bar]'])*10**5)
        data['Th_wall [K]'].append(float(row['Th_wall[°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        data['mdot [kg/s]'].append(float(row['mdot [g/s]'])*10**(-3))
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Pout [W]'].append(float(row['Pout [W]']))
        data['Tout [K]'].append(float(row['Tout [°C]']) + 273.15)
        Pr.append(data['pout [pa]'][-1]/data['pin [pa]'][-1])
        Res.append(data['Pheating [W]'][-1] + data['Pmotor [W]'][-1] - data['Pcooling [W]'][-1] - data['Pout [W]'][-1])
        Res_ratio.append(Res[-1]/data['Pcomb [W]'][-1])


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

points = np.linspace(0, len(Res) - 1, len(Res))
avg_res = np.mean(Res)

# Plot setup
plt.figure(figsize=(8, 6))

# Plot scatter points
plt.scatter(points, Res, label='Thermal Compressor', color='darkorange', s=35, edgecolors='black', alpha=0.8)

# Axis labels
plt.xlabel('Samples', fontsize=12)
plt.ylabel('Residual [W]', fontsize=12)

# Grid and ticks
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Force integer ticks on x-axis
plt.gca().xaxis.set_major_locator(MultipleLocator(10))

# Horizontal lines
plt.axhline(0, color='gray', linewidth=1, linestyle='--', label='Zero Line')
plt.axhline(avg_res, color='red', linewidth=1.5, linestyle='--', label=f'Average: {avg_res:.2f} W')

# Legend
plt.legend(loc='best', fontsize=10, frameon=False)

# Show plot
plt.tight_layout()
plt.show()

# # Extract arrays
# omega = np.array(data['omega [rpm]'])
# Pr = np.array(Pr)
# Pmotor = np.array(data['Pmotor [W]'])
# Pheating = np.array(data['Pheating [W]'])
# Pcooling = np.array(data['Pcooling [W]'])
# Pout = np.array(data['Pout [W]'])  # Make sure this is populated correctly

# # Set up 4 subplots
# fig, axs = plt.subplots(1, 4, figsize=(22, 6), sharey=True)

# # --- Motor Power ---
# sc1 = axs[0].scatter(omega, Pr, c=Pmotor, cmap='Reds', edgecolor='k', s=60)
# axs[0].set_title('Motor Power')
# axs[0].set_xlabel('Speed [rpm]')
# axs[0].set_ylabel('Pressure Ratio [-]')
# cbar1 = plt.colorbar(sc1, ax=axs[0])
# cbar1.set_label('Power [W]')

# # --- Heating Power ---
# sc2 = axs[1].scatter(omega, Pr, c=Pheating, cmap='Greens', edgecolor='k', s=60)
# axs[1].set_title('Heating Power')
# axs[1].set_xlabel('Speed [rpm]')
# cbar2 = plt.colorbar(sc2, ax=axs[1])
# cbar2.set_label('Power [W]')

# # --- Cooling Power ---
# sc3 = axs[2].scatter(omega, Pr, c=Pcooling, cmap='Blues', edgecolor='k', s=60)
# axs[2].set_title('Cooling Power')
# axs[2].set_xlabel('Speed [rpm]')
# cbar3 = plt.colorbar(sc3, ax=axs[2])
# cbar3.set_label('Power [W]')

# # --- Output Power ---
# sc4 = axs[3].scatter(omega, Pr, c=Pout, cmap='Purples', edgecolor='k', s=60)
# axs[3].set_title('Output Power')
# axs[3].set_xlabel('Speed [rpm]')
# cbar4 = plt.colorbar(sc4, ax=axs[3])
# cbar4.set_label('Power [W]')

# # Styling
# for ax in axs:
#     ax.grid(True, alpha=0.3)

# # plt.suptitle('Power vs Compressor Speed and Pressure Ratio', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title
# plt.show()
