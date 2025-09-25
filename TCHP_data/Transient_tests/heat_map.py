import csv
import numpy as np
from utils import (
	get_state,
)
import CoolProp.CoolProp as CP

# Initialize lists to hold the column data
t = []
Hpev = []
Lpev = []
mw_dot = []
mmpg_dot = []
Tw_in = []
Tmpg_in = []
Tw_out = []
Tmpg_out = []
pc = []
p25 = []
pi = []
pe = []
Tc_out = []
Te_out = []
omega1 = []
omega2 = []
omega3 = []
omegab = []
Theater1 = []
Theater2 = []
Pcomb = []
Pheat_out = []
Pevap = []
Pm1 = []
Pm2 = []
Pm3 = []
COP = []
SH = []
i = 0
cp_w = 4186

mpg_percentage = 25
cp_mpg = (100 - mpg_percentage)/100*cp_w + mpg_percentage/100 * 0.6 * cp_w

step_counter = 0
# Read the CSV file
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_state
import CoolProp.CoolProp as CP

# ------------------------------
# Files and Input-Output Mapping
# ------------------------------
step_test_files = {
    'Lpev': 'LPV PRBS.csv',
    'Hpev': 'HPV PRBS.csv',
    'omegab': 'bf step.csv',
    'omega1': 'M1 step.csv',
    'omega2': 'M2 step.csv',
    'omega3': 'M3 step.csv',
    'Tw_in': 'Tw step.csv',
    'mw_dot': 'mw step.csv',
    'Tmpg_in': 'Tmpg step.csv',
    'mmpg_dot': 'mmpg step.csv'
}

# Define folder path (update to your actual path)
folder_path = r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in'

# All outputs to include
output_keys = [
    'Tw_out', 'Tmpg_out', 'Tc_out', 'Te_out', 'Theater1', 'Theater2',
    'pc', 'pe', 'pi', 'SH',
    'COP', 'Pcomb', 'Pheat_out', 'Pevap',
    'Pm1', 'Pm2', 'Pm3',
]

# Create a DataFrame to hold correlations: inputs x outputs
correlation_matrix = pd.DataFrame(index=step_test_files.keys(), columns=output_keys)

# ------------------------------
# Main loop for all input files
# ------------------------------
for input_var, filename in step_test_files.items():
    file_path = os.path.join(folder_path, filename)
    
    # Initialize lists
    vars_dict = {key: [] for key in ['omegab', 'Tw_in', 'Tmpg_in', 'mmpg_dot', 'mw_dot',
                                     'Hpev', 'Lpev', 'omega1', 'omega2', 'omega3',
                                     'Theater1', 'Theater2', 'pc', 'p25', 'pi', 'pe',
                                     'Tc_out', 'Te_out', 'Pheat_out', 'Pm1', 'Pm2', 'Pm3',
                                     'Tw_out', 'Tmpg_out']}
    
    with open(file_path, newline='') as csv_file:
        reader = pd.read_csv(csv_file, delimiter=';')
        for col in reader.columns:
            for key in vars_dict:
                if key in col:
                    try:
                        values = pd.to_numeric(reader[col], errors='coerce')

                        # Only assign to vars_dict if the column contains at least one non-NaN value
                        if values.notna().any():
                            vars_dict[key] = values.tolist()
                        # Else, skip and leave the list empty
                    except Exception as e:
                        print(f"Error in column {col}: {e}")

    # Shortcuts
    omegab = vars_dict['omegab']
    Tw_in = vars_dict['Tw_in']
    Tmpg_in = vars_dict['Tmpg_in']
    mmpg_dot = np.array(vars_dict['mmpg_dot']) / 60  # l/min to kg/s
    mw_dot = np.array(vars_dict['mw_dot']) / 60
    Pm1 = vars_dict['Pm1']
    Pm2 = vars_dict['Pm2']
    Pm3 = vars_dict['Pm3']
    Hpev = vars_dict['Hpev']
    Lpev = vars_dict['Lpev']
    omega1 = vars_dict['omega1']
    omega2 = vars_dict['omega2']
    omega3 = vars_dict['omega3']
    Theater1 = vars_dict['Theater1']
    Theater2 = vars_dict['Theater2']
    pc = vars_dict['pc']
    pi = vars_dict['pi']
    pe = vars_dict['pe']
    Tc_out = vars_dict['Tc_out']
    Te_out = vars_dict['Te_out']
    Pheat_out = vars_dict['Pheat_out']
    Tw_out = vars_dict['Tw_out']
    Tmpg_out = vars_dict['Tmpg_out']

    Te_out = [x for x in Te_out if not np.isnan(x)]
    Tc_out = [x for x in Tc_out if not np.isnan(x)]
    Pm1 = [x for x in Pm1 if not np.isnan(x)]
    Pm2 = [x for x in Pm2 if not np.isnan(x)]
    Pm3 = [x for x in Pm3 if not np.isnan(x)]

    Te_out = np.interp(np.linspace(0, len(Te_out) - 1, len(omegab)), np.arange(len(Te_out)), Te_out)
    Tc_out = np.interp(np.linspace(0, len(Tc_out) - 1, len(omegab)), np.arange(len(Tc_out)), Tc_out)
    Pm1 = np.interp(np.linspace(0, len(Pm1) - 1, len(omegab)), np.arange(len(Pm1)), Pm1)
    Pm2 = np.interp(np.linspace(0, len(Pm2) - 1, len(omegab)), np.arange(len(Pm2)), Pm2)
    Pm3 = np.interp(np.linspace(0, len(Pm3) - 1, len(omegab)), np.arange(len(Pm3)), Pm3)
    # print(Pm1)

    cp_w = 4186
    mpg_percentage = 25
    cp_mpg = (100 - mpg_percentage)/100*cp_w + mpg_percentage/100 * 0.6 * cp_w

    # Derived variables
    COP, Pcomb, Pevap, SH = [], [], [], []
    LHV = 50e6

    for i in range(len(omegab)):
        try:
            mCH4_dot = (0.0022 * omegab[i] - 2.5965) * 0.657 / 60000
            Pcomb.append(LHV * mCH4_dot)
            Pevap.append(mmpg_dot[i] * cp_mpg * (Tmpg_in[i] - Tmpg_out[i]))
            state = get_state(CP.PQ_INPUTS, pe[i] * 1e5, 1)
            SH.append(Te_out[i] - state.T())
            COP.append(Pheat_out[i] / Pcomb[-1] if Pcomb[-1] != 0 else np.nan)
        except:
            Pcomb.append(np.nan)
            COP.append(np.nan)
            Pevap.append(np.nan)
            SH.append(np.nan)

    # Create DataFrame
    df = pd.DataFrame({
        input_var: vars_dict[input_var],
        "Tw_out": Tw_out,
        "Tmpg_out": Tmpg_out,
        "Tc_out": Tc_out,
        "Te_out": Te_out,
        "Theater1": Theater1,
        "Theater2": Theater2,
        "pc": pc,
        "pe": pe,
        "pi": pi,
        "SH": SH,
        "COP": COP,
        "Pcomb": Pcomb,
        "Pheat_out": Pheat_out,
        "Pevap": Pevap,
        "Pm1": Pm1,
        "Pm2": Pm2,
        "Pm3": Pm3,
    }).dropna()

    # Correlation with respect to this input
    corr = df.corr().loc[input_var, output_keys]
    correlation_matrix.loc[input_var] = corr

    latex_labels = {
        # OUTPUTS
        "Tw_out": r"$T_{w,\mathrm{out}}$",
        "Tmpg_out": r"$T_{\mathrm{mpg,\,out}}$",
        "Tc_out": r"$T_{gc,\mathrm{out}}$",
        "Te_out": r"$T_{evap,\mathrm{out}}$",
        "Theater1": r"$T_{\mathrm{heater1}}$",
        "Theater2": r"$T_{\mathrm{heater2}}$",
        "pc": r"$p_{\mathrm{gc}}$",
        "pe": r"$p_{\mathrm{evap}}$",
        "pi": r"$p_{\mathrm{i}}$",
        "SH": r"$\mathrm{SH}$",
        "COP": r"$\mathrm{COP}$",
        "Pcomb": r"$\dot{Q}_{\mathrm{comb}}$",
        "Pheat_out": r"$\dot{Q}_{\mathrm{heat\,out}}$",
        "Pevap": r"$\dot{Q}_{\mathrm{evap}}$",
        "Pm1": r"$P_{\mathrm{m1}}$",
        "Pm2": r"$P_{\mathrm{m2}}$",
        "Pm3": r"$P_{\mathrm{m3}}$",

        # INPUTS (used as index in correlation_matrix)
        "Lpev": r"$\mathrm{OP_{LPV}}$",
        "Hpev": r"$\mathrm{OP_{HPV}}$",
        "omegab": r"$\omega_{\mathrm{b}}$",
        "omega1": r"$\omega_1$",
        "omega2": r"$\omega_2$",
        "omega3": r"$\omega_3$",
        "Tw_in": r"$T_{w,\mathrm{in}}$",
        "Tmpg_in": r"$T_{\mathrm{mpg,\,in}}$",
        "mw_dot": r"$\dot{m}_w$",
        "mmpg_dot": r"$\dot{m}_{\mathrm{mpg}}$"
    }


# Apply LaTeX labels to correlation matrix axes
correlation_matrix = correlation_matrix.rename(index=latex_labels, columns=latex_labels)
plt.figure(figsize=(16, 8))

# Transpose the correlation matrix to switch axes
ax = sns.heatmap(
    correlation_matrix.astype(float).T,   # Transposed here
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt=".2f",
    annot_kws={"fontsize": 14},
    cbar_kws={"label": "Correlation Coefficient"},
    linewidths=0.5,
    linecolor='gray'
)

# Update labels accordingly
plt.xlabel("Input Step Test", fontsize=18, labelpad=20)
plt.ylabel("Output Variables", fontsize=18, labelpad=20)

# Ticks and labels
plt.xticks(fontsize=14, rotation=45, ha='right')
plt.yticks(fontsize=14, rotation=0)

# Colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
cbar.set_label("Correlation Coefficient", fontsize=18, labelpad=15)

plt.tight_layout()

plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\transient\TCHP_IO_heat_map.eps", format='eps', bbox_inches='tight')

plt.show()




# import matplotlib.pyplot as plt

# # Create time vector
# time = np.arange(len(omegab))

# # Setup 5 subplots
# fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# # Set global font sizes
# label_fontsize = 14
# tick_fontsize = 12
# legend_fontsize = 12

# # 1. COP
# axs[0].plot(time, COP, label='COP', color='tab:green')
# axs[0].set_ylabel("COP", fontsize=label_fontsize)
# axs[0].tick_params(axis='both', labelsize=tick_fontsize)
# axs[0].legend(fontsize=legend_fontsize)
# axs[0].grid(True)

# # 2. Powers
# axs[1].plot(time, Pcomb, label='Combustion')
# axs[1].plot(time, Pheat_out, label='Heat output')
# axs[1].plot(time, Pevap, label='Evaporation')
# axs[1].set_ylabel("Power [W]", fontsize=label_fontsize)
# axs[1].tick_params(axis='both', labelsize=tick_fontsize)
# axs[1].legend(fontsize=legend_fontsize)
# axs[1].grid(True)

# # 3. Temperatures (dual axis)
# # 3. Heater Temperatures (single y-axis)
# axs[2].plot(time, Theater1, label='Heater1', linestyle='-', color='tab:red')
# axs[2].plot(time, Theater2, label='Heater2', linestyle='-', color='tab:orange')
# axs[2].set_ylabel("Heater Temp [°C]", fontsize=label_fontsize)
# axs[2].tick_params(axis='both', labelsize=tick_fontsize)
# axs[2].legend(loc='upper right', fontsize=legend_fontsize)
# axs[2].grid(True)


# # 4. Pressure
# axs[3].plot(time, pc, label='High', linestyle='-.')
# axs[3].plot(time, pe, label='Low', linestyle='-.')
# axs[3].plot(time, pi, label='Intermediary', linestyle='-.')
# axs[3].set_ylabel("Pressure [Bar]", fontsize=label_fontsize)
# axs[3].tick_params(axis='both', labelsize=tick_fontsize)
# axs[3].legend(fontsize=legend_fontsize)
# axs[3].grid(True)

# # 5. Valve openings
# axs[4].plot(time, Lpev, label='Low-pressure', color='tab:blue')
# axs[4].plot(time, Hpev, label='High-pressure', color='tab:red')
# axs[4].set_ylabel("Valve opening [%]", fontsize=label_fontsize)
# axs[4].set_xlabel("Time [s]", fontsize=label_fontsize)
# axs[4].tick_params(axis='both', labelsize=tick_fontsize)
# axs[4].legend(fontsize=legend_fontsize)
# axs[4].grid(True)

# # # 4. Input: Lpev
# # axs[4].plot(time, omegab, label='Burner fan', color='tab:brown')
# # # axs[4].plot(time, omega1, label='Motor1', color='tab:blue')
# # # axs[4].plot(time, omega2, label='Motor2', color='tab:red')
# # # axs[4].plot(time, omega3, label='Motor3', color='tab:orange')
# # axs[4].set_ylabel("Speed [rpm]")
# # axs[4].set_xlabel("Time [s]")
# # # axs[4].set_title("Input Signal")
# # axs[4].legend()
# # axs[4].grid(True)

# # # 4. Input: Lpev
# # axs[4].plot(time, Tw_in, label='Water inlet', color='tab:red')
# # axs[4].plot(time, Tmpg_in, label='MPG inlet', color='tab:blue')
# # # axs[4].plot(time, omega3, label='Motor3', color='tab:orange')
# # axs[4].set_ylabel("Temperature [°C]")
# # axs[4].set_xlabel("Time [s]")
# # # axs[4].set_title("Input Signal")
# # axs[4].legend()
# # axs[4].grid(True)

# # # 4. Input: Lpev
# # axs[4].plot(time, mw_dot, label='Water', color='tab:red')
# # axs[4].plot(time, mmpg_dot, label='MPG', color='tab:blue')
# # # axs[4].plot(time, omega3, label='Motor3', color='tab:orange')
# # axs[4].set_ylabel("Mass flow rate [Kg/s]")
# # axs[4].set_xlabel("Time [s]")
# # # axs[4].set_title("Input Signal")
# # axs[4].legend()
# # axs[4].grid(True)
# plt.tight_layout()

# fig2 = plt.figure()
# plt.plot(time, Pm1)
# plt.plot(time, Pm2)
# plt.plot(time, Pm3)
# plt.ylim([-20, 300])
# plt.show()
