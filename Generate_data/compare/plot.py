import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Load all three CSV files
file1 = pd.read_csv('outputs_lstm2.csv', sep=';')
file2 = pd.read_csv('outputs_rnn4.csv', sep=';')
file3 = pd.read_csv('outputs_PID.csv', sep=';')

# Ensure 'current_time' column is numeric
# file1['current_time'] = pd.to_numeric(file1['current_time'], errors='coerce')
file2['current_time'] = pd.to_numeric(file2['current_time'], errors='coerce')
file3['current_time'] = pd.to_numeric(file3['current_time'], errors='coerce')

# Merge all three datasets on 'current_time'
data = pd.merge(pd.merge(file1, file2, on='current_time', suffixes=('_lstm', '_rnn')), 
                file3, on='current_time', suffixes=('', '_pid'))

# Rename columns to ensure consistency
data = data.rename(columns={'Tw_out': 'Tw_out_pid', 'Theater1': 'Theater1_pid', 'pc': 'pc_pid',
                            'omegab_opt': 'omegab_opt_pid', 'Hpev_opt': 'Hpev_opt_pid',
                            'COP': 'COP_pid', 'Tw_out_sp': 'Tw_out_sp_pid'})

# Convert temperatures from Kelvin to Celsius
data['Tw_out_lstm'] = data['Tw_out_lstm'] - 273.15
data['Tw_out_rnn'] = data['Tw_out_rnn'] - 273.15
data['Tw_out_pid'] = data['Tw_out_pid'] - 273.15
data['Tw_out_sp_lstm'] = data['Tw_out_sp_lstm'] - 273.15
data['Tw_out_sp_rnn'] = data['Tw_out_sp_rnn'] - 273.15
data['Tw_out_sp_pid'] = data['Tw_out_sp_pid'] - 273.15
data['Theater1_lstm'] = data['Theater1_lstm'] - 273.15
data['Theater1_rnn'] = data['Theater1_rnn'] - 273.15
data['Theater1_pid'] = data['Theater1_pid'] - 273.15
data['Theater_opt'] = data['Theater_opt'] - 273.15


# Figure 1: Supply Temperature, Theater Temperature, Pressure
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
ax1.plot(data['current_time'], data['Tw_out_lstm'], color='r', label='MPC-LSTM', zorder=2)
ax1.plot(data['current_time'], data['Tw_out_rnn'], color='g', label='MPC-RNN', zorder=2)
ax1.plot(data['current_time'], data['Tw_out_pid'], color='b', label='PID', zorder=2)
ax1.plot(data['current_time'], data['Tw_out_sp_lstm'], linestyle=':', color='k', linewidth=2, label=r'$T^{\text{sp}}_{\text{w,sup}}$', zorder=2)
ax1.set_ylabel(r"$T_{\text{w,sup}}$ [°C]", fontsize=16)
ax1.grid(alpha=0.2, zorder=0)
ax1.legend(fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=14)

ax2.plot(data['current_time'], data['Theater1_lstm'], color='r', zorder=2)
ax2.plot(data['current_time'], data['Theater1_rnn'], color='g', zorder=2)
ax2.plot(data['current_time'], data['Theater1_pid'], color='b', zorder=2)
ax2.plot(data['current_time'], data['Theater_opt'], linestyle='-.', color='k', linewidth=2, label=r'$T^{\text{sp}}_{\text{h1}}$', zorder=2)
ax2.set_ylabel(r"$T_{\text{h1}}$ [°C]", fontsize=16)
ax2.grid(alpha=0.2, zorder=0)
ax2.legend(fontsize=12)
ax2.tick_params(axis='both', which='major', labelsize=14)

ax3.plot(data['current_time'], data['pc_lstm'], color='r', zorder=2)
ax3.plot(data['current_time'], data['pc_rnn'], color='g', zorder=2)
ax3.plot(data['current_time'], data['pc_pid'], color='b', zorder=2)
ax3.plot(data['current_time'], data['pc_opt'], linestyle='--', color='k', linewidth=2, label=r'$p^{\text{sp}}_{\text{gc}}$')
ax3.set_ylabel(r"$p_{\text{gc}}$ [bar]", fontsize=16)
ax3.set_xlabel("Time [s]", fontsize=16)
ax3.grid(alpha=0.2, zorder=0)
ax3.legend(fontsize=12)
ax3.tick_params(axis='both', which='major', labelsize=14)

fig1.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\control\y_opt.eps", format='eps', bbox_inches='tight')

fan_min, fan_max = 2000, 9500          # burner fan speed [rpm]
valve_min, valve_max = 11, 100         # high-pressure valve opening [%]

# Figure 2: Control Variables (Fan Speed and Valve Opening)
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(data['current_time'], data['omegab_opt_lstm'], color='r', label='MPC-LSTM', zorder=2)
ax1.plot(data['current_time'], data['omegab_opt_rnn'], color='g', label='MPC-RNN', zorder=2)
ax1.plot(data['current_time'], data['omegab_opt_pid'], color='b', label='PID', zorder=2)
ax1.axhline(y=fan_min, color='k', linestyle='--', linewidth=1, label='Control bounds')
ax1.axhline(y=fan_max, color='k', linestyle='--', linewidth=1)
ax1.set_ylabel(r"$\omega_{\text{bf}}$ [rpm]", fontsize=16)
ax1.grid(alpha=0.2, zorder=0)

ax1.legend(fontsize=12)
ax1.tick_params(axis='both', which='major', labelsize=14)

ax2.plot(data['current_time'], data['Hpev_opt_lstm'], color='r', label='MPC-LSTM', zorder=2)
ax2.plot(data['current_time'], data['Hpev_opt_rnn'], color='g', label='MPC-RNN', zorder=2)
ax2.plot(data['current_time'], data['Hpev_opt_pid'], color='b', label='PID', zorder=2)
ax2.axhline(y=valve_min, color='k', linestyle='--', linewidth=1, label='Control bounds')
ax2.axhline(y=valve_max, color='k', linestyle='--', linewidth=1)
ax2.set_ylabel(r"$ \varphi_{\text{hpv}}$ [%]", fontsize=16)
ax2.set_xlabel("Time [s]", fontsize=16)
ax2.grid(alpha=0.2, zorder=0)
ax2.tick_params(axis='both', which='major', labelsize=14)

fig2.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\control\u_opt.eps", format='eps', bbox_inches='tight')

# Compute average COPs
avg_COP_lstm = data['COP_lstm'].mean()
avg_COP_rnn = data['COP_rnn'].mean()
avg_COP_pid = data['COP_pid'].mean()

# Calculate average temperature error of Tw_out with respect to Tw_out_sp
avg_Tw_out_error_lstm = np.mean(np.abs(data['Tw_out_lstm'] - data['Tw_out_sp_lstm']))
avg_Tw_out_error_rnn = np.mean(np.abs(data['Tw_out_rnn'] - data['Tw_out_sp_rnn']))
avg_Tw_out_error_pid = np.mean(np.abs(data['Tw_out_pid'] - data['Tw_out_sp_pid']))

# Define rise time and settling time functions
def calculate_rise_time(time, actual, setpoint, lower_limit=0.1, upper_limit=0.9):
    final_value = setpoint.iloc[-1]
    lower_threshold = final_value * lower_limit
    upper_threshold = final_value * upper_limit
    try:
        t_start = time[np.where(actual >= lower_threshold)[0][0]]
        t_end = time[np.where(actual >= upper_threshold)[0][0]]
        return t_end - t_start
    except IndexError:
        return np.nan

def calculate_settling_time(time, actual, setpoint, tolerance=0.02):
    final_value = setpoint.iloc[-1]
    lower_bound = final_value * (1 - tolerance)
    upper_bound = final_value * (1 + tolerance)
    try:
        settled_time = time[np.where((actual <= upper_bound) & (actual >= lower_bound))[0][-1]]
        return settled_time - time[0]
    except IndexError:
        return np.nan

# Extract data for metrics
time = data['current_time']
Tw_out_lstm, Tw_out_rnn, Tw_out_pid = data['Tw_out_lstm'], data['Tw_out_rnn'], data['Tw_out_pid']
Tw_out_sp_lstm, Tw_out_sp_rnn, Tw_out_sp_pid = data['Tw_out_sp_lstm'], data['Tw_out_sp_rnn'], data['Tw_out_sp_pid']

# Calculate metrics
rise_time_lstm = calculate_rise_time(time, Tw_out_lstm, Tw_out_sp_lstm)
rise_time_rnn = calculate_rise_time(time, Tw_out_rnn, Tw_out_sp_rnn)
rise_time_pid = calculate_rise_time(time, Tw_out_pid, Tw_out_sp_pid)
settling_time_lstm = calculate_settling_time(time, Tw_out_lstm, Tw_out_sp_lstm)
settling_time_rnn = calculate_settling_time(time, Tw_out_rnn, Tw_out_sp_rnn)
settling_time_pid = calculate_settling_time(time, Tw_out_pid, Tw_out_sp_pid)

# Prepare data for bar plot
labels = ['MPC-LSTM', 'MPC-RNN', 'PID']
COP_values = [avg_COP_lstm, avg_COP_rnn, avg_COP_pid]
Tw_out_error_values = [avg_Tw_out_error_lstm, avg_Tw_out_error_rnn, avg_Tw_out_error_pid]
rise_time_values = [rise_time_lstm, rise_time_rnn, rise_time_pid]
settling_time_values = [settling_time_lstm, settling_time_rnn, settling_time_pid]

# Combine metrics into a DataFrame
metrics_df = pd.DataFrame({
    'Labels': labels,
    'Average COP': COP_values,
    'Average Error (°C)': Tw_out_error_values,
    'Rise Time (s)': rise_time_values,
    'Settling Time (s)': settling_time_values
})

# Save and display metrics
# metrics_df.to_csv('performance_metrics_celsius.csv', index=False)
print(metrics_df)

bar_width = 0.4
x = np.arange(len(labels))

bars1 = ax1.bar(x - bar_width / 2, COP_values, bar_width, label='Average Thermal COP [-]', color='green', zorder=2)
bars2 = ax2.bar(x + bar_width / 2, Tw_out_error_values, bar_width, label='Average Temperature Error [°C]', color='orange', zorder=2)

# # Figure 3: Bar Chart Comparison with Dual Axes
# fig3, ax1 = plt.subplots(figsize=(12, 6))
# ax2 = ax1.twinx()
# ax1.set_ylabel('Thermal COP', fontsize=16)
# ax2.set_ylabel('$|T^{sp}_{w,sup} - T_{w,sup}|$', fontsize=16)
# ax1.set_xticks(x)
# ax1.set_xticklabels(labels, fontsize=16)
# ax1.tick_params(axis='both', which='major', labelsize=14)
# ax2.tick_params(axis='both', which='major', labelsize=14)

# ax1.legend(loc='upper left', fontsize=12)
# ax2.legend(loc='upper right', fontsize=12)
# ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# fig3.tight_layout()
# plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\control\Average.eps", format='eps', bbox_inches='tight')

# --- Figure 1: COP ---
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(x, COP_values, bar_width, color='green', zorder=2)
ax1.set_ylabel('Thermal COP', fontsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12)
# ax1.set_title('Average COP Comparison', fontsize=16)
ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.legend(fontsize=12)

fig1.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\control\Average_COP.eps",
            format='eps', bbox_inches='tight')

# --- Figure 2: Temperature Error ---
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.bar(x, Tw_out_error_values, bar_width, color='orange', zorder=2)
ax2.set_ylabel(r'$|T^{\text{sp}}_{\text{w,sup}} - T_{\text{w,sup}}|$ [K]', fontsize=16)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=12)
# ax2.set_title('Average Temperature Error', fontsize=16)
ax2.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.legend(fontsize=12)

fig2.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\control\Average_TempError.eps",
            format='eps', bbox_inches='tight')

# ----------- Compute Improvements over PID -----------
cop_improvement_lstm = 100 * (avg_COP_lstm - avg_COP_pid) / avg_COP_pid
cop_improvement_rnn  = 100 * (avg_COP_rnn  - avg_COP_pid) / avg_COP_pid

temp_error_reduction_lstm = 100 * (avg_Tw_out_error_pid - avg_Tw_out_error_lstm) / avg_Tw_out_error_pid
temp_error_reduction_rnn  = 100 * (avg_Tw_out_error_pid - avg_Tw_out_error_rnn) / avg_Tw_out_error_pid

print("\n--- COP Improvement over PID ---")
print(f"LSTM: {cop_improvement_lstm:.2f}%")
print(f"RNN:  {cop_improvement_rnn:.2f}%")

print("\n--- Temperature Error Reduction over PID ---")
print(f"LSTM: {temp_error_reduction_lstm:.2f}%")
print(f"RNN:  {temp_error_reduction_rnn:.2f}%")

plt.show()
