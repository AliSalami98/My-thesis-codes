# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load RNN and PID data
# rnn_data = pd.read_csv('outputs_rnn4.csv', sep=';')
# pid_data = pd.read_csv('outputs_PID.csv', sep=';')

# # Convert time to numeric
# rnn_data['current_time'] = pd.to_numeric(rnn_data['current_time'], errors='coerce')
# pid_data['current_time'] = pd.to_numeric(pid_data['current_time'], errors='coerce')

# # First add suffixes to each dataframe before merging
# rnn_data = rnn_data.add_suffix('_rnn')
# pid_data = pid_data.add_suffix('_pid')

# # Rename the time columns back (since we don't want suffixes on them)
# rnn_data = rnn_data.rename(columns={'current_time_rnn': 'current_time'})
# pid_data = pid_data.rename(columns={'current_time_pid': 'current_time'})

# # Now merge - no need for suffixes parameter
# data = pd.merge(rnn_data, pid_data, on='current_time')

# # Convert units
# for col in ['Two_rnn', 'Two_pid', 'Two_sp_rnn', 'Two_sp_pid', 
#             'Theater1_rnn', 'Theater1_pid', 'Theater_sp_pid']:
#     if col in data.columns:
#         data[col] = data[col] - 273.15  # Kelvin to Celsius

# for col in ['pc_rnn', 'pc_pid']:
#     if col in data.columns:
#         data[col] = data[col] / 1e5  # Pa to bar

# # Figure 1: System Performance Comparison
# fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# # Plot 1: Supply Temperature
# ax1.plot(data['current_time'], data['Two_rnn'], 'g', label='MPC-RNN')
# ax1.plot(data['current_time'], data['Two_pid'], 'b', label='PID')
# ax1.plot(data['current_time'], data['Two_sp_rnn'], ':k', linewidth=2, label='Setpoint')
# ax1.set_ylabel("$T_{w,out}$ [°C]", fontsize=12)
# ax1.legend(loc='upper right', fontsize=10)
# ax1.grid(True)

# # Plot 2: Theater Temperature
# ax2.plot(data['current_time'], data['Theater1_rnn'], 'g')
# ax2.plot(data['current_time'], data['Theater1_pid'], 'b')
# # ax2.plot(data['current_time'], data['Theater_sp'], '-.k', linewidth=2)
# ax2.set_ylabel("$T_{heater}$ [°C]", fontsize=12)
# ax2.grid(True)

# # Plot 3: High Pressure
# ax3.plot(data['current_time'], data['pc_rnn'], 'g')
# ax3.plot(data['current_time'], data['pc_pid'], 'b')
# ax3.set_ylabel("$p_c$ [bar]", fontsize=12)
# ax3.set_xlabel("Time [s]", fontsize=12)
# ax3.grid(True)

# fig1.tight_layout()

# # Figure 2: Control Signals
# fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# # Fan Speed
# ax1.plot(data['current_time'], data['omegab_opt_rnn'], 'g', label='RNN')
# ax1.plot(data['current_time'], data['omegab_opt_pid'], 'b', label='PID')
# ax1.set_ylabel("Fan Speed [rpm]", fontsize=12)
# ax1.legend()
# ax1.grid(True)

# # Valve Opening
# ax2.plot(data['current_time'], data['Hpev_opt_rnn'], 'g')
# ax2.plot(data['current_time'], data['Hpev_opt_pid'], 'b')
# ax2.set_ylabel("Valve Opening [%]", fontsize=12)
# ax2.set_xlabel("Time [s]", fontsize=12)
# ax2.grid(True)

# fig2.tight_layout()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and prepare data
rnn_data = pd.read_csv('outputs_rnn4.csv', sep=';')
pid_data = pd.read_csv('outputs_PID.csv', sep=';')

# Convert time to numeric and add suffixes
rnn_data['current_time'] = pd.to_numeric(rnn_data['current_time'], errors='coerce')
pid_data['current_time'] = pd.to_numeric(pid_data['current_time'], errors='coerce')

rnn_data = rnn_data.add_suffix('_rnn')
pid_data = pid_data.add_suffix('_pid')

# Fix time column names
rnn_data = rnn_data.rename(columns={'current_time_rnn': 'current_time'})
pid_data = pid_data.rename(columns={'current_time_pid': 'current_time'})

# Merge data
data = pd.merge(rnn_data, pid_data, on='current_time')

# Unit conversions
for col in ['Two_rnn', 'Two_pid', 'Theater1_rnn', 'Theater1_pid']:
    if col in data.columns:
        data[col] = data[col] - 273.15  # Kelvin to Celsius

# Create figure with 3 subplots
plt.rcParams.update({'font.size': 14})  # Set default font size
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Set tick parameters for all axes
for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)

# 1. Water Outlet Temperature (Tw_out)
ax1.plot(data['current_time'], data['Two_rnn'], 'g-', linewidth=2, label='MPC')
ax1.plot(data['current_time'], data['Two_pid'], 'b-', linewidth=2, label='PID')
if 'Two_sp_rnn' in data.columns:
    ax1.plot(data['current_time'], data['Two_sp_rnn']-273.15, 'k--', linewidth=2, label='Setpoint')
ax1.set_ylabel('$T_{w,out}$ [°C]', fontsize=18)
ax1.legend(loc='upper right', fontsize=14)

# 2. Heater Temperature (Theater)
ax2.plot(data['current_time'], data['Theater1_rnn'], 'g-', linewidth=2)
ax2.plot(data['current_time'], data['Theater1_pid'], 'b-', linewidth=2)
if 'Theater_sp_pid' in data.columns:
    ax2.plot(data['current_time'], data['Theater_sp_pid']-273.15, 'k--', linewidth=2)
ax2.set_ylabel('$T_\text{heater}$ [°C]', fontsize=18)

# 3. Burner Fan Speed
ax3.plot(data['current_time'], data['omegab_opt_rnn'], 'g-', linewidth=2, label='RNN')
ax3.plot(data['current_time'], data['omegab_opt_pid'], 'b-', linewidth=2, label='PID')
ax3.set_ylabel('$\omega_b$ [rpm]', fontsize=18)
ax3.set_xlabel('Time [s]', fontsize=18)

plt.tight_layout()
plt.show()


# Values
controllers = ['MPC', 'PID']
avg_cop = [
    data['COP_rnn'].mean(),
    data['COP_pid'].mean()
]
temp_error = [
    np.abs(data['Two_rnn'] + 273.15 - data['Two_sp_rnn']).mean(),
    np.abs(data['Two_pid'] + 273.15 - data['Two_sp_pid']).mean()
]

x = np.arange(len(controllers))  # [0, 1]
width = 0.35

# Plot
fig, ax1 = plt.subplots(figsize=(9, 6))

# First bar (COP)
bars1 = ax1.bar(x - width/2, avg_cop, width, label='Avg COP', color='tab:blue')
ax1.set_ylabel('Thermal COP [-]', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Second y-axis for temp error
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, temp_error, width, label='Temp Error [°C]', color='tab:red')
ax2.set_ylabel('Error [°C]', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# X-axis
plt.xticks(x, controllers)
ax1.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Compute improvements
cop_rnn = avg_cop[0]
cop_pid = avg_cop[1]
err_rnn = temp_error[0]
err_pid = temp_error[1]

cop_improvement_pct = (cop_rnn - cop_pid) / cop_pid * 100
error_reduction_pct = (err_pid - err_rnn) / err_pid * 100

print(f"\nCOP Improvement: {cop_improvement_pct:.2f}%")
print(f"Temperature Error Reduction: {error_reduction_pct:.2f}%")
