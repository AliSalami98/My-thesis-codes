import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load two CSV files
file1 = pd.read_csv('files/outputs_rnn.csv', sep=';')
file2 = pd.read_csv('files/outputs_lstm.csv', sep=';')

# Ensure 'current_time' is numeric
file1['current_time'] = pd.to_numeric(file1['current_time'], errors='coerce')
file2['current_time'] = pd.to_numeric(file2['current_time'], errors='coerce')

# Rename columns with explicit suffixes before merging
file1 = file1.add_suffix('_rnn').rename(columns={'current_time_rnn': 'current_time'})
file2 = file2.add_suffix('_lstm').rename(columns={'current_time_lstm': 'current_time'})

# Merge data on 'current_time' with outer join to keep all data
data = pd.merge(file1, file2, on='current_time', how='outer')

# Define colors for differentiation
colors = ['r', 'b']  # Red (20_10), Blue (20_10_2)

# Define bounds for control inputs
Hpev_bounds = (11, 100)      # High pressure valve opening [%]
omegab_bounds = (1950, 9500)  # Burner fan speed [rpm]
Lpev_bounds = (11, 100)      # Low pressure valve opening [%]

# Figure 1: Supply Temperature, Heater Temperature, Burner Fan Speed
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))

# Supply Temperature
ax1.plot(data['current_time'], data['Tw_out_rnn'], color='r', label='Tw_out rnn')
# ax1.plot(data['current_time'], data['Tw_out_rnn_rnn'], linestyle=':', color='r', label='Tw_out rnn rnn')
ax1.plot(data['current_time'], data['Tw_out_sp_rnn'], linestyle='--', color='r', label='Tw_out_sp rnn')
ax1.plot(data['current_time'], data['Tw_out_lstm'], color='b', label='Tw_out lstm')
# ax1.plot(data['current_time'], data['Tw_out_rnn_lstm'], linestyle=':', color='b', label='Tw_out lstm lstm')
ax1.plot(data['current_time'], data['Tw_out_sp_lstm'], linestyle='--', color='b', label='Tw_out_sp lstm')
ax1.set_ylabel("Water Supply Temperature [K]", fontsize=14)
ax1.grid(True)
ax1.legend()
ax1.tick_params(axis='both', which='major', labelsize=14)

# Heater Temperature
ax2.plot(data['current_time'], data['Theater1_rnn'], color='r', label='Theater1 rnn')
ax2.plot(data['current_time'], data['Theater_opt_rnn'], linestyle='--', color='r', label='Theater_opt rnn')
ax2.plot(data['current_time'], data['Theater1_lstm'], color='b', label='Theater1 lstm')
ax2.plot(data['current_time'], data['Theater_opt_lstm'], linestyle='--', color='b', label='Theater_opt lstm')
ax2.set_ylabel("Heater Temperature [K]", fontsize=14)
ax2.grid(True)
ax2.legend()
ax2.tick_params(axis='both', which='major', labelsize=14)

# Burner Fan Speed with bounds
ax3.plot(data['current_time'], data['omegab_opt_rnn'], color='r', label='omegab_opt rnn')
ax3.plot(data['current_time'], data['omegab_opt_lstm'], color='b', label='omegab_opt lstm')
ax3.axhline(y=omegab_bounds[0], color='k', linestyle='--', label='omegab_min')
ax3.axhline(y=omegab_bounds[1], color='k', linestyle='--', label='omegab_max')
ax3.set_ylabel("Burner Fan Speed [rpm]", fontsize=14)
ax3.grid(True)
ax3.legend()
ax3.tick_params(axis='both', which='major', labelsize=14)

fig1.tight_layout()

# Figure 2: High Pressure and High Pressure Valve Opening
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# High Pressure
ax1.plot(data['current_time'], data['pc_rnn'], color='r', label='pc rnn')
# ax1.plot(data['current_time'], data['pc_opt_rnn'], linestyle='--', color='r', label='pc opt')
ax1.plot(data['current_time'], data['pc_lstm'], color='b', label='pc lstm')
# ax1.plot(data['current_time'], data['pc_opt_lstm'], linestyle='--', color='b', label='pc lstm')
ax1.set_ylabel("High Pressure [Pa]", fontsize=14)
ax1.grid(True)
ax1.legend()
ax1.tick_params(axis='both', which='major', labelsize=14)

# High Pressure Valve Opening with bounds
ax2.plot(data['current_time'], data['Hpev_opt_rnn'], color='r', label='Hpev_opt rnn')
ax2.plot(data['current_time'], data['Hpev_opt_lstm'], color='b', label='Hpev_opt lstm')
ax2.axhline(y=Hpev_bounds[0], color='k', linestyle='--', label='Hpev_min')
ax2.axhline(y=Hpev_bounds[1], color='k', linestyle='--', label='Hpev_max')
ax2.set_ylabel("High Pressure Valve Opening [%]", fontsize=14)
ax2.grid(True)
ax2.legend()
ax2.tick_params(axis='both', which='major', labelsize=14)

fig2.tight_layout()

# Figure 3: Superheat and Low Pressure Valve Opening
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# Superheat
ax1.plot(data['current_time'], data['SH_rnn'], color='r', label='SH rnn')
ax1.plot(data['current_time'], data['SH_sp_rnn'], linestyle='--', color='r', label='SH_sp rnn')
ax1.plot(data['current_time'], data['SH_lstm'], color='b', label='SH lstm')
ax1.plot(data['current_time'], data['SH_sp_lstm'], linestyle='--', color='b', label='SH_sp lstm')
ax1.set_ylabel("Superheat [K]", fontsize=14)
ax1.grid(True)
ax1.legend()
ax1.tick_params(axis='both', which='major', labelsize=14)

# Low Pressure Valve Opening with bounds
ax2.plot(data['current_time'], data['Lpev_opt_rnn'], color='r', label='Lpev_opt rnn')
ax2.plot(data['current_time'], data['Lpev_opt_lstm'], color='b', label='Lpev_opt lstm')
ax2.axhline(y=Lpev_bounds[0], color='k', linestyle='--', label='Lpev_min')
ax2.axhline(y=Lpev_bounds[1], color='k', linestyle='--', label='Lpev_max')
ax2.set_ylabel("Low Pressure Valve Opening [%]", fontsize=14)
ax2.grid(True)
ax2.legend()
ax2.tick_params(axis='both', which='major', labelsize=14)

fig3.tight_layout()

# Figure 4: Power and COP
fig4, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 10))

# Gas Cooler Power
ax1.plot(data['current_time'], data['Pgc_rnn'], color='r', label='Pgc rnn')
ax1.plot(data['current_time'], data['Pgc_lstm'], color='b', label='Pgc lstm')
ax1.set_ylabel("Gas Cooler Power [W]", fontsize=14)
ax1.grid(True)
ax1.legend()
ax1.tick_params(axis='both', which='major', labelsize=14)

# Recuperator Power
ax2.plot(data['current_time'], data['Prec_rnn'], color='r', label='Prec rnn')
ax2.plot(data['current_time'], data['Prec_lstm'], color='b', label='Prec lstm')
ax2.set_ylabel("Heat Recovered Power [W]", fontsize=14)
ax2.grid(True)
ax2.legend()
ax2.tick_params(axis='both', which='major', labelsize=14)

# Combustion Power
ax3.plot(data['current_time'], data['Pcomb_rnn'], color='r', label='Pcomb rnn')
ax3.plot(data['current_time'], data['Pcomb_lstm'], color='b', label='Pcomb lstm')
ax3.set_ylabel("Combustion Power [W]", fontsize=14)
ax3.grid(True)
ax3.legend()
ax3.tick_params(axis='both', which='major', labelsize=14)

# Coefficient of Performance
ax4.plot(data['current_time'], data['COP_rnn'], color='r', label='COP rnn')
ax4.plot(data['current_time'], data['COP_lstm'], color='b', label='COP lstm')
ax4.set_ylabel("COP [-]", fontsize=14)
ax4.grid(True)
ax4.legend()
ax4.tick_params(axis='both', which='major', labelsize=14)

fig4.tight_layout()

plt.show()

# ------------------------
# COP and Temperature Error Metrics
# ------------------------

# Compute average COPs
avg_COP_rnn = data['COP_rnn'].mean()
avg_COP_lstm = data['COP_lstm'].mean()

# Compute average temperature error (absolute) for Tw_out
avg_Tw_out_error_rnn = np.mean(np.abs(data['Tw_out_rnn'] - data['Tw_out_sp_rnn']))
avg_Tw_out_error_lstm = np.mean(np.abs(data['Tw_out_lstm'] - data['Tw_out_sp_lstm']))

# Optionally, compute average heater temperature error
avg_Theater_error_rnn = np.mean(np.abs(data['Theater1_rnn'] - data['Theater_opt_rnn']))
avg_Theater_error_lstm = np.mean(np.abs(data['Theater1_lstm'] - data['Theater_opt_lstm']))

# Print performance metrics
print("\n=== Performance Comparison ===")
print(f"Average COP (RNN):  {avg_COP_rnn:.3f}")
print(f"Average COP (LSTM): {avg_COP_lstm:.3f}")
print(f"Average Tw_out Error (RNN) [K]:  {avg_Tw_out_error_rnn:.3f}")
print(f"Average Tw_out Error (LSTM) [K]: {avg_Tw_out_error_lstm:.3f}")
print(f"Average Theater Error (RNN) [K]:  {avg_Theater_error_rnn:.3f}")
print(f"Average Theater Error (LSTM) [K]: {avg_Theater_error_lstm:.3f}")

# Optional: Save as CSV
metrics_df = pd.DataFrame({
    'Controller': ['RNN', 'LSTM'],
    'Average COP': [avg_COP_rnn, avg_COP_lstm],
    'Tw_out Error [K]': [avg_Tw_out_error_rnn, avg_Tw_out_error_lstm],
    'Theater Error [K]': [avg_Theater_error_rnn, avg_Theater_error_lstm]
})
# metrics_df.to_csv("metrics_comparison_lstm_lstm.csv", index=False)
