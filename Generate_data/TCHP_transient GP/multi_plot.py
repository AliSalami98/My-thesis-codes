import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File names
file_names = [f'files/file{k}.csv' for k in range(5)]  # Generates 'file0.csv' to 'file4.csv'

# Initialize arrays for each file
data = {f'current_time{k}': [] for k in range(5)}
columns_to_extract = [
    'current_time', 'Tw_out', 'Tw_out_sp', 'pc', 'SH', 'SH_sp',
    'Theater1', 'Theater_opt', 'omegab_opt', 'Hpev_opt', 'Lpev_opt',
    'COP', 'Pgc', 'Prec', 'Pcomb'
]

# Initialize empty arrays for all columns across files
extracted_data = {f'{col}{k}': [] for col in columns_to_extract for k in range(5)}

# Loop through files and populate arrays
for k, file_name in enumerate(file_names):
    file_data = pd.read_csv(file_name)
    for col in columns_to_extract:
        key = f'{col}{k}'
        if col in file_data.columns:
            extracted_data[key] = file_data[col].tolist()
        else:
            print(f"Warning: Column '{col}' not found in {file_name}.")

# Convert lists to numpy arrays if needed
for key in extracted_data:
    extracted_data[key] = np.array(extracted_data[key])


# Example access: extracted_data['Tw_out0'], extracted_data['time0'], etc.

# Plotting
import matplotlib.cm as cm
alpha_values = [0, 0.25, 0.5, 0.75, 1]
colors = cm.viridis([0, 0.25, 0.5, 0.75, 1])  # Using a colormap for distinct colors

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 10))

# Supply Temperature Plot
for k, (alpha, color) in enumerate(zip(alpha_values, colors)):
    ax1.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'Tw_out{k}'], 
        label=f'alpha = {alpha}', 
        color=color
    )
    ax1.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'Tw_out_sp{k}'], 
        linestyle='--', 
        color=color
    )
ax1.set_ylabel("Temperature [°C]", fontsize=14)
ax1.legend(loc='best', fontsize=12)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_title("Supply Temperature")

# Theater Temperature Plot
for k, (alpha, color) in enumerate(zip(alpha_values, colors)):
    ax2.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'Theater1{k}'], 
        label=f'alpha = {alpha}', 
        color=color
    )
    ax2.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'Theater_opt{k}'], 
        linestyle='--', 
        color=color
    )
ax2.set_ylabel("Temperature [°C]", fontsize=14)
ax2.legend(loc='best', fontsize=12)
ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.set_title("Theater Temperature")

# Burner Fan Speed Plot
for k, (alpha, color) in enumerate(zip(alpha_values, colors)):
    ax3.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'omegab_opt{k}'], 
        label=f'alpha = {alpha}', 
        color=color
    )
ax3.set_ylabel("Burner fan speed [rpm]", fontsize=14)
ax3.legend(loc='best', fontsize=12)
ax3.grid(True)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.set_title("Burner Fan Speed")

fig1.tight_layout()

fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

# High Pressure Plot
for k, (alpha, color) in enumerate(zip(alpha_values, colors)):
    ax1.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'pc{k}'], 
        label=f'alpha = {alpha}', 
        color=color
    )
    # ax1.plot(
    #     extracted_data[f'current_time{k}'], 
    #     extracted_data[f'pc_opt{k}'], 
    #     linestyle='--', 
    #     color=color
    # )
ax1.set_ylim([40e5, 120e5])
ax1.set_ylabel("Pressure [Pa]", fontsize=14)
ax1.legend(loc='best', fontsize=12)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_title("High Pressure")

# High Pressure Valve Opening
for k, (alpha, color) in enumerate(zip(alpha_values, colors)):
    ax2.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'Hpev_opt{k}'], 
        label=f'alpha = {alpha}', 
        color=color
    )
ax2.set_ylabel("High pressure valve opening [%]", fontsize=14)
ax2.legend(loc='best', fontsize=12)
ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=14)
fig2.tight_layout()

# Superheat Plot
fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
for k, (alpha, color) in enumerate(zip(alpha_values, colors)):
    ax1.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'SH{k}'], 
        label=f'alpha = {alpha}', 
        color=color
    )
    ax1.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'SH_sp{k}'], 
        linestyle='--', 
        color=color
    )
ax1.set_ylabel("Temperature [°C]", fontsize=14)
ax1.legend(loc='best', fontsize=12)
ax1.grid(True)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.set_title("Superheat Comparison")

# Low Pressure Valve Opening
for k, (alpha, color) in enumerate(zip(alpha_values, colors)):
    ax2.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'Lpev_opt{k}'], 
        label=f'alpha = {alpha}', 
        color=color
    )
ax2.set_ylabel("Low pressure valve opening [%]", fontsize=14)
ax2.legend(loc='best', fontsize=12)
ax2.grid(True)
ax2.tick_params(axis='both', which='major', labelsize=14)
fig3.tight_layout()

# COP Plot
plt.figure()
for k, (alpha, color) in enumerate(zip(alpha_values, colors)):
    plt.plot(
        extracted_data[f'current_time{k}'], 
        extracted_data[f'COP{k}'], 
        label=f'alpha = {alpha}', 
        color=color
    )
plt.ylabel("COP [-]", fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title("Coefficient of Performance")

plt.tight_layout()
plt.show()

# Calculate Average COP and Supply Temperature Error
average_COPs = []
temperature_errors = []

for k in range(len(alpha_values)):  # Loop over the files/strategies
    # Average COP Calculation
    if f'COP{k}' in extracted_data and len(extracted_data[f'COP{k}']) > 0:
        avg_cop = np.nanmean(extracted_data[f'COP{k}'])  # Use nanmean to handle NaNs gracefully
        average_COPs.append(avg_cop)
        print(f"Average COP for strategy alpha = {alpha_values[k]}: {avg_cop:.2f}")
    else:
        print(f"No COP data available for strategy alpha = {alpha_values[k]}.")
        average_COPs.append(None)

    # Supply Temperature Error Calculation
    if f'Tw_out{k}' in extracted_data and f'Tw_out_sp{k}' in extracted_data:
        abs_error = np.abs(
            np.array(extracted_data[f'Tw_out{k}']) - np.array(extracted_data[f'Tw_out_sp{k}'])
        )
        mae = np.nanmean(abs_error)  # Mean Absolute Error
        temperature_errors.append(mae)
        print(f"Mean Absolute Error for supply temperature in strategy alpha = {alpha_values[k]}: {mae:.2f} K")
    else:
        print(f"No temperature data available for strategy alpha = {alpha_values[k]}.")
        temperature_errors.append(None)

# Visualization: Combine Average COP and Temperature Errors
colors = cm.viridis(np.linspace(0, 1, len(alpha_values)))  # Generate consistent colors for strategies

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot Average COP as bars
bars = ax.bar(alpha_values, [cop if cop is not None else 0 for cop in average_COPs],
              color=colors, alpha=0.8, width=0.1, label='Average COP')

# Plot Temperature Errors as a line
ax.plot(alpha_values, [err if err is not None else 0 for err in temperature_errors],
        color='black', marker='o', linestyle='-', linewidth=2, label='Temperature Error')

# Customize axis
ax.set_xlabel("Alpha Values", fontsize=14)
ax.set_ylabel("Values", fontsize=14)
ax.set_title("Comparison of Average COP and Temperature Error", fontsize=16)
ax.set_xticks(alpha_values)
ax.set_xticklabels([f"{alpha:.1f}" for alpha in alpha_values])

# Add legends
ax.legend(loc='upper left', fontsize=12)

# Add grid
ax.grid(True, linestyle='--', alpha=0.5)

# Show plot
plt.tight_layout()
plt.show()
