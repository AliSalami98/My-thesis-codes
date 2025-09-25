import numpy as np
import pandas as pd

# Example dataset
num_rows = 20000
realistic_data = pd.DataFrame(index=range(num_rows))

# Original function for less important variables (no forced extremes)
def apply_independent_variations_with_ramp(column, min_value, max_value, percentage, time_range, ramp_size):
    start_idx = 0
    previous_value = np.random.uniform(min_value, max_value)
    
    while start_idx < num_rows:
        block_size = np.random.randint(time_range[0], time_range[1] + 1)
        end_idx = min(start_idx + block_size, num_rows)
        
        lower_bound = max(previous_value * (1 - percentage), min_value)
        upper_bound = min(previous_value * (1 + percentage), max_value)
        random_value = np.random.uniform(lower_bound, upper_bound)
        
        ramp_length = min(ramp_size, end_idx - start_idx)
        if ramp_length > 0:
            ramp = np.linspace(previous_value, random_value, ramp_length)
            realistic_data.loc[start_idx:start_idx + ramp_length - 1, column] = ramp
            start_idx += ramp_length
        
        if start_idx < end_idx:
            realistic_data.loc[start_idx:end_idx - 1, column] = random_value
            start_idx = end_idx
        
        previous_value = random_value

# Enhanced function for key variables (forces full range coverage)
def apply_variations_with_ramp_and_extremes(column, min_value, max_value, percentage, time_range, ramp_size, force_extreme_interval=5):
    start_idx = 0
    previous_value = np.random.uniform(min_value, max_value)
    block_count = 0
    
    while start_idx < num_rows:
        block_size = np.random.randint(time_range[0], time_range[1] + 1)
        end_idx = min(start_idx + block_size, num_rows)
        
        block_count += 1
        if block_count % force_extreme_interval == 0:
            random_value = min_value if np.random.rand() > 0.5 else max_value
        else:
            lower_bound = max(previous_value * (1 - percentage), min_value)
            upper_bound = min(previous_value * (1 + percentage), max_value)
            random_value = np.random.uniform(lower_bound, upper_bound)
        
        ramp_length = min(ramp_size, end_idx - start_idx)
        if ramp_length > 0:
            ramp = np.linspace(previous_value, random_value, ramp_length)
            realistic_data.loc[start_idx:start_idx + ramp_length - 1, column] = ramp
            start_idx += ramp_length
        
        if start_idx < end_idx:
            realistic_data.loc[start_idx:end_idx - 1, column] = random_value
            start_idx = end_idx
        
        previous_value = random_value

# Apply variations for each column
# Less important variables (no forced extremes)
apply_independent_variations_with_ramp('Tw_in [°C]', 24, 40, 0.15, (200, 400), 100)
apply_independent_variations_with_ramp('mw_dot [kg/s]', 10/60, 30/60, 0.2, (200, 400), 10)
# Key variables (force full range coverage)
apply_variations_with_ramp_and_extremes('Hpev', 15, 100, 0.2, (100, 150), 40, force_extreme_interval=5)
apply_variations_with_ramp_and_extremes('Theater2 [°C]', 500, 800, 0.2, (150, 300), 80, force_extreme_interval=5)

# Save the dataset
output_file_realistic = 'created_data_Theater.csv'
realistic_data.to_csv(output_file_realistic, index=False, encoding='utf-8', sep=';')
print(f"Dataset saved to {output_file_realistic}")

# Visualize to verify range coverage for key variables
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(realistic_data['Hpev'], label='Hpev', color='b')
plt.axhline(15, color='r', linestyle='--', label='Min/Max')
plt.axhline(100, color='r', linestyle='--')
plt.title('Hpev (15–100)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(realistic_data['Theater2 [°C]'], label='Theater2', color='g')
plt.axhline(500, color='r', linestyle='--', label='Min/Max')
plt.axhline(800, color='r', linestyle='--')
plt.title('Theater2 [°C] (500–800)')
plt.legend()

plt.tight_layout()
plt.show()