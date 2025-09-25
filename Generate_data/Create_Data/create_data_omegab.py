import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict_next_Theater1(X1_current, U1_current, D1_current, A1_coeffs, B1_coeffs, D1_coeffs, poly_U1):
    # Compute A_k (parameter-independent in this case)
    A1_k = A1_coeffs

    # Compute parameter-dependent B matrix
    input_vector = poly_U1.transform(U1_current.reshape(1, -1))  # Transform current input
    B1_k = B1_coeffs @ input_vector.T  # Input-dependent B


    D1_k = D1_coeffs
    # Compute next state (D remains independent)
    X1_next = (A1_k @ X1_current) + (B1_k.flatten()) + (D1_k @ D1_current)
    return X1_next

def predict_next_Theater2(X2_current, U2_current, D2_current, A2_coeffs, B2_coeffs, D2_coeffs, poly_U2):
    # Compute A_k (parameter-independent in this case)
    A2_k = A2_coeffs

    # Compute parameter-dependent B matrix
    input_vector = poly_U2.transform(U2_current.reshape(1, -1))  # Transform current input
    B2_k = B2_coeffs @ input_vector.T  # Input-dependent B


    D2_k = D2_coeffs
    # Compute next state (D remains independent)
    X2_next = (A2_k @ X2_current) + (B2_k.flatten()) + (D2_k @ D2_current)
    return X2_next
# Example dataset
num_rows = 40000
realistic_data = pd.DataFrame(index=range(num_rows))

# Define function to apply independent variations with ramp transitions
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

def apply_prbs_like_variations_with_full_range(
    column,
    min_value,
    max_value,
    percentage,
    time_range,
    ramp_length,
    force_extreme_interval=5,
    min_jump=0.3
):
    start_idx = 0
    previous_value = np.random.uniform(min_value, max_value)
    block_count = 0

    while start_idx < num_rows:
        block_size = np.random.randint(time_range[0], time_range[1] + 1)
        end_idx = min(start_idx + block_size, num_rows)
        block_count += 1

        # Force extreme value occasionally
        if block_count % force_extreme_interval == 0:
            random_value = min_value if np.random.rand() > 0.5 else max_value
        else:
            # Pick direction
            direction = 1 if np.random.rand() > 0.5 else -1

            # Define jump bounds
            lower_rel = 1 + direction * min_jump
            upper_rel = 1 + direction * percentage

            # Apply bounded jump and clip within min/max range
            raw_value = previous_value * np.random.uniform(lower_rel, upper_rel)
            random_value = np.clip(raw_value, min_value, max_value)

        # Ramp transition
        if ramp_length > 0:
            ramp_steps = min(ramp_length, end_idx - start_idx)
            ramp = np.linspace(previous_value, random_value, ramp_steps)
            realistic_data.loc[start_idx:start_idx + ramp_steps - 1, column] = ramp
            start_idx += ramp_steps

        # Constant hold after ramp
        if start_idx < end_idx:
            realistic_data.loc[start_idx:end_idx - 1, column] = random_value
            start_idx = end_idx

        previous_value = random_value



# Apply variations with ramp transitions for each column
apply_independent_variations_with_ramp('Tw_in [C]', 20, 40, 0.3, (300, 400), 20)
apply_independent_variations_with_ramp('mw_dot [kg/s]', 10/60, 30/60, 0.3, (300, 400), 10)
# apply_independent_variations_with_ramp('omega2 [rpm]', 100, 220, 0.20, (300, 400), 10)
apply_prbs_like_variations_with_full_range(
    'Hpev', 15, 100, 0.4, (50, 150), 10, force_extreme_interval=5, min_jump=0.3
)

print(apply_prbs_like_variations_with_full_range(
    'omegab [rpm]', 1950, 9500, 0.4, (100, 200), 10, force_extreme_interval=5, min_jump=0.3
))

import joblib
omega1 = 200
omega2 = 150
omega3 = 100
mmpg_dot = 16.5/60
Tmpg_in = 3
Tw_in = 30
mw_dot = 16.5/60
Lpev = 50
Hpev= 50
omegab = 4000
Theater1 = 800.15
Theater2 = 773.15


A1_coeffs = joblib.load("Theater/A1_coeffs.pkl")
B1_coeffs = joblib.load("Theater/B1_coeffs.pkl")
D1_coeffs = joblib.load("Theater/D1_coeffs.pkl")
poly_U1 = joblib.load("Theater/poly_U1.pkl")

A2_coeffs = joblib.load("Theater/A2_coeffs.pkl")
B2_coeffs = joblib.load("Theater/B2_coeffs.pkl")
D2_coeffs = joblib.load("Theater/D2_coeffs.pkl")
poly_U2 = joblib.load("Theater/poly_U2.pkl")

Theater1_current = np.column_stack([Theater1])
U1_current = np.column_stack([omegab])
D1_current = np.column_stack([omega1])

Theater2_current = np.column_stack([Theater2])
U2_current = np.column_stack([omegab])
D2_current = np.column_stack([omega2])

d_Theater1 = []
d_Theater2 = []
d_omegab = []
d_Hpev = []
d_Tw_in = []
d_mw_dot = []

d_t = []

dt = 1
import random
t_change_omegab = 250
t_change_Hpev = 300
t_change_Tw_in = 350
t_change_mw_dot = 400

for i in range(num_rows):
    if i == t_change_omegab:
        # if omegab == 1950:
        #     omegab = 9500
        # else:
        #     omegab = 1950 
        omegab = random.randint(1950, 9500)
        t_change_omegab = random.randint(200 + t_change_omegab, 500 + t_change_omegab)
    if i == t_change_Hpev: 
        Hpev = random.randint(11, 100)
        # if Hpev == 100:
        #     Hpev = 15
        # else:
        #     Hpev = 100
        t_change_Hpev = random.randint(200 + t_change_Hpev, 500 + t_change_Hpev)
    if i == t_change_Tw_in: 
        variation = Tw_in * np.random.uniform(-0.05, 0.05)
        Tw_in = np.clip(Tw_in + variation, 22, 45)
        t_change_Tw_in = random.randint(400 + t_change_Tw_in, 800 + t_change_Tw_in)

    if i == t_change_mw_dot: 
        variation = mw_dot * np.random.uniform(-0.05, 0.05)
        mw_dot = np.clip(mw_dot + variation, 10/60, 30/60)
        t_change_mw_dot = random.randint(400 + t_change_mw_dot, 800 + t_change_mw_dot)

    # Predict next Theater1 and Theater2
    Theater1_current = predict_next_Theater2(Theater1_current, U1_current, D1_current, A1_coeffs, B1_coeffs, D1_coeffs, poly_U1)
    Theater2_current = predict_next_Theater2(Theater2_current, U2_current, D2_current, A2_coeffs, B2_coeffs, D2_coeffs, poly_U2)

    Theater1 = Theater1_current[0][0]
    Theater2 = Theater2_current[0][0]

    if Theater1 > 1050:
        omegab = 1950
    if Theater2 < 800:
        omegab = 9500
        
    U1_current = np.column_stack([omegab])
    D1_current = np.column_stack([omega1])

    U2_current = np.column_stack([omegab])
    D2_current = np.column_stack([omega2])

    d_Theater1.append(Theater1)
    d_Theater2.append(Theater2)
    d_omegab.append(omegab)
    d_Hpev.append(Hpev)
    d_Tw_in.append(30) #Tw_in)
    d_mw_dot.append(0.23) #mw_dot)
    d_t.append(i)

import pandas as pd

# Create a DataFrame with the lists
df = pd.DataFrame({
    # 't': d_t,
    'Tw_in [°C]': d_Tw_in,
    'mw_dot [kg/s]': d_mw_dot,
    'Hpev': d_Hpev,
    'omegab [rpm]': d_omegab,
    # 'Theater1': d_Theater1,
    # 'Theater2': d_Theater2,
})

# Save to CSV
df.to_csv('created_data_omegab3.csv', index=False)
# fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

# axs[0].plot(d_t, d_Theater1)
# axs[0].set_ylabel("Theater1")

# axs[1].plot(d_t, d_Theater2)
# axs[1].set_ylabel("Theater2")

# axs[2].plot(d_t, d_omegab)
# axs[2].set_ylabel("omegab")

# axs[3].plot(d_t, d_Hpev)
# axs[3].set_ylabel("Hpev")

# axs[4].plot(d_t, d_Tw_in)
# axs[4].set_ylabel("Tw_in")

# axs[5].plot(d_t, d_mw_dot)
# axs[5].set_ylabel("mw_dot")
# axs[5].set_xlabel("Time step")

# plt.tight_layout()
# plt.show()
# # apply_independent_variations_with_ramp('Hpev', 15, 100, 0.4, (100, 250), 10)
# # apply_independent_variations_with_ramp('omegab [rpm]', 3000, 6000, 0.4, (300, 600), 10)

# # Save the dataset
# output_file_realistic = 'created_datd_omegab.csv'
# realistic_data.to_csv(output_file_realistic, index=False, encoding='utf-8', sep=';')

# print(f"Dataset saved to {output_file_realistic}")