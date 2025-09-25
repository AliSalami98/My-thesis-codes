import csv
import numpy as np
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
COP = []

i = 0

csv_files = [
    r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\burner fan PRBS\sim\pc.csv',
    r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\bf step.csv',
    r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\M2 step.csv'
]

# Read each CSV file and fill the lists
for file_idx, file_path in enumerate(csv_files):
    with open(file_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for i, row in enumerate(csv_reader, start=1):
            i = i + 1
            # if i > 100:
            #     break
            try:
                # Check for empty strings and convert to float
                # if row['Tw_in [°C]']:
                #     Tw_in.append(float(row['Tw_in [°C]']) + 273.15)
                # if row['omega1 [rpm]']:
                #     omega1.append(float(row['omega1 [rpm]']))
                if row['omega2 [rpm]']:
                    omega2.append(float(row['omega2 [rpm]']))
                # if row['omega3 [rpm]']:
                #     omega3.append(float(row['omega3 [rpm]']))
                if row['omegab [rpm]']:
                    omegab.append(float(row['omegab [rpm]']))
                # if row['Theater1 [°C]']:
                #     Theater1.append(float(row['Theater1 [°C]']) + 273.15)
                if row['Theater2 [°C]']:
                    Theater2.append(float(row['Theater2 [°C]']) + 273.15)                

            except ValueError as e:
                print(f"Skipping row {i} in file {file_idx + 1} due to error: {e}")



# def predict_next_Theater2(X2_current, U2_current, A2_coeffs, B2_coeffs, poly_U2):
#     # Compute A_k (parameter-independent in this case)
#     A2_k = A2_coeffs

#     # Compute parameter-dependent B matrix
#     input_vector = poly_U2.transform(U2_current.reshape(1, -1))  # Transform current input
#     B2_k = B2_coeffs @ input_vector.T  # Input-dependent B


#     # Compute next state
#     X2_next = (A2_k @ X2_current) + (B2_k.flatten())
#     return X2_next

# X_current = np.column_stack([450])
# # U_current = np.column_stack([6000, 100])

# # print(X_current)
# # print(U_current)
# import joblib

# A2_coeffs = joblib.load("A2_coeffs.pkl")
# B2_coeffs = joblib.load("B2_coeffs.pkl")
# poly_U2 = joblib.load("poly_U2.pkl")

# # # Predict the next state
# a_X_current = []

# for i in range(len(omega2)):     
#     U_current = np.column_stack([omegab[i]])
#     X_current = predict_next_Theater2(X_current, U_current, A2_coeffs, B2_coeffs, poly_U2)
#     a_X_current.append(Theater2[i] - 

# import csv
# import numpy as np

# # Initialize lists to hold the training and testing data for each variable
# train_data = {"omegab": [], "omega2": [], "Theater2": []}
# test_data = {"omegab": [], "omega2": [], "Theater2": []}

# csv_files = [
#     r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\burner fan PRBS\sim\pc.csv',
#     r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\bf step.csv',
#     # r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\M2 step.csv'
# ]

# # Process each CSV file
# for file_idx, file_path in enumerate(csv_files):
#     # Temporary lists to store data for each column
#     omegab, omega2, Theater2 = [], [], []
    
#     with open(file_path) as csv_file:
#         csv_reader = csv.DictReader(csv_file, delimiter=';')
#         for i, row in enumerate(csv_reader):
#             try:

#                 if row['omega2 [rpm]']:
#                     omega2.append(float(row['omega2 [rpm]']))
#                 if row['omegab [rpm]']:
#                     omegab.append(float(row['omegab [rpm]']))
#                 if row['Theater2 [°C]']:
#                     Theater2.append(float(row['Theater2 [°C]']) + 273.15)
#             except ValueError as e:
#                 print(f"Skipping row {i + 1} in file {file_idx + 1} due to error: {e}")

#     # Determine the split index (80% training, 20% testing)
#     split_idx = int(len(omegab) * 0.5)

#     # Append training data
#     train_data["omegab"].extend(omegab[:split_idx])
#     train_data["omega2"].extend(omega2[:split_idx])
#     train_data["Theater2"].extend(Theater2[:split_idx])

#     # Append testing data
#     test_data["omegab"].extend(omegab[split_idx:])
#     test_data["omega2"].extend(omega2[split_idx:])
#     test_data["Theater2"].extend(Theater2[split_idx:])
