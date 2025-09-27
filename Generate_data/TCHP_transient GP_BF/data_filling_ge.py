import csv
import numpy as np
# Initialize lists to hold the d_column data
d_t = []
d_Hpev = []
d_Lpev = []
d_mw_dot = []
d_mmpg_dot = []
d_Tw_in = []
d_Tmpg_in = []
d_Tw_out = []
d_Tmpg_out = []
d_pc = []
d_p25 = []
d_pi = []
d_pe = []
d_Tc_out = []
d_Te_out = []
d_omega1 = []
d_omega2 = []
d_omega3 = []
d_omegab = []
d_Theater1 = []
d_Theater2 = []
d_Pcomb = []
d_Pheat_out = []
d_COP = []

i = 0

# Read the CSV file

with open(r'C:\Users\ali.salame\Desktop\Python\Thesis_codes\Generate_data\Create_Data\data_omegab4.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        try:
            # Check for empty strings and convert to float
            if row['Tw_in [°C]']:
                d_Tw_in.append(float(row['Tw_in [°C]']) + 273.15)
            if row['mw_dot [kg/s]']:
                d_mw_dot.append(float(row['mw_dot [kg/s]']))
            if row['Hpev']:
                d_Hpev.append(float(row['Hpev']))
            if row['omegab [rpm]']:
                d_omegab.append(float(row['omegab [rpm]']))   
            # if row['omega2 [rpm]']:
            #     d_omega2.append(float(row['omega2 [rpm]']))         
        except ValueError as e:
            print(f"Skipping row {i} due to error: {e}")
            


