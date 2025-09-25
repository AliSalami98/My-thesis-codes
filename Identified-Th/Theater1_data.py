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
    r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\M1 step.csv',
    # r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\all in\LPV step.csv',


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
                if row['Lpev']:
                    Lpev.append(float(row['Lpev']))
                if row['omega1 [rpm]']:
                    omega1.append(float(row['omega1 [rpm]']))
                if row['omegab [rpm]']:
                    omegab.append(float(row['omegab [rpm]']))
                if row['Theater1 [°C]']:
                    Theater1.append(float(row['Theater1 [°C]']) + 273.15)
            except ValueError as e:
                print(f"Skipping row {i} in file {file_idx + 1} due to error: {e}")
