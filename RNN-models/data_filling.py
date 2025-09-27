import csv
import numpy as np

# Initialize lists to hold the training and testing data for each variable
train_data = {"Hpev": [], "omegab": [],"omega1": [],  "Theater1": [], "pc": [], "mw_dot": [], "Tw_in": [], "Tw_out": [], "Pheat_out": []}
test_data = {"Hpev": [], "omegab": [], "omega1": [], "Theater1": [], "pc": [], "mw_dot": [], "Tw_in": [], "Tw_out": [], "Pheat_out": []}

csv_files = [
    # r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\pc_Two_SID\bf step.csv',
    # r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\pc_Two_SID\hpev step.csv',
    r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\pc_Two_SID\bf PRBS2.csv',
    r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\pc_Two_SID\hpev PRBS2.csv',
    # r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\pc_Two_SID\Twi step.csv',
    # r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\control tests\pc_Two_SID\mw step.csv',
    r'C:\Users\ali.salame\Desktop\Python\Thesis_codes\Generate_data\Create_data\data_omegab4.csv',
    # r'C:\Users\ali.salame\Desktop\Python\TCS%20Dynamic%20Model\HP dynamic model\Cycle\Create_Data\data_omegab4.csv',
    # r'C:\Users\ali.salame\Desktop\Python\TCS%20Dynamic%20Model\HP dynamic model\Cycle\Create_Data\data_omegab2.csv',
    # r'C:\Users\ali.salame\Desktop\Python\TCS%20Dynamic%20Model\HP dynamic model\Cycle\Create_Data\data_omegab2_2.csv',
    # r'C:\Users\ali.salame\Desktop\Python\TCS%20Dynamic%20Model\HP dynamic model\Cycle\Create_Data\data_omegab2_3.csv',
    # r'C:\Users\ali.salame\Desktop\Python\TCS%20Dynamic%20Model\HP dynamic model\Cycle\Create_Data\data_omegab2_4.csv',

    # r'C:\Users\ali.salame\Desktop\Python\TCS%20Dynamic%20Model\HP dynamic model\Cycle\Create_Data\data_omegabb.csv',
    # r'C:\Users\ali.salame\Desktop\Python\TCS%20Dynamic%20Model\HP dynamic model\Cycle\Create_Data\data_omegab5.csv',


]

# Process each CSV file
for file_idx, file_path in enumerate(csv_files):
    # Temporary lists to store data for each column
    t, Hpev, omegab, omega1, Theater1, pc, mw_dot, Tw_in, Tw_out, Pheat_out = [], [], [], [], [], [], [], [], [], []
    
    with open(file_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';')
        for i, row in enumerate(csv_reader):
            try:
                if row['Hpev']:
                    Hpev.append(float(row['Hpev']))
                if row['mw_dot [kg/s]']:
                    mw_dot.append(float(row['mw_dot [kg/s]']))
                if row['omegab [rpm]']:
                    omegab.append(float(row['omegab [rpm]']))
                # if row['omega1 [rpm]']:
                #     omega1.append(float(row['omega1 [rpm]']))
                # if row['Theater1 [°C]']:
                #     Theater1.append(float(row['Theater1 [°C]']) + 273.15)
                if row['Tw_in [°C]']:
                    Tw_in.append(float(row['Tw_in [°C]']) + 273.15)
                if row['Tw_out [°C]']:
                    Tw_out.append(float(row['Tw_out [°C]']) + 273.15)
                if row['pc [bar]']:
                    pc.append(float(row['pc [bar]']))
                if row['Pheat_out [W]']:
                    Pheat_out.append(float(row['Pheat_out [W]']))
            except ValueError as e:
                print(f"Skipping row {i + 1} in file {file_idx + 1} due to error: {e}")
            # if i > 20000:
            #     break

    # Determine the split index (80% training, 20% testing)
    split_idx = int(len(Hpev) * 0.8)

    # Append training data
    train_data["Hpev"].extend(Hpev[:split_idx])
    # train_data["omega1"].extend(omega1[:split_idx])
    train_data["omegab"].extend(omegab[:split_idx])
    # train_data["Theater1"].extend(Theater1[:split_idx])
    train_data["pc"].extend(pc[:split_idx])
    train_data["mw_dot"].extend(mw_dot[:split_idx])
    train_data["Tw_in"].extend(Tw_in[:split_idx])
    train_data["Tw_out"].extend(Tw_out[:split_idx])
    train_data["Pheat_out"].extend(Pheat_out[:split_idx])

    # Append testing data
    test_data["Hpev"].extend(Hpev[split_idx:])
    test_data["omegab"].extend(omegab[split_idx:])
    # test_data["omega1"].extend(omega1[split_idx:])
    # test_data["Theater1"].extend(Theater1[split_idx:])
    test_data["pc"].extend(pc[split_idx:])
    test_data["mw_dot"].extend(mw_dot[split_idx:])
    test_data["Tw_in"].extend(Tw_in[split_idx:])
    test_data["Tw_out"].extend(Tw_out[split_idx:])
    test_data["Pheat_out"].extend(Pheat_out[split_idx:])