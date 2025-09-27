import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
# Create an AbstractState object using the HEOS backend and CO2

data = {
    "omegab [rpm]": [],
    "Tw_in [K]": [],

    "T1_in [K]": [],
    "p1_in [pa]": [],
    "p1_out [pa]": [],
    "Theater1 [K]": [],
    "omega1 [rpm]": [],
    "mdot1 [g/s]": [],
    "Pcooling1 [W]": [],
    "T1_out [K]": [],

    "T2_in [K]": [],
    "p2_in [pa]": [],
    "p2_out [pa]": [],
    "Theater2 [K]": [],
    "omega2 [rpm]": [],
    "mdot [g/s]": [],
    "Pcooling2 [W]": [],
    "T2_out [K]": [],

    "T3_in [K]": [],
    "p3_in [pa]": [],
    "p3_out [pa]": [],
    "omega3 [rpm]": [],
    "Pcooling3 [W]": [],
    "T3_out [K]": [],

    "PRt": [],
    "PR23": [],
    "Pcooling23 [W]": [],
    "Pmotor [W]": [],
    "Pcomb [W]": [],
    "Pheating [W]": [],
    "Prec_total [W]": [],
    "Prec [W]": [],

    "mw_dot [kg/s]": []
}
counter = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\comps 2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        counter += 1
        data['omegab [rpm]'].append(float(row['omegab [rpm]']))
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['T1_in [K]'].append(float(row['T1_in [°C]']) + 273.15)
        data['p1_in [pa]'].append(float(row['p1_in [bar]'])*10**5)
        data['p1_out [pa]'].append(float(row['p1_out [bar]'])*10**5)
        data['Theater1 [K]'].append(float(row['Theater1 [°C]']) + 273.15)
        data['omega1 [rpm]'].append(float(row['omega1 [rpm]']))
        data['mdot1 [g/s]'].append(float(row['mdot1 [g/s]']))
        data['Pcooling1 [W]'].append(float(row['Pcooling1 [W]']))
        data['T1_out [K]'].append(float(row['T1_out [°C]']) + 273.15)

        data['T2_in [K]'].append(float(row['T2_in [°C]']) + 273.15)
        data['p2_in [pa]'].append(float(row['p2_in [bar]'])*10**5)
        data['p2_out [pa]'].append(float(row['p2_out [bar]'])*10**5)
        data['Theater2 [K]'].append(float(row['Theater2 [°C]']) + 273.15)
        data['omega2 [rpm]'].append(float(row['omega2 [rpm]']))
        data['mdot [g/s]'].append(float(row['mdot [g/s]']))
        data['Pcooling2 [W]'].append(float(row['Pcooling2 [W]']))
        data['T2_out [K]'].append(float(row['T2_out [°C]']) + 273.15)

        data['T3_in [K]'].append(float(row['T3_in [°C]']) + 273.15)
        data['p3_in [pa]'].append(float(row['p3_in [bar]'])*10**5)
        data['p3_out [pa]'].append(float(row['p3_out [bar]'])*10**5)
        data['omega3 [rpm]'].append(float(row['omega3 [rpm]']))
        data['Pcooling3 [W]'].append(float(row['Pcooling3 [W]']))
        data['T3_out [K]'].append(float(row['T3_out2 [°C]']) + 273.15)

        data['PRt'].append(float(row['PRt']))
        data['PR23'].append(data['p3_out [pa]'][-1]/data['p2_in [pa]'][-1])
        data['Pcooling23 [W]'].append(data['Pcooling2 [W]'][-1] + data['Pcooling3 [W]'][-1])
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Prec_total [W]'].append(float(row['Prec_total [W]']))
        data['Prec [W]'].append(float(row['Prec [W]']))
        data['mw_dot [kg/s]'].append(float(row['mw_dot [l/min]']))


Pr1 = np.array([s/r for s,r in zip(data['p1_out [pa]'][:], data['p1_in [pa]'][:])])
Pr2 = np.array([s/r for s,r in zip(data['p2_out [pa]'][:], data['p2_in [pa]'][:])])
Pr3 = np.array([s/r for s,r in zip(data['p3_out [pa]'][:], data['p3_in [pa]'][:])])

Tr1 = np.array([s/r for s,r in zip(data['Theater1 [K]'][:], data['Tw_in [K]'][:])])
Tr2 = np.array([s/r for s,r in zip(data['Theater2 [K]'][:], data['Tw_in [K]'][:])])

w1 = np.array(data['omega1 [rpm]'])
w2 = np.array(data['omega2 [rpm]'])
w3 = np.array(data['omega3 [rpm]'])

# mCH4_dot = np.array(data['mCH4_dot [kg/s]'])
omegab = np.array(data['omegab [rpm]'])

X1_array = np.array([Pr1, data['Theater1 [K]'], data['Tw_in [K]'], w1])
X2_array = np.array([Pr2, data['Theater2 [K]'], data['Tw_in [K]'], w2])
X3_array = np.array([data['PR23'], data['Theater2 [K]'], data['Tw_in [K]'], w2])

X4_array = np.array([data['PRt'], omegab, data['Tw_in [K]']])

# X1_array = np.array([Pr1, w1, data['Tw_in [K]']])
# X2_array = np.array([Pr2, Tr2, w2])
# X3_array = np.array([Pr3, omegab, data['Tw_in [K]']])

# X1_array = np.array([data['p1_in [pa]'], data['p1_out [pa]'], data['omega1 [rpm]'], data['Tw_in [K]']])
# X2_array = np.array([data['p2_in [pa]'], data['p2_out [pa]'], data['omega2 [rpm]'], omegab])
# X3_array = np.array([data['p3_in [pa]'], data['p3_out [pa]'], data['omega3 [rpm]'], omegab])

# X1_array = np.array([Pr, Tr]) #, data['omega [rpm]']])

X1 = X1_array.T
X2 = X2_array.T
X3 = X3_array.T
X4 = X4_array.T


y1_1 = np.array(data['mdot1 [g/s]'])
y1_2 = np.array(data['T1_out [K]'])
y1_3 = np.array(data['Pcooling1 [W]'])

y2_1 = np.array(data['mdot [g/s]'])
y2_2 = np.array(data['T2_out [K]'])
y2_3 = np.array(data['Pcooling2 [W]'])

y3_1 = np.array(data['mdot [g/s]'])
y3_2 = np.array(data['T3_out [K]'])
y3_3 = np.array(data['Pcooling23 [W]'])

y4 = np.array(data['Prec_total [W]'])

from sklearn.model_selection import train_test_split
X1_1_train, X1_1_test, y1_1_train, y1_1_test = train_test_split(X1, y1_1, test_size = 0.01, random_state = 0)
X1_2_train, X1_2_test, y1_2_train, y1_2_test = train_test_split(X1, y1_2, test_size = 0.01, random_state = 0)
X1_3_train, X1_3_test, y1_3_train, y1_3_test = train_test_split(X1, y1_3, test_size = 0.01, random_state = 0)

X2_1_train, X2_1_test, y2_1_train, y2_1_test = train_test_split(X2, y2_1, test_size = 0.01, random_state = 0)
X2_2_train, X2_2_test, y2_2_train, y2_2_test = train_test_split(X2, y2_2, test_size = 0.01, random_state = 0)
X2_3_train, X2_3_test, y2_3_train, y2_3_test = train_test_split(X2, y2_3, test_size = 0.01, random_state = 0)

X3_1_train, X3_1_test, y3_1_train, y3_1_test = train_test_split(X3, y3_1, test_size = 0.01, random_state = 0)
X3_2_train, X3_2_test, y3_2_train, y3_2_test = train_test_split(X3, y3_2, test_size = 0.01, random_state = 0)
X3_3_train, X3_3_test, y3_3_train, y3_3_test = train_test_split(X3, y3_3, test_size = 0.01, random_state = 0)

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.01, random_state = 0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model1_1_LR = LinearRegression()
model1_1_LR.fit(X1_1_train, y1_1_train)

model1_2_LR = LinearRegression()
model1_2_LR.fit(X1_2_train, y1_2_train)

model1_3_LR = LinearRegression()
model1_3_LR.fit(X1_3_train, y1_3_train)

model2_1_LR = LinearRegression()
model2_1_LR.fit(X2_1_train, y2_1_train)

model2_2_LR = LinearRegression()
model2_2_LR.fit(X2_2_train, y2_2_train)

model2_3_LR = LinearRegression()
model2_3_LR.fit(X2_3_train, y2_3_train)

model3_1_LR = LinearRegression()
model3_1_LR.fit(X3_1_train, y3_1_train)

model3_2_LR = LinearRegression()
model3_2_LR.fit(X3_2_train, y3_2_train)

model3_3_LR = LinearRegression()
model3_3_LR.fit(X3_3_train, y3_3_train)

# print(model1_1_LR.coef_)
# print(model1_1_LR.instercept_)

model4_LR = LinearRegression()
model4_LR.fit(X4_train, y4_train)

y1_1_pred = model1_1_LR.predict(X1)
y1_2_pred = model1_2_LR.predict(X1)
y1_3_pred = model1_3_LR.predict(X1)

y2_1_pred = model2_1_LR.predict(X2)
y2_2_pred = model2_2_LR.predict(X2)
y2_3_pred = model2_3_LR.predict(X2)

y3_1_pred = model3_1_LR.predict(X3)
y3_2_pred = model3_2_LR.predict(X3)
y3_3_pred = model3_3_LR.predict(X3)

y4_pred = model4_LR.predict(X4)

from sklearn.metrics import mean_absolute_percentage_error, r2_score

# --- MAPE and R² for y1 ---
mape1_1_LR = mean_absolute_percentage_error(y1_1, y1_1_pred) * 100
r2_1_1_LR  = r2_score(y1_1, y1_1_pred)

mape1_2_LR = mean_absolute_percentage_error(y1_2, y1_2_pred) * 100
r2_1_2_LR  = r2_score(y1_2, y1_2_pred)

mape1_3_LR = mean_absolute_percentage_error(y1_3, y1_3_pred) * 100
r2_1_3_LR  = r2_score(y1_3, y1_3_pred)

# # --- MAPE and R² for y2 ---
# mape2_1_LR = mean_absolute_percentage_error(y2_1, y2_1_pred) * 100
# r2_2_1_LR  = r2_score(y2_1, y2_1_pred)

# mape2_2_LR = mean_absolute_percentage_error(y2_2, y2_2_pred) * 100
# r2_2_2_LR  = r2_score(y2_2, y2_2_pred)

# mape2_3_LR = mean_absolute_percentage_error(y2_3, y2_3_pred) * 100
# r2_2_3_LR  = r2_score(y2_3, y2_3_pred)

# --- MAPE and R² for y3 ---
mape3_1_LR = mean_absolute_percentage_error(y3_1, y3_1_pred) * 100
r2_3_1_LR  = r2_score(y3_1, y3_1_pred)

mape3_2_LR = mean_absolute_percentage_error(y3_2, y3_2_pred) * 100
r2_3_2_LR  = r2_score(y3_2, y3_2_pred)

mape3_3_LR = mean_absolute_percentage_error(y3_3, y3_3_pred) * 100
r2_3_3_LR  = r2_score(y3_3, y3_3_pred)

# --- MAPE and R² for y4 ---
mape4_LR = mean_absolute_percentage_error(y4, y4_pred) * 100
r2_4_LR  = r2_score(y4, y4_pred)

# # Optional: print results
# print(f"Y1_1: MAPE = {mape1_1_LR:.2f}%, R² = {r2_1_1_LR}")
# print(f"Y1_2: MAPE = {mape1_2_LR:.2f}%, R² = {r2_1_2_LR}")
# print(f"Y1_3: MAPE = {mape1_3_LR:.2f}%, R² = {r2_1_3_LR}")

# # print(f"Y2_1: MAPE = {mape2_1_LR:.2f}%, R² = {r2_2_1_LR}")
# # print(f"Y2_2: MAPE = {mape2_2_LR:.2f}%, R² = {r2_2_2_LR}")
# # print(f"Y2_3: MAPE = {mape2_3_LR:.2f}%, R² = {r2_2_3_LR}")

# print(f"Y3_1: MAPE = {mape3_1_LR:.2f}%, R² = {r2_3_1_LR}")
# print(f"Y3_2: MAPE = {mape3_2_LR:.2f}%, R² = {r2_3_2_LR}")
# print(f"Y3_3: MAPE = {mape3_3_LR:.2f}%, R² = {r2_3_3_LR}")

# print(f"Y4:    MAPE = {mape4_LR:.2f}%, R² = {r2_4_LR}")
