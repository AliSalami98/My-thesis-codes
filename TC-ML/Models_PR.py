import csv
import numpy as np
from utils import get_state, CP
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
# Create an AbstractState object using the HEOS backend and CO2


psi_in = []
psi_out = []
Pout = []
Ex_h = []
Ex_k = []
Ex_out = []
eff_ind = []
eff_Ex = []
counter = 0
# ------------------------------------
# 3. Read the data
# ------------------------------------
data = {
    "Tsuc [K]": [], "psuc [pa]": [], "pdis [pa]": [],
    "Theater [K]": [], "Tw_in [K]": [], "omega [rpm]": [], "omegab [rpm]": [],
    "mdot [kg/s]": [], "Pcomb [W]": [], "Pheating [W]": [],
    "Pcooling [W]": [], "Pmotor [W]": [], "Tdis [K]": [],
    "Pcomp [W]": [], "eff [%]": [], "Ploss [W]": [], "Pmech [W]": []
}

with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\Tc experiments\all2_filtered.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        data['Tsuc [K]'].append(float(row['Tsuc [K]']))
        data['psuc [pa]'].append(float(row['psuc [bar]']) * 1e5)
        data['pdis [pa]'].append(float(row['pdis [bar]']) * 1e5)
        data['Theater [K]'].append(float(row['Theater [K]']))
        data['Tw_in [K]'].append(float(row['Tw_in [K]']))
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        # data['omegab [rpm]'].append(float(row['omegab [rpm]']))
        data['mdot [kg/s]'].append(float(row['mdot [kg/s]']))
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Pmech [W]'].append(float(row['Pmotor [W]']) * 0.9)
        data['Tdis [K]'].append(float(row['Tdis [K]']))
        data['Pcomp [W]'].append(float(row['Pcomp [W]']))
        # data['eff [%]'].append(100 * float(row['eff [%]']))
        data['Ploss [W]'].append(float(row['Ploss [W]']))

a_Pr = [s/r for s, r in zip(data['pdis [pa]'], data['psuc [pa]'])]
a_Tr = [s/r for s, r in zip(data['Theater [K]'], data['Tw_in [K]'])]
a_pcharged = [np.sqrt(s * r) for s, r in zip(data['pdis [pa]'], data['psuc [pa]'])]

# Build X dynamically
X = np.column_stack([data['psuc [pa]'], data['pdis [pa]'], data['omega [rpm]'], data['Theater [K]'], data['Tw_in [K]'], data['Tsuc [K]']])
# X = np.column_stack([a_Pr, data['omega [rpm]'], a_Tr, data['Tsuc [K]']])

X1 = X
X2 = X
X3 = X
X4 = X
X5 = X
X6 = X

# X1_array = np.array([a_Pr, a_pcharged, a_Tr, data['omega [rpm]'], data['Tin [K]']])
# X2_array = np.array([a_Pr, a_pcharged, a_Tr, data['omega [rpm]'], data['Tin [K]']])
# X3_array = np.array([a_Pr, a_pcharged, a_Tr, data['omega [rpm]'], data['Tin [K]']])
# X4_array = np.array([a_Pr, a_pcharged, a_Tr, data['omega [rpm]'], data['Tin [K]']])
# X5_array = np.array([a_Pr, a_pcharged, a_Tr, data['omega [rpm]'], data['Tin [K]']])
# X6_array = np.array([a_Pr, a_pcharged, a_Tr, data['omega [rpm]'], data['Tin [K]']])

# X1_array = np.array([a_Pr, a_Tr, a_pcharged, data['omega [rpm]']])
# X2_array = np.array([a_Pr, a_Tr, a_pcharged, data['omega [rpm]']])
# X3_array = np.array([a_Pr, a_Tr, a_pcharged, data['omega [rpm]']])
# X4_array = np.array([a_Pr, a_Tr, a_pcharged, data['omega [rpm]']])
# X5_array = np.array([a_Pr, a_Tr, a_pcharged, data['omega [rpm]']])
# X6_array = np.array([a_Pr, a_Tr, a_pcharged, data['omega [rpm]']])

# X1 = X1_array.T
# X2 = X2_array.T
# X3 = X3_array.T
# X4 = X4_array.T
# X5 = X5_array.T
# X6 = X6_array.T

y1 = np.array(data['mdot [kg/s]'])
y2 = np.array(data['Pheating [W]'])
y3 = np.array(data['Pcooling [W]'])
y4 = np.array(data['Pmech [W]'])
y5 = np.array(data['Tdis [K]'])
y6 = np.array(data['Ploss [W]']) #np.array(data['Pcomb [W]'])

from sklearn.preprocessing import StandardScaler,  MinMaxScaler
scaler1_x = MinMaxScaler(feature_range=(0.1, 0.9))
scaler1_y = MinMaxScaler(feature_range=(0.1, 0.9))

scaler2_x = MinMaxScaler(feature_range=(0.1, 0.9))
scaler2_y = MinMaxScaler(feature_range=(0.1, 0.9))

scaler3_x = MinMaxScaler(feature_range=(0.1, 0.9))
scaler3_y = MinMaxScaler(feature_range=(0.1, 0.9))

scaler4_x = MinMaxScaler(feature_range=(0.1, 0.9))
scaler4_y = MinMaxScaler(feature_range=(0.1, 0.9))

scaler5_x = MinMaxScaler(feature_range=(0.1, 0.9))
scaler5_y = MinMaxScaler(feature_range=(0.1, 0.9))

scaler6_x = MinMaxScaler(feature_range=(0.1, 0.9))
scaler6_y = MinMaxScaler(feature_range=(0.1, 0.9))


X1_scaled = scaler1_x.fit_transform(X1)
y1_scaled = scaler1_y.fit_transform(y1.reshape(-1, 1)).flatten()

X2_scaled = scaler2_x.fit_transform(X2)
y2_scaled = scaler2_y.fit_transform(y2.reshape(-1, 1)).flatten()

X3_scaled = scaler3_x.fit_transform(X3)
y3_scaled = scaler3_y.fit_transform(y3.reshape(-1, 1)).flatten()

X4_scaled = scaler4_x.fit_transform(X4)
y4_scaled = scaler4_y.fit_transform(y4.reshape(-1, 1)).flatten()

X5_scaled = scaler5_x.fit_transform(X5)
y5_scaled = scaler5_y.fit_transform(y5.reshape(-1, 1)).flatten()

X6_scaled = scaler6_x.fit_transform(X6)
y6_scaled = scaler6_y.fit_transform(y6.reshape(-1, 1)).flatten()

from sklearn.model_selection import train_test_split
X1_train_scaled, X1_test_scaled, y1_train_scaled, y1_test_scaled = train_test_split(X1_scaled, y1_scaled, test_size = 0.2, random_state = 0)
X2_train_scaled, X2_test_scaled, y2_train_scaled, y2_test_scaled = train_test_split(X2_scaled, y2_scaled, test_size = 0.2, random_state = 0)
X3_train_scaled, X3_test_scaled, y3_train_scaled, y3_test_scaled = train_test_split(X3_scaled, y3_scaled, test_size = 0.2, random_state = 0)
X4_train_scaled, X4_test_scaled, y4_train_scaled, y4_test_scaled = train_test_split(X4_scaled, y4_scaled, test_size = 0.2, random_state = 0)
X5_train_scaled, X5_test_scaled, y5_train_scaled, y5_test_scaled = train_test_split(X5_scaled, y5_scaled, test_size = 0.2, random_state = 0)
X6_train_scaled, X6_test_scaled, y6_train_scaled, y6_test_scaled = train_test_split(X6_scaled, y6_scaled, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly1_reg = PolynomialFeatures(degree = 2)
X1_poly = poly1_reg.fit_transform(X1_train_scaled)
mdot_model = LinearRegression()
mdot_model.fit(X1_poly, y1_train_scaled)

poly2_reg = PolynomialFeatures(degree = 2)
X2_poly = poly2_reg.fit_transform(X2_train_scaled)
Tout_model = LinearRegression()
Tout_model.fit(X2_poly, y2_train_scaled)

poly3_reg = PolynomialFeatures(degree = 2)
X3_poly = poly3_reg.fit_transform(X3_train_scaled)
Pcool_model = LinearRegression()
Pcool_model.fit(X3_poly, y3_train_scaled)

poly4_reg = PolynomialFeatures(degree = 2)
X4_poly = poly4_reg.fit_transform(X4_train_scaled)
Pmotor_model = LinearRegression()
Pmotor_model.fit(X4_poly, y4_train_scaled)

poly5_reg = PolynomialFeatures(degree = 2)
X5_poly = poly5_reg.fit_transform(X5_train_scaled)
Pheat_model = LinearRegression()
Pheat_model.fit(X5_poly, y5_train_scaled)

poly6_reg = PolynomialFeatures(degree = 2)
X6_poly = poly6_reg.fit_transform(X6_train_scaled)
Pcomb_model = LinearRegression()
Pcomb_model.fit(X6_poly, y6_train_scaled)

y1_pred_scaled = mdot_model.predict(poly1_reg.transform(X1_test_scaled))
y2_pred_scaled = Tout_model.predict(poly2_reg.transform(X2_test_scaled))
y3_pred_scaled = Pcool_model.predict(poly3_reg.transform(X3_test_scaled))
y4_pred_scaled = Pmotor_model.predict(poly4_reg.transform(X4_test_scaled))
y5_pred_scaled = Pheat_model.predict(poly5_reg.transform(X5_test_scaled))
y6_pred_scaled = Pcomb_model.predict(poly6_reg.transform(X6_test_scaled))


mape1 = mean_absolute_percentage_error(y1_test_scaled, y1_pred_scaled) *100
Rsquare1 = r2_score(y1_test_scaled, y1_pred_scaled) *100

mape2 = mean_absolute_percentage_error(y2_test_scaled, y2_pred_scaled) *100
Rsquare2 = r2_score(y2_test_scaled, y2_pred_scaled) *100

mape3 = mean_absolute_percentage_error(y3_test_scaled, y3_pred_scaled) *100
Rsquare3 = r2_score(y3_test_scaled, y3_pred_scaled) *100

mape4 = mean_absolute_percentage_error(y4_test_scaled, y4_pred_scaled) *100
Rsquare4 = r2_score(y4_test_scaled, y4_pred_scaled) *100

mape5 = mean_absolute_percentage_error(y5_test_scaled, y5_pred_scaled) *100
Rsquare5 = r2_score(y5_test_scaled, y5_pred_scaled) *100

mape6 = mean_absolute_percentage_error(y6_test_scaled, y6_pred_scaled) *100
Rsquare6 = r2_score(y6_test_scaled, y6_pred_scaled) *100


# print(mape1, Rsquare1)
# print(mape2, Rsquare2)
# print(mape3, Rsquare3)
# print(mape4, Rsquare4)
# print(mape5, Rsquare5)
# print(mape6, Rsquare6)
