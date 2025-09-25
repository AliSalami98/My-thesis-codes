import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
# Create an AbstractState object using the HEOS backend and CO2
T0 = 298.15
p0 = 101325
state = AbstractState("HEOS", "CO2")
state.update(CP.PT_INPUTS, p0, T0)
h0 = state.hmass()
s0 = state.smass()
CP.set_reference_state('CO2',T0, p0, h0, s0)

data = {
    "Tin [K]": [],
    "pin [pa]": [],
    "pout [pa]": [],
    "Th_wall [K]": [],
    "Tw_in [K]": [],
    "omega [rpm]": [],
    "mdot [g/s]": [],
    "Pcomb [W]": [],
    "Pheating [W]": [],
    "Pcooling [W]": [],
    "Pmotor [W]": [],
    "Tout [K]": [],
    "Pout [W]": [],
    "eff [%]": [],
}
counter = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\I-O data4.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        counter += 1
        data['Tin [K]'].append(float(row['Tin [°C]']) + 273.15)
        data['pin [pa]'].append(float(row['pin [bar]'])*10**5)
        data['pout [pa]'].append(float(row['pout [bar]'])*10**5)
        data['Th_wall [K]'].append(float(row['Th_wall[°C]']) + 273.15)
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['omega [rpm]'].append(float(row['omega [rpm]']))
        data['mdot [g/s]'].append(float(row['mdot [g/s]']))
        data['Pcomb [W]'].append(float(row['Pcomb [W]']))
        data['Pheating [W]'].append(float(row['Pheating [W]']))
        data['Pcooling [W]'].append(float(row['Pcooling [W]']))
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Tout [K]'].append(float(row['Tout [°C]']) + 273.15)
        data['Pout [W]'].append(float(row['Pout [W]']))
        data['eff [%]'].append(100*float(row['eff [%]']))
        # if counter > 52:
        #     break

Pr = [s/r for s,r in zip(data['pout [pa]'][:], data['pin [pa]'][:])]
Tr = [s/r for s,r in zip(data['Th_wall [K]'][:], data['Tw_in [K]'][:])]

X1_array = np.array([data['pin [pa]'],data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]'], data['Tw_in [K]']])
X2_array = np.array([data['pin [pa]'],data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]'], data['Tw_in [K]']])
X3_array = np.array([data['pin [pa]'],data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]'], data['Tw_in [K]']])
X4_array = np.array([data['pin [pa]'],data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]'], data['Tw_in [K]']])
X5_array = np.array([data['pin [pa]'],data['pout [pa]'], data['omega [rpm]'], data['Th_wall [K]'], data['Tw_in [K]']])

X1 = X1_array.T
X2 = X2_array.T
X3 = X3_array.T
X4 = X4_array.T
X5 = X5_array.T

y1 = np.array(data['mdot [g/s]'])
y2 = np.array(data['Pcomb [W]'])
y3 = np.array(data['Pcooling [W]'])
y4 = np.array(data['Pmotor [W]'])
y5 = np.array(data['Tout [K]'])
from sklearn.preprocessing import StandardScaler,  MinMaxScaler

# scaler1 = MinMaxScaler(feature_range=(0.1, 0.9))
# scaler2 = MinMaxScaler(feature_range=(0.1, 0.9))
# scaler3 = MinMaxScaler(feature_range=(0.1, 0.9))
# scaler4 = MinMaxScaler(feature_range=(0.1, 0.9))
# scaler5 = MinMaxScaler(feature_range=(0.1, 0.9))

# # # Normalize X
# X1 = scaler1.fit_transform(X1)
# X2 = scaler2.fit_transform(X2)
# X3 = scaler3.fit_transform(X3)
# X4 = scaler4.fit_transform(X4)
# X5 = scaler5.fit_transform(X5)

# # # Normalize y1, y2, y3, y4, y5
# y1 = scaler1.fit_transform(y1.reshape(-1, 1)).flatten()
# y2 = scaler2.fit_transform(y2.reshape(-1, 1)).flatten()
# y3= scaler3.fit_transform(y3.reshape(-1, 1)).flatten()
# y4 = scaler4.fit_transform(y4.reshape(-1, 1)).flatten()
# y5 = scaler5.fit_transform(y5.reshape(-1, 1)).flatten()

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.2, random_state = 0)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.2, random_state = 0)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size = 0.2, random_state = 0)

# y1_test_scaled = scaler1.inverse_transform(y1_test.reshape(-1, 1)).flatten()
# y2_test_scaled = scaler2.inverse_transform(y2_test.reshape(-1, 1)).flatten()
# y3_test_scaled = scaler3.inverse_transform(y3_test.reshape(-1, 1)).flatten()
# y4_test_scaled = scaler4.inverse_transform(y4_test.reshape(-1, 1)).flatten()
# y5_test_scaled = scaler5.inverse_transform(y5_test.reshape(-1, 1)).flatten()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly1_reg = PolynomialFeatures(degree = 2)
X1_poly = poly1_reg.fit_transform(X1_train)
model1_PR = LinearRegression()
model1_PR.fit(X1_poly, y1_train)

poly2_reg = PolynomialFeatures(degree = 2)
X2_poly = poly2_reg.fit_transform(X2_train)
model2_PR = LinearRegression()
model2_PR.fit(X2_poly, y2_train)

poly3_reg = PolynomialFeatures(degree = 2)
X3_poly = poly3_reg.fit_transform(X3_train)
model3_PR = LinearRegression()
model3_PR.fit(X3_poly, y3_train)

poly4_reg = PolynomialFeatures(degree = 2)
X4_poly = poly4_reg.fit_transform(X4_train)
model4_PR = LinearRegression()
model4_PR.fit(X4_poly, y4_train)

poly5_reg = PolynomialFeatures(degree = 2)
X5_poly = poly5_reg.fit_transform(X5_train)
model5_PR = LinearRegression()
model5_PR.fit(X5_poly, y5_train)

y1_PR_pred = model1_PR.predict(poly1_reg.transform(X1_test))
y2_PR_pred = model2_PR.predict(poly2_reg.transform(X2_test))
y3_PR_pred = model3_PR.predict(poly3_reg.transform(X3_test))
y4_PR_pred = model4_PR.predict(poly4_reg.transform(X4_test))
y5_PR_pred = model5_PR.predict(poly5_reg.transform(X5_test))

# print(X1_test)
# ---------------------------
# Pr model based
# ---------------------------

n = 20
a_Pr = np.linspace(np.min(Pr), np.max(Pr), n)
max_pout = np.max(data['pout [pa]'])
min_pout = np.min(data['pout [pa]'])
min_pin = max_pout/np.max(Pr)
max_pin = min_pout/np.min(Pr)
a_pin = np.linspace(max_pin, min_pin, n)
a_pout = np.linspace(min_pout, max_pout, n)
a_omega = np.linspace(np.min(data['omega [rpm]']), np.max(data['omega [rpm]']), n)
a_Tr = np.linspace(np.min(Tr), np.max(Tr), n)
max_Th = np.max(data['Th_wall [K]'])
min_Th = np.min(data['Th_wall [K]'])
min_Tw = max_Th/np.max(Tr)
max_Tw = min_Th/np.min(Tr)
a_Tw = np.linspace(max_Tw, min_Tw, n)
a_Th = np.linspace(min_Th, max_Th, n)
# a_Th = np.linspace(np.min(data['Th_wall [K]']), np.max(data['Th_wall [K]']), n)
# a_Tw_in = np.linspace(min(data['Tw_in [K]']), max(data['Tw_in [K]']), n)

model1_3d = [[[0 for _ in a_omega] for _ in a_Th] for _ in a_Pr]
model2_3d = [[[0 for _ in a_omega] for _ in a_Th] for _ in a_Pr]
eff_3d = [[[0 for _ in a_omega] for _ in a_Th] for _ in a_Pr]

# omega_3D = []
# Pr_3D = []
# Tr_3D = []
# # Iterate over each parameter with nested loops
# for i, Pr in enumerate(a_Pr):
#     for j, Th in enumerate(a_Th):
#         for k, omega in enumerate(a_omega):
#             test1 = np.column_stack((a_pin[i], a_pout[i], a_omega[k], a_Th[j], a_Tw[j]))
#             model1_3d[i][j][k] = model1_PR.predict(poly1_reg.transform(test1))
#             # if model1_3d[i][j][k] < 0:
#                 # print(a_Pr[i])
#             model2_3d[i][j][k] = model2_PR.predict(poly2_reg.transform(test1)) * 0.001
#             eff_3d[i][j][k] = model1_3d[i][j][k]/model2_3d[i][j][k]
#             omega_3D.append(a_omega[k])
#             Pr_3D.append(a_Pr[i])
#             Tr_3D.append(a_Tr[j])
# fig1 = plt.figure(1)
# ax = fig1.add_subplot(111, projection='3d')

# # Scatter plot
# sc = ax.scatter(Pr_3D, Tr_3D, omega_3D, c= model1_3d, cmap='jet')

# # Labels
# ax.set_xlabel('$P_r$ [-]', fontsize=14)
# ax.set_ylabel('$T_r$ [-]', fontsize=14)
# ax.set_zlabel('$\omega$ [rpm]', fontsize=14)

# # Color bar
# cbar_ax = fig1.add_axes([0.1, 0.15, 0.03, 0.7])
# cbar = plt.colorbar(sc, cax=cbar_ax)
# cbar.set_label('$\dot{m}$ [kg/s]',fontsize=14, labelpad=-60, y=0.5, rotation=90)
# plt.show()

# -------------------------------------------------
# 2D plots
# -------------------------------------------------

model1_2D = np.zeros((n, n))
model2_2D = np.zeros((n, n))

eff_2D = np.zeros((n, n))

mdot_Tr = np.zeros((n, n))
mdot_opt = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        # test1 = np.column_stack((45*10**5, 60*10**5, a_omega[j], a_Th[i], min_Tw))
        test1 = np.column_stack((a_pin[i], a_pout[i], a_omega[j], 973.15, 303.15))
        model1_2D[i, j] = max(model1_PR.predict(poly1_reg.transform(test1)), min(data['mdot [g/s]'])) # omega are columns and Tr are rows
        model2_2D[i, j] = model5_PR.predict(poly2_reg.transform(test1))-273.15 
        eff_2D[i, j] = model1_2D[i, j]/model2_2D[i, j]

# fig1 = plt.figure(1)
# contourf = plt.contourf(a_omega, a_Th, eff_2D, cmap='jet')  # omega are columns (x-axis) and Tr are rows (y-axis)
# contour = plt.contour(a_omega, a_Th, eff_2D, colors='k')  # The contour lines
# # contou2 = plt.contour(omega_opt, Tr_opt, mdot_opt, colors='k')  # The contour lines
# plt.clabel(contour, colors='k', inline=True)
# cbar = plt.colorbar(contourf)
# cbar.set_label('$\dot{m}$ [g/s]', fontsize=12)
# plt.xlabel('$\omega$ [rpm]', fontsize=12)
# plt.ylabel('$T_r$ [-]', fontsize=12)
# plt.xticks(fontsize= 12)
# plt.yticks(fontsize= 12)

fig1 = plt.figure(1)
contourf = plt.contourf(a_omega, a_Pr, model1_2D, cmap='jet')  # omega are columns (x-axis) and Tr are rows (y-axis)
contour = plt.contour(a_omega, a_Pr, model1_2D, colors='k')  # The contour lines
# contou2 = plt.contour(omega_opt, Tr_opt, mdot_opt, colors='k')  # The contour lines
plt.clabel(contour, colors='k', inline=True)
cbar = plt.colorbar(contourf)
cbar.set_label('$\dot{m}$ [g/s]', fontsize=12)
plt.xlabel('$\omega$ [rpm]', fontsize=12)
plt.ylabel('$P_r$ [-]', fontsize=12)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)

fig2 = plt.figure(2)
contourf = plt.contourf(a_omega, a_Pr, model2_2D, cmap='jet')  # The filled contour plot
contour = plt.contour(a_omega, a_Pr, model2_2D, colors='k')  # The contour lines
plt.clabel(contour, colors='k', inline=True)
cbar = plt.colorbar(contourf)
cbar.set_label('$T_{out}$ [°C]', fontsize=12)
plt.xlabel('$\omega$ [rpm]', fontsize=12)
plt.ylabel('$P_r$ [-]', fontsize=12)
plt.xticks(fontsize= 12)
plt.yticks(fontsize= 12)
plt.show()
# -------------------
# data based
# --------------------
# a_Pr = Pr #np.sort(Pr)
# a_omega = data['omega [rpm]'] #np.sort(data['omega [rpm]'])
# a_Th = data['Th_wall [K]']
# a_Tr = Tr #np.sort(Tr)

# mdot_3D = [[[0 for _ in a_omega] for _ in a_Tr] for _ in a_Pr]
# omega_3D = []
# Pr_3D = []

# fig1 = plt.figure(1)
# ax = fig1.add_subplot(111, projection='3d')

# # Scatter plot
# sc = ax.scatter(a_Pr, a_Tr, a_omega, c= data['mdot [g/s]'], cmap='jet')

# # Labels
# ax.set_xlabel('$P_r$ [-]', fontsize=14)
# ax.set_ylabel('$T_r$ [-]', fontsize=14)
# ax.set_zlabel('$\omega$ [rpm]', fontsize=14)

# # Color bar
# cbar_ax = fig1.add_axes([0.1, 0.15, 0.03, 0.7])
# cbar = plt.colorbar(sc, cax=cbar_ax)
# cbar.set_label('$\dot{m}$ [kg/s]',fontsize=14, labelpad=-60, y=0.5, rotation=90)
# plt.show()








# Tr_opt = np.zeros(n)
# omega_opt = np.zeros(n)
# for j in range(n):
#     for i in range(n):
#         if mdot_opt[i,j] = np.max(model1_2D[:, j]):
#             mdot_opt[i,j] = np.max(model1_2D[:, j])
#             Tr_opt[j] = a_Tr[j]
#             omega_opt[j] = a_omega[j]
# # print(mdot_opt)
      
# mdot_omega_scaled = mdot_omega*1000 
# mdot_Tr_scaled = mdot_Tr*1000
# # Pcomb_omega_scaled = Pcomb_omega*0.001
# # Pcomb_Tr_scaled = Pcomb_Tr*0.001
                   
# # # Create the filled contour plot with contour lines
# fig1 = plt.figure(1)
# contourf = plt.contourf(a_omega, a_Pr, mdot_omega_scaled, cmap='jet')  # The filled contour plot
# contour = plt.contour(a_omega, a_Pr, mdot_omega_scaled, colors='k')  # The contour lines
# plt.clabel(contour, colors='k', inline=True)
# cbar = plt.colorbar(contourf)
# cbar.set_label('$\dot{m}$ [g/s]', fontsize=12)
# plt.xlabel('$\omega$ [rpm]', fontsize=12)
# plt.ylabel('$P_r$ [-]', fontsize=12)
# plt.xticks(fontsize= 12)
# plt.yticks(fontsize= 12)

# fig2 = plt.figure(2)
# contourf = plt.contourf(a_Tr, a_Pr, mdot_Tr_scaled, cmap='viridis')  # The filled contour plot
# contour = plt.contour(a_Tr, a_Pr, mdot_Tr_scaled, colors='k')  # The contour lines
# plt.clabel(contour, colors='k', inline=True)
# cbar = plt.colorbar(contourf)
# cbar.set_label('mass flow rate [g/s]', fontsize=12)
# plt.xlabel('Heater Temperature [°C]', fontsize=12)
# plt.ylabel('Pressure ratio [-]', fontsize=12)
# plt.xticks(fontsize= 12)
# plt.yticks(fontsize= 12)
# plt.show()

# fig3 = plt.figure(3)
# contourf = plt.contourf(a_omega, a_Pr, Pcomb_omega_scaled, cmap='viridis')  # The filled contour plot
# contour = plt.contour(a_omega, a_Pr, Pcomb_omega_scaled, colors='k')  # The contour lines
# plt.clabel(contour, colors='k', inline=True)
# cbar = plt.colorbar(contourf)
# cbar.set_label('$P_{comb}$ [kW]', fontsize=14)
# plt.xlabel('$\omega$ [rpm]', fontsize=14)
# plt.ylabel('$P_r$ [-]', fontsize=14)
# plt.xticks(fontsize= 14)
# plt.yticks(fontsize= 14)

# fig4 = plt.figure(4)
# contourf = plt.contourf(a_Th, a_Pr, Pcomb_Th_scaled, cmap='viridis')  # The filled contour plot
# contour = plt.contour(a_Th, a_Pr, Pcomb_Th_scaled, colors='k')  # The contour lines
# plt.clabel(contour, colors='k', inline=True)
# cbar = plt.colorbar(contourf)
# cbar.set_label('Combustion power [kW]', fontsize=14)
# plt.xlabel('Heater Temperature [°C]', fontsize=14)
# plt.ylabel('Pressure ratio [-]', fontsize=14)
# # plt.xticks(fontsize= 14)
# # plt.yticks(fontsize= 14)
# plt.show()