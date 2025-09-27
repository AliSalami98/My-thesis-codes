import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import os
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# 1) Point to your installed file (from your screenshot)
path = r"C:\Users\ali.salame\AppData\Local\Microsoft\Windows\Fonts\CHARTERBT-ROMAN.OTF"
# (add the Bold/Italic too if you use them)
# fm.fontManager.addfont(r"...\CHARTERBT-BOLD.OTF")
# fm.fontManager.addfont(r"...\CHARTERBT-ITALIC.OTF")

# 2) Register and use the exact internal name
fm.fontManager.addfont(path)
prop = fm.FontProperties(fname=path)
mpl.rcParams["font.family"] = prop.get_name()   # e.g., "Bitstream Charter"
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10

GP_models_dir = 'GP_models'
models_GP = [joblib.load(os.path.join(GP_models_dir, f'GP_model_{i+1}.pkl')) for i in range(3)]

scalers_dir = 'scalers'
scaler_X = joblib.load(os.path.join(scalers_dir, 'scaler_X.pkl'))
scalers_y = [joblib.load(os.path.join(scalers_dir, f'scaler_y_{i+1}.pkl')) for i in range(3)]

# Data loading and preprocessing remain unchanged
data = {
    "omegab [rpm]": [], "Tw_in [K]": [], "T1_in [K]": [], "p1_in [pa]": [], "p1_out [pa]": [],
    "Theater1 [K]": [], "omega1 [rpm]": [], "mdot1 [kg/s]": [], "Pcooling1 [W]": [], "T1_out [K]": [],
    "T2_in [K]": [], "p2_in [pa]": [], "p2_out [pa]": [], "Theater2 [K]": [], "omega2 [rpm]": [],
    "mdot [kg/s]": [], "Pcooling2 [W]": [], "T2_out [K]": [], "T3_in [K]": [], "p3_in [pa]": [],
    "p3_out [pa]": [], "omega3 [rpm]": [], "Pcooling3 [W]": [], "T3_out [K]": [], "PRt": [],
    "PR23": [], "Pcooling23 [W]": [], "Pcomb [W]": [], "Pmotor [W]": [], "Pheating [W]": [],
    "Prec_total [W]": [], "mw_dot [kg/s]": []
}

counter = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\ML correlations\comps 2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter=';')
    for row in csv_reader:
        counter += 1
        data['omegab [rpm]'].append(float(row['omegab [rpm]']))
        data['Tw_in [K]'].append(float(row['Tw_in [°C]']) + 273.15)
        data['T1_in [K]'].append(float(row['T1_in [°C]']) + 273.15)
        data['p1_in [pa]'].append(float(row['p1_in [bar]'])*10**5)
        data['p1_out [pa]'].append(float(row['p1_out [bar]'])*10**5)
        data['Theater1 [K]'].append(float(row['Theater1 [°C]']) + 273.15)
        data['omega1 [rpm]'].append(float(row['omega1 [rpm]']))
        data['mdot1 [kg/s]'].append(float(row['mdot1 [g/s]']) * 1e-3)
        data['Pcooling1 [W]'].append(float(row['Pcooling1 [W]']))
        data['T1_out [K]'].append(float(row['T1_out [°C]']) + 273.15)
        data['T2_in [K]'].append(float(row['T2_in [°C]']) + 273.15)
        data['p2_in [pa]'].append(float(row['p2_in [bar]'])*10**5)
        data['p2_out [pa]'].append(float(row['p2_out [bar]'])*10**5)
        data['Theater2 [K]'].append(float(row['Theater2 [°C]']) + 273.15)
        data['omega2 [rpm]'].append(float(row['omega2 [rpm]']))
        data['mdot [kg/s]'].append(float(row['mdot [g/s]']) * 1e-3)
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
        # data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        # data['Prec_total [W]'].append(float(row['Prec_total [W]']))
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
Theater1 = np.array(data['Theater1 [K]']) - 273.15
Theater2 = np.array(data['Theater2 [K]']) - 273.15
Tw = np.array(data['Tw_in [K]']) - 273.15
T1_in = np.array(data['T1_in [K]']) - 273.15
T2_in = np.array(data['T2_in [K]']) - 273.15
T3_in = np.array(data['T3_in [K]']) - 273.15

X1_array = np.array([Pr1, Theater1, Tw, w1, T1_in])
X2_array = np.array([Pr2, Theater2, Tw, w2])
# X3_array = np.array([data['PR23'], Theater2, Tw, w2, T2_in])
X3_array = np.array([Pr3, Theater2, Tw, w3, T3_in])

X4_array = np.array([w1 * w2 * w3])

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


mdot_TC1 = np.array(data['mdot1 [kg/s]']) * 1e3
T_TC1_out = np.array(data['T1_out [K]']) - 273.15
Qcooler1 = np.array(data['Pcooling1 [W]']) * 1e-3

Qcooler2 = np.array(data['mdot [kg/s]']) * 1e3
y2_2 = np.array(data['T2_out [K]']) - 273.15
y2_3 = np.array(data['Pcooling2 [W]']) * 1e-3

y3_1 = np.array(data['mdot [kg/s]']) * 1e3
y3_2 = np.array(data['T3_out [K]'])  - 273.15
y3_3 = np.array(data['Pcooling3 [W]'])* 1e-3

# y4 = np.array(data['Prec_total [W]'])
# y4 = np.array(data['Pelec [W]'])

from sklearn.model_selection import train_test_split
X1_1_train, X1_1_test, mdot_TC1_train, mdot_TC1_test = train_test_split(X1, mdot_TC1, test_size = 0.01, random_state = 0)
X1_2_train, X1_2_test, T_TC1_out_train, T_TC1_out_test = train_test_split(X1, T_TC1_out, test_size = 0.01, random_state = 0)
X1_3_train, X1_3_test, Qcooler1_train, Qcooler1_test = train_test_split(X1, Qcooler1, test_size = 0.01, random_state = 0)

X2_1_train, X2_1_test, Qcooler2_train, Qcooler2_test = train_test_split(X2, Qcooler2, test_size = 0.01, random_state = 0)
X2_2_train, X2_2_test, y2_2_train, y2_2_test = train_test_split(X2, y2_2, test_size = 0.01, random_state = 0)
X2_3_train, X2_3_test, y2_3_train, y2_3_test = train_test_split(X2, y2_3, test_size = 0.01, random_state = 0)

X3_1_train, X3_1_test, y3_1_train, y3_1_test = train_test_split(X3, y3_1, test_size = 0.01, random_state = 0)
X3_2_train, X3_2_test, y3_2_train, y3_2_test = train_test_split(X3, y3_2, test_size = 0.01, random_state = 0)
X3_3_train, X3_3_test, y3_3_train, y3_3_test = train_test_split(X3, y3_3, test_size = 0.01, random_state = 0)

# X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.01, random_state = 0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

model1_1_LR = LinearRegression()
model1_1_LR.fit(X1_1_train, mdot_TC1_train)

# from sklearn.pipeline import make_pipeline

# poly_degree = 2
# poly_model1_1 = make_pipeline(PolynomialFeatures(degree=poly_degree), LinearRegression())
# poly_model1_1.fit(X1_1_train, mdot_TC1_train)


model1_2_LR = LinearRegression()
model1_2_LR.fit(X1_2_train, T_TC1_out_train)

model1_3_LR = LinearRegression()
model1_3_LR.fit(X1_3_train, Qcooler1_train)

model2_1_LR = LinearRegression()
model2_1_LR.fit(X2_1_train, Qcooler2_train)

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


# model4_LR = LinearRegression()
# model4_LR.fit(X4_train, y4_train)

# Model 1_1
print("Model1_1 Coefficients:", model1_1_LR.coef_)
print("Model1_1 Intercept:", model1_1_LR.intercept_)

# Model 1_2
print("Model1_2 Coefficients:", model1_2_LR.coef_)
print("Model1_2 Intercept:", model1_2_LR.intercept_)

# Model 1_3
print("Model1_3 Coefficients:", model1_3_LR.coef_)
print("Model1_3 Intercept:", model1_3_LR.intercept_)

# Model 2_1
print("Model2_1 Coefficients:", model2_1_LR.coef_)
print("Model2_1 Intercept:", model2_1_LR.intercept_)

# Model 2_2
print("Model2_2 Coefficients:", model2_2_LR.coef_)
print("Model2_2 Intercept:", model2_2_LR.intercept_)

# Model 2_3
print("Model2_3 Coefficients:", model2_3_LR.coef_)
print("Model2_3 Intercept:", model2_3_LR.intercept_)

# Model 3_1
print("Model3_1 Coefficients:", model3_1_LR.coef_)
print("Model3_1 Intercept:", model3_1_LR.intercept_)

# Model 3_2
print("Model3_2 Coefficients:", model3_2_LR.coef_)
print("Model3_2 Intercept:", model3_2_LR.intercept_)

# Model 3_3
print("Model3_3 Coefficients:", model3_3_LR.coef_)
print("Model3_3 Intercept:", model3_3_LR.intercept_)

# # Model 4
# print("Model4 Coefficients:", model4_LR.coef_)
# print("Model4 Intercept:", model4_LR.intercept_)
mdot_TC1_pred = model1_1_LR.predict(X1)
T_TC1_out_pred = model1_2_LR.predict(X1)
Qcooler1_pred = model1_3_LR.predict(X1)


mdot_TC2_pred = model2_1_LR.predict(X2)
T_TC2_out_pred = model2_2_LR.predict(X2)
Qcooler2_pred = model2_3_LR.predict(X2)

mdot_TC3_pred = model3_1_LR.predict(X3)
T_TC3_out_pred = model3_2_LR.predict(X3)
Qcooler3_pred = model3_3_LR.predict(X3)

# Pelec_pred = model4_LR.predict(X4)

mdot_TC1_real = np.array(data['mdot1 [kg/s]']) * 1e3
T_TC1_out_real = np.array(data['T1_out [K]']) - 273.15
Qcooler1_real = np.array(data['Pcooling1 [W]']) * 1e-3
Qcooler1_pred = np.array(Qcooler1_pred)

mdot_TC2_real = np.array(data['mdot [kg/s]']) * 1e3
T_TC2_out_real = np.array(data['T2_out [K]']) - 273.15
Qcooler2_real = np.array(data['Pcooling2 [W]']) * 1e-3
Qcooler2_pred = np.array(Qcooler2_pred)

mdot_TC3_real = np.array(data['mdot [kg/s]']) * 1e3
T_TC3_out_real = np.array(data['T3_out [K]']) - 273.15
Qcooler3_real = np.array(data['Pcooling3 [W]']) * 1e-3
Qcooler3_pred = np.array(Qcooler3_pred)

mape_mdot_TC1 = mean_absolute_percentage_error(mdot_TC1_real, mdot_TC1_pred) * 100
r2_mdot_TC1 = r2_score(mdot_TC1_real, mdot_TC1_pred)
mae_T_TC1_out = mean_absolute_error(T_TC1_out_real, T_TC1_out_pred)
r2_T_TC1_out = r2_score(T_TC1_out_real, T_TC1_out_pred)
mape_Qcooler1 = mean_absolute_percentage_error(Qcooler1_real, Qcooler1_pred) * 100
r2_Qcooler1 = r2_score(Qcooler1_real, Qcooler1_pred)


mape_mdot_TC2 = mean_absolute_percentage_error(mdot_TC2_real, mdot_TC2_pred) * 100
r2_mdot_TC2 = r2_score(mdot_TC2_real, mdot_TC2_pred)
mae_T_TC2_out = mean_absolute_error(T_TC2_out_real, T_TC2_out_pred)
r2_T_TC2_out = r2_score(T_TC2_out_real, T_TC2_out_pred)
mape_Qcooler2 = mean_absolute_percentage_error(Qcooler2_real, Qcooler2_pred) * 100
r2_Qcooler2 = r2_score(Qcooler2_real, Qcooler2_pred)

mape_mdot_TC3 = mean_absolute_percentage_error(mdot_TC3_real, mdot_TC3_pred) * 100
r2_mdot_TC3 = r2_score(mdot_TC3_real, mdot_TC3_pred)
mae_T_TC3_out = mean_absolute_error(T_TC3_out_real, T_TC3_out_pred)
r2_T_TC3_out = r2_score(T_TC3_out_real, T_TC3_out_pred)
mape_Qcooler3 = mean_absolute_percentage_error(Qcooler3_real, Qcooler3_pred) * 100
r2_Qcooler3 = r2_score(Qcooler3_real, Qcooler3_pred)

# mape_Pelec = mean_absolute_percentage_error(data['Pelec [W]'], Pelec_pred) * 100
# r2_Pelec = r2_score(data['Pelec [W]'], Pelec_pred)

# ------------ helpers ------------
def _annotate_metrics(text):
    plt.text(
        0.68, 0.20, text,
        transform=plt.gca().transAxes,
        fontsize=12, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5),
    )

# ------------ Parity: mdot (kg/s), ±10% error ------------
plt.figure()
plt.plot(mdot_TC2_real, mdot_TC2_real, c="k", label="Ideal line")
yt_sorted = np.sort(mdot_TC2_real)
plt.plot(yt_sorted, 0.9*yt_sorted, "--", label="±10% error", c="b")
plt.plot(yt_sorted, 1.1*yt_sorted, "--", c="b")
plt.scatter(mdot_TC2_real, mdot_TC2_pred, s=50, edgecolor="red", facecolor="lightgrey", label="Simulated", linewidths=2.0)
plt.xlabel(r"Measured $\dot{m}_\mathrm{tc2}$ [g/s]", fontsize=14)
plt.ylabel(r"Predicted $\dot{m}_\mathrm{tc2}$ [g/s]", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(fontsize=11)
_annotate_metrics(f"$MAPE$: {mape_mdot_TC2:.1f}%\n$R^2$: {r2_mdot_TC2:.2f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_mdot_TC2.eps", format='eps', bbox_inches='tight')

# ------------ Parity: Pcool (W), ±10% error ------------
plt.figure()
plt.plot(Qcooler2_real, Qcooler2_real, c="k", label="Ideal line")
yt_sorted = np.sort(Qcooler2_real)
plt.plot(yt_sorted, 0.9*yt_sorted, "--", label="±10% error", c="b")
plt.plot(yt_sorted, 1.1*yt_sorted, "--", c="b")
plt.scatter(Qcooler2_real, Qcooler2_pred, s=50, edgecolor="red", facecolor="lightgrey", label="Simulated", linewidths=2.0)
plt.xlabel(r"Measured $\dot{Q}_\mathrm{k2}$ [kW]", fontsize=14)
plt.ylabel(r"Predicted $\dot{Q}_\mathrm{k2}$ [kW]", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(fontsize=11)
_annotate_metrics(f"$MAPE$: {mape_Qcooler2:.1f}%\n$R^2$: {r2_Qcooler2:.2f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Qcooler2.eps", format='eps', bbox_inches='tight')

# ------------ Parity: T_out (K), ±2 K error (report MAE) ------------
plt.figure()
plt.plot(T_TC2_out_real, T_TC2_out_real, c="k", label="Ideal line")
yt_sorted = np.sort(T_TC2_out_real)
plt.plot(yt_sorted, yt_sorted - 2.0, "--", label="±2 K error", c="b")
plt.plot(yt_sorted, yt_sorted + 2.0, "--", c="b")
plt.scatter(T_TC2_out_real, T_TC2_out_pred, s=50, edgecolor="red", facecolor="lightgrey", label="Simulated", linewidths=2.0)
plt.xlabel(r"Measured $T_\mathrm{tc2, dis}$ [°C]", fontsize=14)
plt.ylabel(r"Predicted $T_\mathrm{tc2, dis}$ [°C]", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(fontsize=11)
_annotate_metrics(f"$MAE$: {mae_T_TC2_out:.1f} K\n$R^2$: {r2_T_TC2_out:.2f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_T_TC2_out.eps", format='eps', bbox_inches='tight')

plt.show()

# ------------ Parity: mdot (kg/s), ±10% error ------------
plt.figure()
plt.plot(mdot_TC3_real, mdot_TC3_real, c="k", label="Ideal line")
yt_sorted = np.sort(mdot_TC3_real)
plt.plot(yt_sorted, 0.9*yt_sorted, "--", label="±10% error", c="b")
plt.plot(yt_sorted, 1.1*yt_sorted, "--", c="b")
plt.scatter(mdot_TC3_real, mdot_TC3_pred, s=50, edgecolor="red", facecolor="lightgrey", label="Simulated", linewidths=2.0)
plt.xlabel(r"Measured $\dot{m}_\mathrm{tc3}$ [g/s]", fontsize=14)
plt.ylabel(r"Predicted $\dot{m}_\mathrm{tc3}$ [g/s]", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(fontsize=11)
_annotate_metrics(f"$MAPE$: {mape_mdot_TC3:.1f}%\n$R^2$: {r2_mdot_TC3:.2f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_mdot_TC3.eps", format='eps', bbox_inches='tight')

# ------------ Parity: Pcool (W), ±10% error ------------
plt.figure()
plt.plot(Qcooler3_real, Qcooler3_real, c="k", label="Ideal line")
yt_sorted = np.sort(Qcooler3_real)
plt.plot(yt_sorted, 0.9*yt_sorted, "--", label="±10% error", c="b")
plt.plot(yt_sorted, 1.1*yt_sorted, "--", c="b")
plt.scatter(Qcooler3_real, Qcooler3_pred, s=50, edgecolor="red", facecolor="lightgrey", label="Simulated", linewidths=2.0)
plt.xlabel(r"Measured $\dot{Q}_\mathrm{k3}$ [kW]", fontsize=14)
plt.ylabel(r"Predicted $\dot{Q}_\mathrm{k3}$ [kW]", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(fontsize=11)
_annotate_metrics(f"$MAPE$: {mape_Qcooler3:.1f}%\n$R^2$: {r2_Qcooler3:.2f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Qcooler3.eps", format='eps', bbox_inches='tight')

# ------------ Parity: T_out (K), ±2 K error (report MAE) ------------
plt.figure()
plt.plot(T_TC3_out_real, T_TC3_out_real, c="k", label="Ideal line")
yt_sorted = np.sort(T_TC3_out_real)
plt.plot(yt_sorted, yt_sorted - 2.0, "--", label="±2 K error", c="b")
plt.plot(yt_sorted, yt_sorted + 2.0, "--", c="b")
plt.scatter(T_TC3_out_real, T_TC3_out_pred, s=50, edgecolor="red", facecolor="lightgrey", label="Simulated", linewidths=2.0)
plt.xlabel(r"Measured $T_\mathrm{tc3, dis}$ [°C]", fontsize=14)
plt.ylabel(r"Predicted $T_\mathrm{tc3, dis}$ [°C]", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(fontsize=11)
_annotate_metrics(f"$MAE$: {mae_T_TC3_out:.1f} K\n$R^2$: {r2_T_TC3_out:.2f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_T_TC3_out.eps", format='eps', bbox_inches='tight')

plt.show()