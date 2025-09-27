# -------------------------
# IMPORTS
# -------------------------
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from data_filling import (X_scaled, y_scaled, scaler_y, )
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
# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, idx_train, idx_test = train_test_split(X_scaled, np.arange(len(X_scaled)), test_size=0.2)

y_train = [y[idx_train] for y in y_scaled]
y_test = [y[idx_test] for y in y_scaled]

n_features = X_scaled.shape[1]
# ------------------------------------
# Train models: Linear, Polynomial
# ------------------------------------
models_LR, models_PR = [], []
for yt in y_train:
    # # Linear Regression
    # lr = LinearRegression()
    # lr.fit(X_train, yt)
    # models_LR.append(lr)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X_poly = poly.fit_transform(X_train)
    pr = LinearRegression()
    pr.fit(X_poly, yt)
    models_PR.append((pr, poly))

GP_models_dir = 'GP_models'
models_GP = [joblib.load(os.path.join(GP_models_dir, f'GP_model_{i+1}.pkl')) for i in range(6)]

NN_models_dir = 'NN_models'
models_NN = [tf.keras.models.load_model(os.path.join(NN_models_dir, f'NN_model_{i+1}.h5')) for i in range(6)]

# ------------------------------------
# 9. Predictions
# ------------------------------------
# predictions_LR = [model.predict(X_test) for model in models_LR]
predictions_PR = [model.predict(poly.transform(X_test)) for model, poly in models_PR]
predictions_ANN = [model.predict(X_test).flatten() for model in models_NN]
predictions_GP = [model.predict(X_test, return_std=True)[0] for model in models_GP]

# ------------------------------------
# 10. Inverse scaling
# ------------------------------------
y_test_real = [scaler.inverse_transform(yt.reshape(-1, 1)).flatten() for yt, scaler in zip(y_test, scaler_y)]
# y_pred_LR = [scaler.inverse_transform(pred.reshape(-1, 1)).flatten() for pred, scaler in zip(predictions_LR, scaler_y)]
y_pred_PR = [scaler.inverse_transform(pred.reshape(-1, 1)).flatten() for pred, scaler in zip(predictions_PR, scaler_y)]
y_pred_ANN = [scaler.inverse_transform(pred.reshape(-1, 1)).flatten() for pred, scaler in zip(predictions_ANN, scaler_y)]
y_pred_GP = [scaler.inverse_transform(pred.reshape(-1, 1)).flatten() for pred, scaler in zip(predictions_GP, scaler_y)]

# ------------------------------------
# 11. Performance and Plot
# ------------------------------------
labels = [r'$\dot{m}_\text{f}$ [g/s]', r'$\dot{Q}_{\text{h}}$ [kW]', r'$\dot{Q}_{\text{k}}$ [kW]', r'$\dot{W}_{\text{mech}}$ [kW]', r'$T_{\text{dis}}$ [°C]', r'$\dot{Q}_{\text{loss}}$ [kW]']

import os
save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_ML\Comparison'
os.makedirs(save_dir, exist_ok=True)

mape_LR =  []
mape_PR = []
mape_ANN = []
mape_GP = []

mae_LR = []
mae_PR = []
mae_ANN = []
mae_GP = []

r2_LR = []
r2_PR = []
r2_ANN = []
r2_GP = []

for i in range(len(labels)):
    # mape_LR.append(mean_absolute_percentage_error(y_test_real[i], y_pred_LR[i]) * 100)
    mape_PR.append(mean_absolute_percentage_error(y_test_real[i], y_pred_PR[i]) * 100)
    mape_ANN.append(mean_absolute_percentage_error(y_test_real[i], y_pred_ANN[i]) * 100)
    mape_GP.append(mean_absolute_percentage_error(y_test_real[i], y_pred_GP[i]) * 100)

    # mae_LR.append(mean_absolute_error(y_test_real[i], y_pred_LR[i]))
    mae_PR.append(mean_absolute_error(y_test_real[i], y_pred_PR[i]))
    mae_ANN.append(mean_absolute_error(y_test_real[i], y_pred_ANN[i]))
    mae_GP.append(mean_absolute_error(y_test_real[i], y_pred_GP[i]))

    # r2_LR.append(r2_score(y_test_real[i], y_pred_LR[i]) * 100)
    r2_PR.append(r2_score(y_test_real[i], y_pred_PR[i]) * 100)
    r2_ANN.append(r2_score(y_test_real[i], y_pred_ANN[i]) * 100)
    r2_GP.append(r2_score(y_test_real[i], y_pred_GP[i]) * 100)

mdot_real = y_test_real[0] * 1e3
mdot_pred_PR = y_pred_PR[0] * 1e3
mdot_pred_ANN = y_pred_ANN[0] * 1e3
mdot_pred_GP = y_pred_GP[0] * 1e3

Qheater_real = y_test_real[1] * 1e-3
Qheater_pred_PR = y_pred_PR[1] * 1e-3
Qheater_pred_ANN = y_pred_ANN[1] * 1e-3
Qheater_pred_GP = y_pred_GP[1] * 1e-3

Qcooler_real = y_test_real[2] * 1e-3
Qcooler_pred_PR = y_pred_PR[2] * 1e-3
Qcooler_pred_ANN = y_pred_ANN[2] * 1e-3
Qcooler_pred_GP = y_pred_GP[2] * 1e-3

Wmech_real = y_test_real[3] * 1e-3
Wmech_pred_PR = y_pred_PR[3] * 1e-3
Wmech_pred_ANN = y_pred_ANN[3] * 1e-3
Wmech_pred_GP = y_pred_GP[3] * 1e-3

Tdis_real = y_test_real[4] - 273.15
Tdis_pred_PR = y_pred_PR[4] - 273.15
Tdis_pred_ANN = y_pred_ANN[4] - 273.15
Tdis_pred_GP = y_pred_GP[4] - 273.15

Qloss_real = y_test_real[5]  * 1e-3
Qloss_pred_PR = y_pred_PR[5]  * 1e-3
Qloss_pred_ANN = y_pred_ANN[5]  * 1e-3
Qloss_pred_GP = y_pred_GP[5]  * 1e-3

fig = plt.figure(1)
plt.plot(mdot_real, mdot_real, 'k-', label='Ideal line')
y_sorted = np.sort(mdot_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(mdot_real, mdot_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR ($MAPE$={mape_PR[0]:.1f}%, $R^2$={r2_PR[0]:.1f}%)')
plt.scatter(mdot_real, mdot_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'ANN ($MAPE$={mape_ANN[0]:.1f}%, $R^2$={r2_ANN[0]:.1f}%)')
plt.scatter(mdot_real, mdot_pred_GP, color='r', marker='x', label=f'GPR ($MAPE$={mape_GP[0]:.1f}%, $R^2$={r2_GP[0]:.1f}%)')
plt.xlabel(r'Measured mass flow rate $\dot{m}_{\text{f}}$ [g/s]', fontsize=14)
plt.ylabel(r'Predicted mass flow rate $\dot{m}_{\text{f}}$ [g/s]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "mdot_comparison.eps"), format='eps')

fig = plt.figure(2)
plt.plot(Qheater_real, Qheater_real, 'k-', label='Ideal line')
y_sorted = np.sort(Qheater_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(Qheater_real, Qheater_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR ($MAPE$={mape_PR[1]:.1f}%, $R^2$={r2_PR[1]:.1f}%)')
plt.scatter(Qheater_real, Qheater_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'ANN ($MAPE$={mape_ANN[1]:.1f}%, $R^2$={r2_ANN[1]:.1f}%)')
plt.scatter(Qheater_real, Qheater_pred_GP, color='r', marker='x', label=f'GPR ($MAPE$={mape_GP[1]:.1f}%, $R^2$={r2_GP[1]:.1f}%)')
plt.xlabel(r'Measured heater heat transfer rate $\dot{Q}_{\text{h}}$ [kW]', fontsize=14)
plt.ylabel(r'Predicted heater heat transfer rate $\dot{Q}_{\text{h}}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Qheater_comparison.eps"), format='eps')

fig = plt.figure(3)
plt.plot(Qcooler_real, Qcooler_real, 'k-', label='Ideal line')
y_sorted = np.sort(Qcooler_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(Qcooler_real, Qcooler_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR ($MAPE$={mape_PR[2]:.1f}%, $R^2$={r2_PR[2]:.1f}%)')
plt.scatter(Qcooler_real, Qcooler_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'ANN ($MAPE$={mape_ANN[2]:.1f}%, $R^2$={r2_ANN[2]:.1f}%)')
plt.scatter(Qcooler_real, Qcooler_pred_GP, color='r', marker='x', label=f'GPR ($MAPE$={mape_GP[2]:.1f}%, $R^2$={r2_GP[2]:.1f}%)')
plt.xlabel(r'Measured cooler heat transfer rate $\dot{Q}_{\text{k}}$ [kW]', fontsize=14)
plt.ylabel(r'Predicted cooler heat transfer rate $\dot{Q}_{\text{k}}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Qcooler_comparison.eps"), format='eps')

fig = plt.figure(4)
plt.plot(Wmech_real, Wmech_real, 'k-', label='Ideal line')
y_sorted = np.sort(Wmech_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(Wmech_real, Wmech_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR ($MAPE$={mape_PR[3]:.1f}%, $R^2$={r2_PR[3]:.1f}%)')
plt.scatter(Wmech_real, Wmech_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'ANN ($MAPE$={mape_ANN[3]:.1f}%, $R^2$={r2_ANN[3]:.1f}%)')
plt.scatter(Wmech_real, Wmech_pred_GP, color='r', marker='x', label=f'GPR ($MAPE$={mape_GP[3]:.1f}%, $R^2$={r2_GP[3]:.1f}%)')
plt.xlabel(r'Measured mechanical power $\dot{W}_{\text{mech}}$ [kW]', fontsize=14)
plt.ylabel(r'Predicted mechanical power $\dot{W}_{\text{mech}}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Wmech_comparison.eps"), format='eps')

fig = plt.figure(5)
plt.plot(Tdis_real, Tdis_real, 'k-', label='Ideal line')
y_sorted = np.sort(Tdis_real)
plt.fill_between(y_sorted, [x - 2 for x  in y_sorted], [x + 2 for x  in y_sorted], color='lightgray', label='±2K error')
plt.scatter(Tdis_real, Tdis_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR ($MAE$={mae_PR[4]:.1f}K, $R^2$={r2_PR[4]:.1f}%)')
plt.scatter(Tdis_real, Tdis_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'ANN ($MAE$={mae_ANN[4]:.1f}K, $R^2$={r2_ANN[4]:.1f}%)')
plt.scatter(Tdis_real, Tdis_pred_GP, color='r', marker='x', label=f'GPR ($MAE$={mae_GP[4]:.1f}K, $R^2$={r2_GP[4]:.1f}%)')
plt.xlabel(r'Measured discharge temperature $T_{\text{dis}}$ [°C]', fontsize=14)
plt.ylabel(r'Predicted discharge temperature $T_{\text{dis}}$ [°C]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Tdis_comparison.eps"), format='eps')

fig = plt.figure(6)
plt.plot(Qloss_real, Qloss_real, 'k-', label='Ideal line')
y_sorted = np.sort(Qloss_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(Qloss_real, Qloss_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR ($MAPE$={mape_PR[5]:.1f}%, $R^2$={r2_PR[5]:.1f}%)')
plt.scatter(Qloss_real, Qloss_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'ANN ($MAPE$={mape_ANN[5]:.1f}%, $R^2$={r2_ANN[5]:.1f}%)')
plt.scatter(Qloss_real, Qloss_pred_GP, color='r', marker='x', label=f'GPR ($MAPE$={mape_GP[5]:.1f}%, $R^2$={r2_GP[5]:.1f}%)')
plt.xlabel(r'Measured Heat Loss $\dot{Q}_{\text{loss}}$ [kW]', fontsize=14)
plt.ylabel(r'Predicted Heat Loss $\dot{Q}_{\text{loss}}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Qloss_comparison.eps"), format='eps')
plt.show()