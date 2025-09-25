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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from data_filling import (X_scaled, y_scaled, scaler_y, )

# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, idx_train, idx_test = train_test_split(X_scaled, np.arange(len(X_scaled)), test_size=0.2)

y_train = [y[idx_train] for y in y_scaled]
y_test = [y[idx_test] for y in y_scaled]

n_features = X_scaled.shape[1]
# ------------------------------------
# 7. Train models: Linear, Polynomial, ANN, GP
# ------------------------------------
models_LR, models_PR, models_ANN, models_GP = [], [], [], []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models_LR, models_PR = [], []
metrics_summary = {
    f"Target_{i+1}": {
        "LR": {"R2": [], "MAPE": [], "MAE": []},
        "PR": {"R2": [], "MAPE": [], "MAE": []},
        "ANN": {"R2": [], "MAPE": [], "MAE": []},
        "GP": {"R2": [], "MAPE": [], "MAE": []}
    } for i in range(len(y_scaled))
}

for fold_idx, (train_index, test_index) in enumerate(kf.split(X_scaled)):
    print(f"\n--- Fold {fold_idx+1} ---")
    
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train_folds = [y[train_index] for y in y_scaled]
    y_test_folds = [y[test_index] for y in y_scaled]

    for i, yt in enumerate(y_train_folds):
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, yt)
        y_pred_lr = lr.predict(X_test)
        
        # Polynomial Regression
        poly = PolynomialFeatures(degree=2, interaction_only=True)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        pr = LinearRegression()
        pr.fit(X_train_poly, yt)
        y_pred_pr = pr.predict(X_test_poly)

        # Inverse scaling
        y_test_real = scaler_y[i].inverse_transform(y_test_folds[i].reshape(-1, 1)).flatten()
        y_pred_lr = scaler_y[i].inverse_transform(y_pred_lr.reshape(-1, 1)).flatten()
        y_pred_pr = scaler_y[i].inverse_transform(y_pred_pr.reshape(-1, 1)).flatten()

        # Metrics
        r2_lr = r2_score(y_test_real, y_pred_lr) * 100
        r2_pr = r2_score(y_test_real, y_pred_pr) * 100

        mape_lr = mean_absolute_percentage_error(y_test_real, y_pred_lr) * 100
        mape_pr = mean_absolute_percentage_error(y_test_real, y_pred_pr) * 100

        MAE_lr = mean_absolute_error(y_test_real, y_pred_lr)
        MAE_pr = mean_absolute_error(y_test_real, y_pred_pr)

        # ANN
        ann = tf.keras.Sequential([
            tf.keras.layers.Dense(300, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        ann.compile(optimizer='adam', loss='mean_squared_error')
        history = ann.fit(X_train, yt, validation_data=(X_test, y_test_folds[i]), epochs=100, batch_size=20, verbose=0)

        y_pred_ann = ann.predict(X_test).flatten()
        y_pred_ann = scaler_y[i].inverse_transform(y_pred_ann.reshape(-1, 1)).flatten()

        r2_ann = r2_score(y_test_real, y_pred_ann) * 100
        mape_ann = mean_absolute_percentage_error(y_test_real, y_pred_ann) * 100
        MAE_ann = mean_absolute_error(y_test_real, y_pred_ann)

        # GP
        kernel = RBF(length_scale=[1.0]*n_features, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-5)
        GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.0)
        GP.fit(X_train, yt)
        y_pred_GP = GP.predict(X_test)
        y_pred_GP = scaler_y[i].inverse_transform(y_pred_GP.reshape(-1, 1)).flatten()

        r2_GP = r2_score(y_test_real, y_pred_GP) * 100
        mape_GP = mean_absolute_percentage_error(y_test_real, y_pred_GP) * 100
        MAE_GP = mean_absolute_error(y_test_real, y_pred_GP)

        target_key = f"Target_{i+1}"
        metrics_summary[target_key]["LR"]["R2"].append(r2_lr)
        metrics_summary[target_key]["LR"]["MAPE"].append(mape_lr)
        metrics_summary[target_key]["LR"]["MAE"].append(MAE_lr)

        metrics_summary[target_key]["PR"]["R2"].append(r2_pr)
        metrics_summary[target_key]["PR"]["MAPE"].append(mape_pr)
        metrics_summary[target_key]["PR"]["MAE"].append(MAE_pr)

        metrics_summary[target_key]["ANN"]["R2"].append(r2_ann)
        metrics_summary[target_key]["ANN"]["MAPE"].append(mape_ann)
        metrics_summary[target_key]["ANN"]["MAE"].append(MAE_ann)

        metrics_summary[target_key]["GP"]["R2"].append(r2_GP)
        metrics_summary[target_key]["GP"]["MAPE"].append(mape_GP)
        metrics_summary[target_key]["GP"]["MAE"].append(MAE_GP)

        print(f"  Target {i+1}: LR R²={r2_lr:.2f}%, PR R²={r2_pr:.2f}%, ANN R²={r2_ann:.2f}%, GP R²={r2_GP:.2f}%")

print("\n--- Average Metrics Per Target Across All Folds ---")
for target, model_results in metrics_summary.items():
    print(f"\n{target}:")
    for model, metrics in model_results.items():
        r2_mean = np.mean(metrics["R2"])
        r2_std = np.std(metrics["R2"])
        mape_mean = np.mean(metrics["MAPE"])
        mape_std = np.std(metrics["MAPE"])
        MAE_mean = np.mean(metrics["MAE"])
        MAE_std = np.std(metrics["MAE"])
        
        print(f"  {model}: R² = {r2_mean:.2f} ± {r2_std:.2f}, "
              f"MAPE = {mape_mean:.2f} ± {mape_std:.2f}, "
              f"MAE = {MAE_mean:.2f} ± {MAE_std:.2f}")

# ------------------------------------
# Predictions
# ------------------------------------
predictions_LR = [model.predict(X_test) for model in models_LR]
predictions_PR = [model.predict(poly.transform(X_test)) for model, poly in models_PR]
predictions_ANN = [model.predict(X_test).flatten() for model in models_ANN]
predictions_GP = [model.predict(X_test, return_std=True)[0] for model in models_GP]

# ------------------------------------
# 10. Inverse scaling
# ------------------------------------
y_test_real = [scaler.inverse_transform(yt.reshape(-1, 1)).flatten() for yt, scaler in zip(y_test, scaler_y)]
y_pred_LR = [scaler.inverse_transform(pred.reshape(-1, 1)).flatten() for pred, scaler in zip(predictions_LR, scaler_y)]
y_pred_PR = [scaler.inverse_transform(pred.reshape(-1, 1)).flatten() for pred, scaler in zip(predictions_PR, scaler_y)]
y_pred_ANN = [scaler.inverse_transform(pred.reshape(-1, 1)).flatten() for pred, scaler in zip(predictions_ANN, scaler_y)]
y_pred_GP = [scaler.inverse_transform(pred.reshape(-1, 1)).flatten() for pred, scaler in zip(predictions_GP, scaler_y)]

# ------------------------------------
# 11. Performance and Plot
# ------------------------------------
labels = [r'$\dot{m}_\text{f}$ [g/s]', r'$\dot{Q}_{\text{heater}}$ [kW]', r'$\dot{Q}_{\text{cooler}}$ [kW]', r'$\dot{W}_{\text{mech}}$ [kW]', r'$T_{\text{dis}}$ [°C]', r'$\dot{Q}_{\text{loss}}$ [kW]']

import os
save_dir = r'C:\Users\ali.salame\Desktop\plots\Thesis figs\TC_ML\kfold'
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
    mape_LR.append(mean_absolute_percentage_error(y_test_real[i], y_pred_LR[i]) * 100)
    mape_PR.append(mean_absolute_percentage_error(y_test_real[i], y_pred_PR[i]) * 100)
    mape_ANN.append(mean_absolute_percentage_error(y_test_real[i], y_pred_ANN[i]) * 100)
    mape_GP.append(mean_absolute_percentage_error(y_test_real[i], y_pred_GP[i]) * 100)

    mae_LR.append(mean_absolute_error(y_test_real[i], y_pred_LR[i]))
    mae_PR.append(mean_absolute_error(y_test_real[i], y_pred_PR[i]))
    mae_ANN.append(mean_absolute_error(y_test_real[i], y_pred_ANN[i]))
    mae_GP.append(mean_absolute_error(y_test_real[i], y_pred_GP[i]))

    r2_LR.append(r2_score(y_test_real[i], y_pred_LR[i]) * 100)
    r2_PR.append(r2_score(y_test_real[i], y_pred_PR[i]) * 100)
    r2_ANN.append(r2_score(y_test_real[i], y_pred_ANN[i]) * 100)
    r2_GP.append(r2_score(y_test_real[i], y_pred_GP[i]) * 100)

mdot_real = y_test_real[0]
mdot_pred_PR = y_pred_PR[0]
mdot_pred_ANN = y_pred_ANN[0]
mdot_pred_GP = y_pred_GP[0]

Qheater_real = y_test_real[1]
Qheater_pred_PR = y_pred_PR[1]
Qheater_pred_ANN = y_pred_ANN[1]
Qheater_pred_GP = y_pred_GP[1]

Qcooler_real = y_test_real[2]
Qcooler_pred_PR = y_pred_PR[2]
Qcooler_pred_ANN = y_pred_ANN[2]
Qcooler_pred_GP = y_pred_GP[2]

Wmech_real = y_test_real[3]
Wmech_pred_PR = y_pred_PR[3]
Wmech_pred_ANN = y_pred_ANN[3]
Wmech_pred_GP = y_pred_GP[3]

Tdis_real = y_test_real[4]
Tdis_pred_PR = y_pred_PR[4]
Tdis_pred_ANN = y_pred_ANN[4]
Tdis_pred_GP = y_pred_GP[4]

Qloss_real = y_test_real[5]
Qloss_pred_PR = y_pred_PR[5]
Qloss_pred_ANN = y_pred_ANN[5]
Qloss_pred_GP = y_pred_GP[5]

fig = plt.figure(1)
plt.plot(mdot_real, mdot_real, 'k-', label='Perfect Fit')
y_sorted = np.sort(mdot_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(mdot_real, mdot_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR (MAPE={mape_PR[0]:.1f}%, $R^2$={r2_PR[0]:.1f}%)')
plt.scatter(mdot_real, mdot_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'NN (MAPE={mape_ANN[0]:.1f}%, $R^2$={r2_ANN[0]:.1f}%)')
plt.scatter(mdot_real, mdot_pred_GP, color='r', marker='x', label=f'GP (MAPE={mape_GP[0]:.1f}%, $R^2$={r2_GP[0]:.1f}%)')
plt.xlabel(r'Measured Mass Flow Rate $\dot{m}_{\text{f}}$ [g/s]', fontsize=14)
plt.ylabel(r'Predicted Mass Flow Rate $\dot{m}_{\text{f}}$ [g/s]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "mdot_comparison.eps"), format='eps')

fig = plt.figure(2)
plt.plot(Qheater_real, Qheater_real, 'k-', label='Perfect Fit')
y_sorted = np.sort(Qheater_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(Qheater_real, Qheater_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR (MAPE={mape_PR[1]:.1f}%, $R^2$={r2_PR[1]:.1f}%)')
plt.scatter(Qheater_real, Qheater_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'NN (MAPE={mape_ANN[1]:.1f}%, $R^2$={r2_ANN[1]:.1f}%)')
plt.scatter(Qheater_real, Qheater_pred_GP, color='r', marker='x', label=f'GP (MAPE={mape_GP[1]:.1f}%, $R^2$={r2_GP[1]:.1f}%)')
plt.xlabel(r'Measured Heating Power $\dot{Q}_{\text{heater}}$ [kW]', fontsize=14)
plt.ylabel(r'Predicted Heating Power $\dot{Q}_{\text{heater}}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Qheater_comparison.eps"), format='eps')

fig = plt.figure(3)
plt.plot(Qcooler_real, Qcooler_real, 'k-', label='Perfect Fit')
y_sorted = np.sort(Qcooler_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(Qcooler_real, Qcooler_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR (MAPE={mape_PR[2]:.1f}%, $R^2$={r2_PR[2]:.1f}%)')
plt.scatter(Qcooler_real, Qcooler_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'NN (MAPE={mape_ANN[2]:.1f}%, $R^2$={r2_ANN[2]:.1f}%)')
plt.scatter(Qcooler_real, Qcooler_pred_GP, color='r', marker='x', label=f'GP (MAPE={mape_GP[2]:.1f}%, $R^2$={r2_GP[2]:.1f}%)')
plt.xlabel(r'Measured Cooling Power $\dot{Q}_{\text{cooler}}$ [kW]', fontsize=14)
plt.ylabel(r'Predicted Cooling Power $\dot{Q}_{\text{cooler}}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Qcooler_comparison.eps"), format='eps')

fig = plt.figure(4)
plt.plot(Wmech_real, Wmech_real, 'k-', label='Perfect Fit')
y_sorted = np.sort(Wmech_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(Wmech_real, Wmech_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR (MAPE={mape_PR[3]:.1f}%, $R^2$={r2_PR[3]:.1f}%)')
plt.scatter(Wmech_real, Wmech_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'NN (MAPE={mape_ANN[3]:.1f}%, $R^2$={r2_ANN[3]:.1f}%)')
plt.scatter(Wmech_real, Wmech_pred_GP, color='r', marker='x', label=f'GP (MAPE={mape_GP[3]:.1f}%, $R^2$={r2_GP[3]:.1f}%)')
plt.xlabel(r'Measured Mechanical Power $\dot{W}_{\text{mech}}$ [kW]', fontsize=14)
plt.ylabel(r'Predicted Mechanical Power $\dot{W}_{\text{mech}}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Wmech_comparison.eps"), format='eps')

fig = plt.figure(5)
plt.plot(Tdis_real, Tdis_real, 'k-', label='Perfect Fit')
y_sorted = np.sort(Tdis_real)
plt.fill_between(y_sorted, [x - 2 for x  in y_sorted], [x + 2 for x  in y_sorted], color='lightgray', label='±2K error')
plt.scatter(Tdis_real, Tdis_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR (MAE={mae_PR[4]:.1f}K, $R^2$={r2_PR[4]:.1f}%)')
plt.scatter(Tdis_real, Tdis_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'NN (MAE={mae_ANN[4]:.1f}K, $R^2$={r2_ANN[4]:.1f}%)')
plt.scatter(Tdis_real, Tdis_pred_GP, color='r', marker='x', label=f'GP (MAE={mae_GP[4]:.1f}K, $R^2$={r2_GP[4]:.1f}%)')
plt.xlabel(r'Measured Discharge Temperature $T_{\text{dis}}$ [°C]', fontsize=14)
plt.ylabel(r'Predicted Discharge Temperature $T_{\text{dis}}$ [°C]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Tdis_comparison.eps"), format='eps')

fig = plt.figure(6)
plt.plot(Qloss_real, Qloss_real, 'k-', label='Perfect Fit')
y_sorted = np.sort(Qloss_real)
plt.fill_between(y_sorted, 0.95*y_sorted, 1.05*y_sorted, color='lightgray', label='±5% error')
plt.scatter(Qloss_real, Qloss_pred_PR, edgecolors='m', facecolors='none', marker='s', label=f'PR (MAPE={mape_PR[5]:.1f}%, $R^2$={r2_PR[5]:.1f}%)')
plt.scatter(Qloss_real, Qloss_pred_ANN, edgecolors='b', facecolors='none', marker='o', label=f'NN (MAPE={mape_ANN[5]:.1f}%, $R^2$={r2_ANN[5]:.1f}%)')
plt.scatter(Qloss_real, Qloss_pred_GP, color='r', marker='x', label=f'GP (MAPE={mape_GP[5]:.1f}%, $R^2$={r2_GP[5]:.1f}%)')
plt.xlabel(r'Measured Heat Loss $\dot{Q}_{\text{loss}}$ [kW]', fontsize=14)
plt.ylabel(r'Predicted Heat Loss $\dot{Q}_{\text{loss}}$ [kW]', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Qloss_comparison.eps"), format='eps')
plt.show()