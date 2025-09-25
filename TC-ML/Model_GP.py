# -------------------------
# IMPORTS
# -------------------------
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from data_filling import (X_scaled, y_scaled)

# -------------------------
# SPLIT DATA
# -------------------------
X_train, X_test, idx_train, idx_test = train_test_split(X_scaled, np.arange(len(X_scaled)), test_size=0.2)

y_train = [y[idx_train] for y in y_scaled]
y_test = [y[idx_test] for y in y_scaled]

n_features = X_scaled.shape[1]
# -------------------------
# TRAIN GP
# -------------------------
models_GP = []
for yt in y_train:
    kernel = RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
    GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-3)  # Avoid alpha=0
    GP.fit(X_train, yt)
    models_GP.append(GP)



save_dir = 'GP_models'
os.makedirs(save_dir, exist_ok=True)

# Save GP models
for i, GP in enumerate(models_GP):
    joblib.dump(GP, os.path.join(save_dir, f'GP_model_{i+1}.pkl'))

# GP_models_dir = 'GP_models'
# models_GP = [joblib.load(os.path.join(GP_models_dir, f'GP_model_{i+1}.pkl')) for i in range(6)]
# # -------------------------
# # 9. PREDICT
# # -------------------------
# predictions = []
# for model in models_GP:
#     y_pred, _ = model.predict(X_test, return_std=True)
#     predictions.append(y_pred)

# # -------------------------
# # 10. PLOT RESULTS
# # -------------------------
# labels = ['$\dot{m}_{f}$ [kg/s]', '$\dot{Q}_{heater}$ [W]', '$\dot{Q}_{cooler}$ [W]', '$P_{motor}$ [W]', '$T_{out}$ [°C]', '$\dot{Q}_{loss}$ [W]']
# scalers_dir = 'scalers'
# scaler_X = joblib.load(os.path.join(scalers_dir, 'scaler_X.pkl'))
# scaler_y = [joblib.load(os.path.join(scalers_dir, f'scaler_y_{i+1}.pkl')) for i in range(6)]
# metrics_summary = []

# for i in range(len(labels)):
#     y_true_real = scaler_y[i].inverse_transform(y_test[i].reshape(-1,1)).flatten()
#     y_pred_real = scaler_y[i].inverse_transform(predictions[i].reshape(-1,1)).flatten()

#     rmse = root_mean_squared_error(y_true_real, y_pred_real)
#     mape = mean_absolute_percentage_error(y_true_real, y_pred_real) * 100
#     r2 = r2_score(y_true_real, y_pred_real) * 100

#     metrics_summary.append((labels[i], mape, r2))

#     # Sorting for plotting
#     y_true_sorted = np.sort(y_true_real)

#     plt.figure(figsize=(7,6))
#     plt.plot(y_true_real, y_true_real, 'k-', label='45° line')

#     if '$T_{out}$ [°C]' in labels[i]:
#         plt.fill_between(y_true_sorted, y_true_sorted-2, y_true_sorted+2, color='lightgray', label='±2 K band')
#         plt.scatter(y_true_real, y_pred_real, edgecolors='red', facecolors='none', label=f'GPR (RMSE={rmse:.2f} K, $R^2$={r2:.2f}%)')

#     else:
#         plt.fill_between(y_true_sorted, 0.95*y_true_sorted, 1.05*y_true_sorted, color='lightgray', label='±5% band')
#         plt.scatter(y_true_real, y_pred_real, edgecolors='red', facecolors='none', label=f'GPR (MAPE={mape:.2f}%, $R^2$={r2:.2f}%)')
#     plt.xlabel(f'Measured {labels[i]}', fontsize=14)
#     plt.ylabel(f'Predicted {labels[i]}', fontsize=14)
#     # plt.title(f'Prediction for {labels[i]}', fontsize=16)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

# # plt.show()
# # -------------------------
# # 11. PRINT METRICS
# # -------------------------
# print("\n--- GPR Evaluation Metrics ---")
# for label, mape, r2 in metrics_summary:
#     print(f"{label}: MAPE = {mape:.2f}%, R² = {r2:.2f}%")