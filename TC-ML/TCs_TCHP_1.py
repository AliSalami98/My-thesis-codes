import csv
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
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
        data['Pmotor [W]'].append(float(row['Pmotor [W]']))
        data['Prec_total [W]'].append(float(row['Prec_total [W]']))
        data['mw_dot [kg/s]'].append(float(row['mw_dot [l/min]']))

L= len(data['Prec_total [W]'])
a_mdot = np.zeros(L)
a_Tout = np.zeros(L)
a_Pmotor = np.zeros(L)
a_Pheat = np.zeros(L)
a_Pcool = np.zeros(L)
a_Pcp = np.zeros(L)
for j in range(L):

    test1 = np.array([[data['p1_in [pa]'][j], data['p1_out [pa]'][j], data['omega1 [rpm]'][j], data['Theater1 [K]'][j], data['Tw_in [K]'][j], data['T1_in [K]'][j]]])  # shape (1, 6)
    # test1 = np.array([[data['p1_out [pa]'][j]/data['p1_in [pa]'][j], np.sqrt(data['p1_out [pa]'][j]*data['p1_in [pa]'][j]), data['Theater1 [K]'][j], data['Tw_in [K]'][j], data['omega1 [rpm]'][j], data['T1_in [K]'][j]]])  # shape (1, 6)
    test1_scaled = scaler_X.transform(test1)
    predictions_GP = [model.predict(test1_scaled) for model in models_GP]
    y_pred_real_GP = [scaler.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()[0]
                    for pred, scaler in zip(predictions_GP, scalers_y)]

    # Assign predictions to each target array
    a_mdot[j]   = y_pred_real_GP[0]
    # a_Pheat[j]  = y_pred_real_GP[1]
    a_Pcool[j]  = y_pred_real_GP[1]
    # a_Pmotor[j] = y_pred_real_GP[3]
    a_Tout[j]   = y_pred_real_GP[2]


# # Define parity plot function
# def parity_plot(y_true, y_pred, variable_name, unit):
#     r2 = r2_score(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)

#     plt.figure(figsize=(6, 6))
#     plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Predicted vs Measured')
#     plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal')
#     plt.xlabel(f"Measured {variable_name} [{unit}]")
#     plt.ylabel(f"Predicted {variable_name} [{unit}]")
#     plt.title(f"Parity Plot: {variable_name}")
#     plt.text(0.05, 0.95, f"R²: {r2:.2f}\nMAE: {mae:.2f} {unit}", transform=plt.gca().transAxes,
#              fontsize=12, verticalalignment='top', color="darkblue")
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()

# # # Create all parity plots
# # parity_plot(data["mdot1 [kg/s]"], a_mdot, "Mass Flow Rate", "kg/s")
# # parity_plot(data["Pcooling1 [W]"], a_Pcool, "Cooling Power", "W")
# # parity_plot(data["T1_out [K]"], a_Tout, "Outlet Temperature", "K")

# # Create all parity plots
# parity_plot(data["mdot [kg/s]"], a_mdot, "Mass Flow Rate", "kg/s")
# parity_plot(data["Pcooling3 [W]"], a_Pcool, "Cooling Power", "W")
# parity_plot(data["T3_out [K]"], a_Tout, "Outlet Temperature", "K")
# plt.show()

# Calculate pressure ratio (example using p1_out and p1_in)
PR = np.array(data['p1_out [pa]']) / np.array(data['p1_in [pa]'])

# ---- Affine correction (calibration) of GP predictions ----
# Toggle to enable/disable calibration
APPLY_CALIBRATION = True

def fit_affine_correction(y_pred, y_true):
    """
    Fit y_true ≈ a * y_pred + b using least squares.
    Returns (a, b). Falls back to (1.0, 0.0) if degenerate.
    """
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    if np.allclose(y_pred.std(), 0):
        return 1.0, 0.0
    # np.polyfit returns slope a and intercept b for y_true ≈ a*y_pred + b
    a, b = np.polyfit(y_pred, y_true, 1)
    return a, b

def apply_affine(y_pred, params):
    a, b = params
    return a * np.asarray(y_pred) + b

if APPLY_CALIBRATION:
    # Choose the target signals that correspond to your predictions
    ytrue_mdot = np.array(data["mdot1 [kg/s]"])
    ytrue_pcl  = np.array(data["Pcooling1 [W]"])
    ytrue_tout = np.array(data["T1_out [K]"])

    # Fit per-output corrections
    calib_mdot = fit_affine_correction(a_mdot,  ytrue_mdot)
    calib_pcl  = fit_affine_correction(a_Pcool, ytrue_pcl)
    calib_tout = fit_affine_correction(a_Tout,  ytrue_tout)

    print("Calibration (a, b):")
    print(f"  mdot:   {calib_mdot}")
    print(f"  Pcool:  {calib_pcl}")
    print(f"  T_out:  {calib_tout}")

    # Apply corrections
    a_mdot  = apply_affine(a_mdot,  calib_mdot)
    a_Pcool = apply_affine(a_Pcool, calib_pcl)
    a_Tout  = apply_affine(a_Tout,  calib_tout)

    # (Optional) persist for future inference without ground truth
    # import json, os
    # with open(os.path.join(GP_models_dir, "gp_affine_calibration.json"), "w") as f:
    #     json.dump({
    #         "mdot":  {"a": calib_mdot[0], "b": calib_mdot[1]},
    #         "Pcool": {"a": calib_pcl[0],  "b": calib_pcl[1]},
    #         "Tout":  {"a": calib_tout[0], "b": calib_tout[1]},
    #     }, f, indent=2)
# -----------------------------------------------------------
# Sort all values by increasing pressure ratio for smooth plots

sorted_indices = np.argsort(PR)
PR_sorted = PR[sorted_indices]

# Sort target and predicted arrays
mdot_true_sorted = np.array(data['mdot1 [kg/s]'])[sorted_indices]
mdot_pred_sorted = a_mdot[sorted_indices]

Pcool_true_sorted = np.array(data['Pcooling1 [W]'])[sorted_indices]
Pcool_pred_sorted = a_Pcool[sorted_indices]

Tout_true_sorted = np.array(data['T1_out [K]'])[sorted_indices]
Tout_pred_sorted = a_Tout[sorted_indices]


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# ------------ choose the ground truths matching your test1 ------------
# If test1 uses (p2_in, p2_out, omega2, Theater2, Tw_in, T2_in), keep these:
ytrue_mdot = np.asarray(data["mdot1 [kg/s]"]) * 1e3          # kg/s
ytrue_pcl  = np.asarray(data["Pcooling1 [W]"]) * 1e-3         # W
ytrue_tout = np.asarray(data["T1_out [K]"]) - 273.15            # K

# (If you were using TC1: replace with mdot1, Pcooling1, T1_out; for TC23: mdot, Pcooling23, T3_out or T23_out)

# ------------ predictions (already filled above) ------------
yhat_mdot = np.maximum(np.asarray(a_mdot), 0.001) * 1e3   # kg/s
yhat_pcl  = np.asarray(a_Pcool) * 1e-3     # W
yhat_tout = np.asarray(a_Tout) - 273.15      # K

# ------------ metrics ------------
mape_mdot = mean_absolute_percentage_error(ytrue_mdot, yhat_mdot) * 100.0
r2_mdot   = r2_score(ytrue_mdot, yhat_mdot)

mape_pcl  = mean_absolute_percentage_error(ytrue_pcl, yhat_pcl) * 100.0
r2_pcl    = r2_score(ytrue_pcl, yhat_pcl)

mae_tout  = mean_absolute_error(ytrue_tout, yhat_tout)
r2_tout   = r2_score(ytrue_tout, yhat_tout)

print(f"mdot  -> MAPE: {mape_mdot:.2f}% | R²: {r2_mdot:.4f}")
print(f"Pcool -> MAPE: {mape_pcl:.2f}%  | R²: {r2_pcl:.4f}")
print(f"T_out -> MAE:  {mae_tout:.2f} K | R²: {r2_tout:.4f}")

# ------------ helpers ------------
def _annotate_metrics(text):
    plt.text(
        0.68, 0.20, text,
        transform=plt.gca().transAxes,
        fontsize=14, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5),
    )

# ------------ Parity: mdot (kg/s), ±10% error ------------
plt.figure()
plt.plot(ytrue_mdot, ytrue_mdot, c="k", label="Ideal line")
yt_sorted = np.sort(ytrue_mdot)
plt.plot(yt_sorted, 0.9*yt_sorted, "--", label="±10% error", c="b")
plt.plot(yt_sorted, 1.1*yt_sorted, "--", c="b")
plt.scatter(ytrue_mdot, yhat_mdot, s=50, edgecolor="red", facecolor="lightgrey", label="Simulation results", linewidths=2.0)
plt.xlabel(r"Measured $\dot{m}_\mathrm{tc1}$ [g/s]", fontsize=16)
plt.ylabel(r"Predicted $\dot{m}_\mathrm{tc1}$ [g/s]", fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=14)
_annotate_metrics(f"$MAPE$: {mape_mdot:.1f}%\n$R^2$: {r2_mdot:.2f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_mdot_TC1.eps", format='eps', bbox_inches='tight')

# ------------ Parity: Pcool (W), ±10% error ------------
plt.figure()
plt.plot(ytrue_pcl, ytrue_pcl, c="k", label="Ideal line")
yt_sorted = np.sort(ytrue_pcl)
plt.plot(yt_sorted, 0.9*yt_sorted, "--", label="±10% error", c="b")
plt.plot(yt_sorted, 1.1*yt_sorted, "--", c="b")
plt.scatter(ytrue_pcl, yhat_pcl, s=50, edgecolor="red", facecolor="lightgrey", label="Simulation results", linewidths=2.0)
plt.xlabel(r"Measured $\dot{Q}_\mathrm{k1}$ [kW]", fontsize=16)
plt.ylabel(r"Predicted $\dot{Q}_\mathrm{k1}$ [kW]", fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=14)
_annotate_metrics(f"$MAPE$: {mape_pcl:.1f}%\n$R^2$: {r2_pcl:.2f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_Qcooler1.eps", format='eps', bbox_inches='tight')

# ------------ Parity: T_out (K), ±2 K error (report MAE) ------------
plt.figure()
plt.plot(ytrue_tout, ytrue_tout, c="k", label="Ideal line")
yt_sorted = np.sort(ytrue_tout)
plt.plot(yt_sorted, yt_sorted - 2.0, "--", label="±2 K error", c="b")
plt.plot(yt_sorted, yt_sorted + 2.0, "--", c="b")
plt.scatter(ytrue_tout, yhat_tout, s=50, edgecolor="red", facecolor="lightgrey", label="Simulation results", linewidths=2.0)
plt.xlabel(r"Measured $T_\mathrm{tc1, dis}$ [°C]", fontsize=16)
plt.ylabel(r"Predicted $T_\mathrm{tc1, dis}$ [°C]", fontsize=16)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
plt.legend(fontsize=14)
_annotate_metrics(f"$MAE$: {mae_tout:.1f} K\n$R^2$: {r2_tout:.2f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_T_TC1_out.eps", format='eps', bbox_inches='tight')

plt.show()

# # Plot mdot vs PR
# plt.figure(figsize=(8, 5))
# plt.plot(PR_sorted, mdot_true_sorted, 'o-', label='Measured mdot1')
# plt.plot(PR_sorted, mdot_pred_sorted, 's--', label='Predicted mdot1')
# plt.xlabel("Pressure Ratio")
# plt.ylabel("Mass Flow Rate [kg/s]")
# plt.title("Mass Flow Rate vs Pressure Ratio")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()

# # Plot Pcool vs PR
# plt.figure(figsize=(8, 5))
# plt.plot(PR_sorted, Pcool_true_sorted, 'o-', label='Measured Pcooling1')
# plt.plot(PR_sorted, Pcool_pred_sorted, 's--', label='Predicted Pcooling1')
# plt.xlabel("Pressure Ratio")
# plt.ylabel("Cooling Power [W]")
# plt.title("Cooling Power vs Pressure Ratio")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()

# # Plot Tout vs PR
# plt.figure(figsize=(8, 5))
# plt.plot(PR_sorted, Tout_true_sorted, 'o-', label='Measured T1_out')
# plt.plot(PR_sorted, Tout_pred_sorted, 's--', label='Predicted T1_out')
# plt.xlabel("Pressure Ratio")
# plt.ylabel("Outlet Temperature [K]")
# plt.title("Outlet Temperature vs Pressure Ratio")
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()

# plt.show()
