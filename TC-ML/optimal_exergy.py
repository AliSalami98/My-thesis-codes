import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import tensorflow as tf
from Models_initilization_PR import(
    scaler1_x,
    scaler1_y,
    scaler2_x,
    scaler2_y,
    scaler3_x,
    scaler3_y,
    scaler4_x,
    scaler4_y,
    scaler5_x,
    scaler5_y,

    poly2_reg,
    poly3_reg,
    poly4_reg,
    poly5_reg,
    data,
    Tout_model,
    Pheat_model,
    Pcool_model,
    Pmotor_model,
)

T0 = 298.15
p0 = 101325
state = AbstractState("HEOS", "CO2")
state.update(CP.PT_INPUTS, p0, T0)
h0 = state.hmass()
s0 = state.smass()

mdot_model = tf.keras.models.load_model(r'models/mdot_model.h5')

n = 10
a_Theater = np.linspace(773.15, np.max(data['Th_wall [K]']), 10)
a_Tw_in = np.linspace(np.min(data['Th_wall [K]']), np.max(data['Th_wall [K]']), n)
a_pin = np.linspace(np.min(data['pin [pa]']), np.max(data['pin [pa]']), n)
a_omega = np.linspace(np.min(data['omega [rpm]']), np.max(data['omega [rpm]']), 10)
a_pout_min = [1.15 * x for x in a_pin]
a_pout_max = [1.65 * x for x in a_pin]
a_Pr = np.linspace(1.15, 1.65, 10)

# a_pcharged = [np.sqrt(Pr * x**2)*1e-5 for x in a_pin]

a_mdot = np.zeros((10, 10, 10))
a_Tout = np.zeros((10, 10, 10))
a_Pmotor = np.zeros((10, 10, 10))
a_Pheat = np.zeros((10, 10, 10))
a_Pcool = np.zeros((10, 10, 10))
a_Pout = np.zeros((10, 10, 10))
eff = np.zeros((10, 10, 10))

Ex_out = np.zeros((10, 10, 10))
Ex_h = np.zeros((10, 10, 10))
Ex_k = np.zeros((10, 10, 10))
Ex_destroyed = np.zeros((10, 10, 10))

eff_Ex = np.zeros((10, 10, 10))

pin = 40e5
state.update(CP.PQ_INPUTS, pin, 1)
Tin = state.T() + 5
Tw = 303.15
# Main calculation loop for thermodynamic analysis
for i, Theater in enumerate(a_Theater):
    for j, Pr in enumerate(a_Pr):
        # Update the state for the current pressure and temperature
        state.update(CP.PT_INPUTS, pin, Tin)
        sin = state.smass()
        hin = state.hmass()
        pout = pin * Pr
        for k, omega in enumerate(a_omega):
            # Prepare input data for models
            test1 = np.array([[pin, pout, omega, Theater, Tw]])
            test1_scaled = scaler1_x.transform(test1)
            
            # Mass flow rate prediction and scaling
            mdot_scaled = mdot_model.predict(test1_scaled)
            a_mdot[i, j, k] = scaler1_y.inverse_transform(mdot_scaled.reshape(-1, 1)).flatten()[0]

            # Outlet temperature prediction and scaling
            Tout_scaled = Tout_model.predict(poly2_reg.transform(test1_scaled))
            a_Tout[i, j, k] = scaler2_y.inverse_transform(Tout_scaled.reshape(-1, 1)).flatten()[0]

            # Heating power prediction and scaling
            Pheat_scaled = Pheat_model.predict(poly5_reg.transform(test1_scaled))
            a_Pheat[i, j, k] = scaler5_y.inverse_transform(Pheat_scaled.reshape(-1, 1)).flatten()[0]

            # Motor power prediction and scaling
            Pmotor_scaled = Pmotor_model.predict(poly4_reg.transform(test1_scaled))
            a_Pmotor[i, j, k] = scaler4_y.inverse_transform(Pmotor_scaled.reshape(-1, 1)).flatten()[0]

            # Update state for outlet properties
            state.update(CP.PT_INPUTS, pout, a_Tout[i, j, k])
            sout = state.smass()
            hout = state.hmass()
            a_Pout[i, j, k] = a_mdot[i, j, k] * 1e-3 * (hout - hin)

            # Cooling power
            a_Pcool[i, j, k] = a_Pheat[i, j, k] + a_Pmotor[i, j, k] - a_Pout[i, j, k]

            # Exergy calculations
            psi_in = (hin - h0) - T0 * (sin - s0)
            psi_out = (hout - h0) - T0 * (sout - s0)
            Ex_out[i, j, k] = a_mdot[i, j, k] * 1e-3 * (psi_out - psi_in)

            Ex_h[i, j, k] = (1 - Tw / Theater) * a_Pheat[i, j, k]
            Ex_k[i, j, k] = a_Pcool[i, j, k] * (1 - T0 / Tw)
            Ex_destroyed[i, j, k] = Ex_h[i, j, k] + a_Pmotor[i, j, k] - Ex_out[i, j, k]

            # Efficiency calculations (clamped to [0, 100])
            eff_Ex[i, j, k] = max(min(100 * Ex_out[i, j, k] / (Ex_h[i, j, k] + a_Pmotor[i, j, k]), 100), 0)
            eff[i, j, k] = max(min(100 * a_Pout[i, j, k] / (a_Pheat[i, j, k] + a_Pmotor[i, j, k]), 100), 0)

# 3D Scatter Plot for Exergy Efficiency
X, Y, Z = np.meshgrid(a_Theater, a_Pr, a_omega, indexing='ij')
X, Y, Z = X.flatten(), Y.flatten(), Z.flatten()  # Flattened parameter space
eff_Ex_flat = eff_Ex.flatten()  # Flattened efficiency values

# Create the 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with colors representing exergy efficiency
sc = ax.scatter(X, Y, Z, c=eff_Ex_flat, cmap='viridis', s=50, alpha=0.7)

# Add color bar to indicate exergy efficiency values
cb = plt.colorbar(sc, ax=ax, pad=0.1)
cb.set_label('Exergy Efficiency (%)', fontsize=14)

# Set labels and titles
ax.set_xlabel('Heater Temperature (Theater) [K]', fontsize=12)
ax.set_ylabel('Pressure Ratio (Pr)', fontsize=12)
ax.set_zlabel('Rotation Speed (Omega) [rpm]', fontsize=12)
ax.set_title('Exergy Efficiency as Function of Theater, Pr, and Omega', fontsize=14)

# Adjust view angle for better visualization
ax.view_init(elev=30, azim=120)

# Show the plot
plt.tight_layout()
plt.show()
