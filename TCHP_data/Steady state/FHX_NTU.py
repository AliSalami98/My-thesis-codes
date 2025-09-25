import csv
import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from scipy.optimize import curve_fit# Create an AbstractState object using the HEOS backend and CO2
T0 = 298.15
p0 = 101325
state = AbstractState("HEOS", "CO2")
state.update(CP.PT_INPUTS, p0, T0)
h0 = state.hmass()
s0 = state.smass()
CP.set_reference_state('CO2',T0, p0, h0, s0)

def Delta_T_lm(Tf_in, Tf_out, Tfw_in, Tfw_out):
    Delta_T1 = Tf_in - Tfw_out
    Delta_T2 = Tf_out - Tfw_in

    delta_T_lm = (Delta_T1 - Delta_T2)/np.log(Delta_T1/Delta_T2)  #(data['Tf_in [K]'][i] - data['Tfw_in [K]'][i])
    return delta_T_lm

Dw = 1000
Dmpg = 1100
cpw = 4168
cpg = 1000

data = {
    'Pfhx [W]': [],
    'Tf_in [K]': [],
    'Tf_out [K]': [],
    'Tfw_in [K]': [],
    'Tfw_out [K]': [],
    'mw_dot [kg/s]': [],
    'mg_dot [kg/s]': [],
    'omegab [rpm]': []
}

with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\cycle measurements\all 2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        T13 = float(row['T13'])+ 273.15
        T19 = float(row['T19'])+ 273.15
        T28 = max(float(row['T28'])+ 273.15, T19 + 0.1)
        mw_dot = float(row['mw_dot'])/60
        mCH4_dot = float(row['mCH4_dot [kg/s]'])
        data['mg_dot [kg/s]'].append(mCH4_dot * 18.125)
        # -------------------------------
        # Recovery heat exchanger
        # -------------------------------
        
        Prec = mw_dot*cpw*(T28 - T19)
        data['Pfhx [W]'].append(Prec)
        data['Tf_in [K]'].append(T13)

        data['Tfw_in [K]'].append(T19)
        data['Tfw_out [K]'].append(T28)

        data['mw_dot [kg/s]'].append(mw_dot)
        data['omegab [rpm]'].append(float(row['omegab']))

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from scipy.optimize import brentq

# --- simple cp(T) for flue gas (tweak coefficients to your range) ---
def cp_flue(TK):
    # J/(kg·K); crude linear fit around 300–700 K
    return 1000.0 + 0.25*(TK - 300.0)  # e.g., ~1100–1200 J/kgK in your range

cp_w = 4180.0

# ---------- crossflow ε–NTU (both fluids unmixed) ----------
# ε = 1 - exp{ [exp(-Cr*NTU^0.78) - 1] * NTU^0.22 / Cr }
def eps_crossflow_both_unmixed(NTU, Cr):
    NTU = max(NTU, 1e-9)
    Cr = min(max(Cr, 1e-9), 1.0 - 1e-9)
    return 1.0 - np.exp((np.exp(-Cr * NTU**0.78) - 1.0) * NTU**0.22 / Cr)

def invert_NTU_from_eps_crossflow(eps_target, Cr, NTU_lo=1e-6, NTU_hi=50.0):
    eps_target = float(np.clip(eps_target, 1e-8, 0.999999))
    f = lambda NTU: eps_crossflow_both_unmixed(NTU, Cr) - eps_target
    try:
        return brentq(f, NTU_lo, NTU_hi, maxiter=200)
    except ValueError:
        NTUs = np.linspace(NTU_lo, NTU_hi, 4000)
        vals = np.abs([f(N) for N in NTUs])
        return NTUs[int(np.argmin(vals))]

# --- build arrays from your 'data' dict (as before) ---
Tf_in   = np.array(data['Tf_in [K]'])
Tw_in   = np.array(data['Tfw_in [K]'])
Tw_out  = np.array(data['Tfw_out [K]'])
Q_meas  = np.array(data['Pfhx [W]'])
m_w     = np.array(data['mw_dot [kg/s]'])
m_f     = np.array(data['mg_dot [kg/s]'])
omega   = np.array(data['omegab [rpm]'])
Tw_in_C = Tw_in - 273.15

# ========= 1) IDENTIFY AU via ε–NTU (no Tf_out needed) =========
AU_id, keep = [], []
for i in range(len(Q_meas)):
    # film temperature for cp_f
    Tfilm = 0.5*(Tf_in[i] + Tw_in[i])
    Cw = m_w[i]*cp_w
    Cf = m_f[i]*cp_flue(Tfilm)
    if Cw <= 0 or Cf <= 0: 
        AU_id.append(0.0); continue
    Cmin, Cmax = min(Cw, Cf), max(Cw, Cf)
    dT_in = Tf_in[i] - Tw_in[i]
    if dT_in <= 0 or Q_meas[i] <= 0:
        AU_id.append(0.0); continue
    Qmax = Cmin*dT_in
    eps  = np.clip(Q_meas[i]/Qmax, 1e-8, 0.999999)
    Cr   = Cmin/Cmax
    NTU  = invert_NTU_from_eps_crossflow(eps, Cr)
    AU   = NTU*Cmin
    AU_id.append(AU); keep.append(i)

AU_id = np.array(AU_id)

# ========= 2) REGRESS AU on richer features =========
X = np.column_stack([omega, Tw_in_C, m_w, m_f])
reg = LinearRegression()
reg.fit(X[keep], AU_id[keep])
b = reg.intercept_; w = reg.coef_
print("AU_fhx = "
      f"{w[0]:.4g}*omega + {w[1]:.4g}*T_w_in(C) + {w[2]:.4g}*m_w + {w[3]:.4g}*m_f + {b:.4g}")

# ========= 3) VALIDATE: predict Q via ε(NTU_pred,Cr)*Qmax =========
Q_pred = []
for i in range(len(Q_meas)):
    Tfilm = 0.5*(Tf_in[i] + Tw_in[i])
    Cw = m_w[i]*cp_w
    Cf = m_f[i]*cp_flue(Tfilm)
    if Cw <= 0 or Cf <= 0:
        Q_pred.append(0.0); continue
    Cmin, Cmax = min(Cw, Cf), max(Cw, Cf)
    dT_in = Tf_in[i] - Tw_in[i]
    if dT_in <= 0:
        Q_pred.append(0.0); continue
    Qmax = Cmin*dT_in
    Cr   = Cmin/Cmax

    AU_hat = max(0.0, reg.predict([[omega[i], Tw_in_C[i], m_w[i], m_f[i]]])[0])
    NTU_hat = AU_hat / max(Cmin, 1e-9)
    eps_hat = eps_crossflow_both_unmixed(NTU_hat, Cr)
    Q_pred.append(eps_hat * Qmax)

Q_pred = np.array(Q_pred)

# ========= 4) METRICS =========
mape = mean_absolute_percentage_error(Q_meas, Q_pred)*100
r2   = r2_score(Q_meas, Q_pred)
rmse = np.sqrt(np.mean((Q_meas - Q_pred)**2))
print(f"FHX Q (crossflow ε–NTU): MAPE={mape:.2f}%  R²={r2:.3f}  RMSE={rmse:.1f} W")

