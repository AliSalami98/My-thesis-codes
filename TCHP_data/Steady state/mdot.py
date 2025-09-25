import numpy as np
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP
import csv
import matplotlib.pyplot as plt

# Create an AbstractState object using the HEOS backend and CO2
state = AbstractState("HEOS", "CO2")

LHV_CH4 = 50*10**6

Dw = 1 #kg/L
Dmpg = 1.04 #kg/L
DCH4 = 0.68*10**(-3) # kg/L
cpw = 4186
cpmpg = 0.7*4186 + 0.3*2500
Apipe = np.pi/4*(12.3*10**(-3))**2


a_T11 = []
a_T22 = []
a_T9 = []
a_T18 = []
a_T24 = []
a_T20 = []
a_T21 = []
a_T23 = []
a_T1 = []
a_T2 = []
a_T3 = []
a_T4 = []
a_T5 = []
a_T6 = []
a_T10 = []
a_T12 = []
a_T19 = []
a_T28 = []

a_Tfume = []
a_Tbuffer1_out = []
a_Tw_buffer1_out = []
a_Theater = []
a_pe = []
a_p25 = []
a_pi = []
a_pc = []
a_h1 = []
a_h2 = []
a_h3 = []
a_h4 = []
a_h5 = []
a_h6 = []
a_h7 = []
a_h11 = []
a_h22 = []
a_hgc_out = []
a_h23 = []
a_h8 = []
a_h24 = []
a_hevap_out = []
a_hbuffer1_out = []

a_Te_out = []

a_Cd = []
a_Delp = []
a_D_gc = []
a_D_IHX = []
a_D_evap = []
a_D_buffer1 = []
a_D_buffer2 = []
a_D_rec = []
a_AU_gc = []
a_AU_IHX = []
a_AU_evap = []
a_AU_buffer1 = []
a_AU_buffer2 = []
a_AU_rec = []
a_LMTD_gc = []
a_LMTD_IHX = []
a_LMTD_evap = []
a_LMTD_buffer1 = []
a_LMTD_buffer2 = []
a_LMTD_rec = []
a_PR1 = []
a_PR2 = []
a_PR3 = []
a_mf1_dot = []
a_mf_dot= []
a_mw2_dot= []
a_mbp_dot = []
a_mhp_dot = []
a_Hpev = []
a_Lpev = []
a_DpHp = []
a_DpLp = []

a_Text = []
a_Pelec = []
a_Pc = []
a_Pe = []
a_PIHX = []
a_Pcomb = []
a_Prec = []
a_Pcomp1 = []
a_Pcomp2 = []
a_Pcomp3 = []
a_Pcooler1 = []
a_Pcooler2 = []
a_Pcooler3 = []
a_Pbuffer1 = []
a_Pbuffer2 = []
a_Pbuffer3 = []
a_PHXs_t = []
a_Pin_comp = []
a_Pout_comp = []
a_Pin = []
a_Pout = []
Econs_cycle = []
Econs_comp = []

Pheat_out = []
COP = []
GUE = []
COP_carnot = []
SLE = []
exergy = []
Res_HP = []
Res_HP_min = []
Res_HP_max = []
Res_TCs = []
f_LPV1 = []
f_LPV2 = []

f_HPV1 = []
f_HPV2 = []

a_Dp_gc = []
a_Dp_ev = []

i = 0
with open(r'C:\Users\ali.salame\Desktop\Thermodynamics\data files\cycle measurements\all 2.csv') as csv_file:    
    csv_reader = csv.DictReader(csv_file, delimiter = ';')
    for row in csv_reader:
        T1 = float(row['T1'])+ 273.15
        T2 = float(row['T2'])+ 273.15
        T3 = float(row['T3'])+ 273.15
        T4 = float(row['T4'])+ 273.15
        T5 = float(row['T5'])+ 273.15
        T6 = float(row['T6'])+ 273.15
        T7 = float(row['T7'])+ 273.15
        T8 = float(row['T8'])+ 273.15
        T9 = float(row['T9'])+ 273.15
        T10 = float(row['T10'])+ 273.15
        T11 = float(row['T11'])+ 273.15
        T12 = float(row['T12'])+ 273.15
        T13 = float(row['T13'])+ 273.15
        T14 = float(row['T14'])+ 273.15
        T15 = float(row['T15'])+ 273.15
        T16 = float(row['T16'])+ 273.15
        T17 = float(row['T17'])+ 273.15
        T18 = float(row['T18'])+ 273.15
        T19 = float(row['T19'])+ 273.15
        T20 = float(row['T20'])+ 273.15
        T21 = float(row['T21'])+ 273.15
        T22 = float(row['T22'])+ 273.15
        T23 = float(row['T23'])+ 273.15
        T24 = float(row['T24'])+ 273.15
        T28 = max(float(row['T28'])+ 273.15, T19 + 0.2)
        Theater = float(row['Theater1'])+ 273.15
        Tw_return = T9
        Tw_supply = T28 #float(row['Tw_supply'])+ 273.15
        a_T19.append(T19)
        a_T28.append(T28)
        Text = float(row['Text']) + 273.15
        Tevap_out = float(row['Tevap_out'])+ 273.15
        Tgc_out = float(row['Tgc_out'])+ 273.15
        pevap = float(row['pe'])*10**5
        p25 = float(row['p25'])*10**5
        pi = float(row['pi'])*10**5
        pgc = float(row['pc'])*10**5
        mf_dot = float(row['mf_dot'])*10**(-3) # kg/s
        mw1_dot = float(row['mw1_dot'])*Dw/60
        mw_dot = float(row['mw_dot'])*Dw/60
        mCH4_dot = float(row['mCH4_dot [kg/s]'])
        mw2_dot = mw_dot - mw1_dot
        mmpg_dot = float(row['mmpg_dot'])*Dmpg/60
        mpg_percentage = -1.136*(Text-273.15) + 43.632
        cpmpg = (100 - mpg_percentage)/100*cpw + mpg_percentage/100 * 2500
        Pelec = float(row['Pelec'])
        Hpev = float(row['Hpev'])
        Lpev = float(row['Lpev'])

        # -------------------------------
        # Combustion
        # -------------------------------
        Pcomb = mCH4_dot*LHV_CH4
        # -------------------------------
        # Recovery heat exchanger
        # -------------------------------
        Prec = mw_dot*cpw*(T28 - T19)
        # ------------------------------
        # Gas cooler
        # ------------------------------
        h22 = CP.PropsSI('H', 'P', pgc, 'T', T22, 'CO2')
        Pc_w = mw2_dot*cpw*(T18-T9)
        h11 = CP.PropsSI('H', 'P', pgc, 'T', T11, 'CO2')
        D11 = CP.PropsSI('D', 'P', pgc, 'T', T11, 'CO2')
        hgc_out = h11 - Pc_w/mf_dot

        a_hgc_out.append(hgc_out)
        # mf_dot = Pc_w/(h11 - h22)
        quality_vapor = CP.PropsSI('Q', 'P', pgc, 'H', hgc_out, 'CO2')
        Tc_out = CP.PropsSI('T', 'P', pgc, 'H', h22, 'CO2')

        # ------------------------------
        # Tank
        # ------------------------------
        hliq_tank = CP.PropsSI('H', 'P', p25, 'Q', 0, 'CO2') #CP.PropsSI('H', 'P', p25, 'Q', 0, 'CO2') # enthalpy of liquid leaving tank
        h7 = CP.PropsSI('H', 'P', p25, 'T', T7, 'CO2') #CP.PropsSI('H', 'T', T8, 'Q', 1, 'CO2')  #  enthalpy of vapor leaving tank
        # ------------------------------
        # Lp valve
        # ------------------------------
        h24 = hliq_tank
        D24 = CP.PropsSI('Dmass', 'P', p25, 'H', h24, 'CO2')
        # m1_dot = Apipe*Cd1*np.sqrt(2*D24*(p25 - pe))
        # ------------------------------
        # Evap
        # ------------------------------
        hevap_out = CP.PropsSI('H', 'P', pevap, 'T', Tevap_out, 'CO2')
        Pe = mmpg_dot*cpmpg*(T20 - T21)
        mf1_dot = Pe/(hevap_out - h24)

        # print(mf1_dot)
        # ------------------------------
        # IHX
        # ------------------------------
        h1 = CP.PropsSI('H', 'P', pevap, 'T', T1, 'CO2')
        h22 = CP.PropsSI('H', 'P', pgc, 'T', T22, 'CO2')

        PIHX = mf1_dot*(h1 - hevap_out) # = mf_dot*(h22 - h23)
        h23 = h22 - PIHX/mf_dot
        # ------------------------------
        # Hp valve
        # ------------------------------
        h8 = h23
        D23 = CP.PropsSI('Dmass', 'P', pgc, 'H', h23, 'CO2')

        
        # if Lpev > 50:
        a_mf_dot.append(mf_dot)
        a_mf1_dot.append(mf1_dot)
        a_mhp_dot.append(Hpev*np.sqrt(2*D23*(pgc - p25)))
        a_mbp_dot.append(Lpev*np.sqrt(2*D24*(p25 - pevap)))
        a_Hpev.append(Hpev)
        a_Dp_gc.append(np.sqrt(2*D23*(pgc - p25)))
        a_Dp_ev.append(np.sqrt(2*D24*(p25 - pevap)))
        if Lpev <= 60:
            f_LPV1.append(mf1_dot/np.sqrt(2*D24*(p25 - pevap)))
        else:
            f_LPV2.append(mf1_dot/np.sqrt(2*D24*(p25 - pevap)))
        if Hpev <= 50:
            f_HPV1.append(mf_dot/np.sqrt(2*D23*(pgc - p25)))
        else:
            f_HPV2.append(mf_dot/np.sqrt(2*D23*(pgc - p25)))
        a_Lpev.append(Lpev)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, r2_score

def _annotate_metrics(text):
    plt.text(
        0.68, 0.20, text,
        transform=plt.gca().transAxes,
        fontsize=12, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, lw=0.5),
    )

# --- Inputs from your accumulators ---
Hpev_all   = np.asarray(a_Hpev)         # valve command for HPV
Dp_gc_all  = np.asarray(a_Dp_gc)        # sqrt(2*rho*Δp_gc) that you computed
mf_true    = np.asarray(a_mf_dot)       # measured mass flow (gas-cooler side)

# You have two lists that were filled conditionally, like LPV:
# if Hpev <= HPEV_SPLIT: f_HPV1.append(...)
# else:                   f_HPV2.append(...)
fHPV1_all  = np.asarray(f_HPV1)         # only for Hpev <= split
fHPV2_all  = np.asarray(f_HPV2)         # only for Hpev >  split

# --- Choose the split used when you filled f_HPV1 / f_HPV2 ---
HPEV_SPLIT = 100   # <-- set this to the exact threshold you used when appending

# --- Masks on the master Hpev array ---
mask1 = Hpev_all <= HPEV_SPLIT
mask2 = Hpev_all >  HPEV_SPLIT

# If you appended in the same order during parsing, map directly:
x1 = Hpev_all[mask1]
x2 = Hpev_all[mask2]
y1 = fHPV1_all
y2 = fHPV2_all

# --- Fit separate polynomials (degree per your choice; here deg=1) ---
deg1, deg2 = 1, 1
coef_HPV1 = np.polyfit(x1, y1, deg1)
coef_HPV2 = np.polyfit(x2, y2, deg2)
poly_HPV1 = np.poly1d(coef_HPV1)
poly_HPV2 = np.poly1d(coef_HPV2)
print(coef_HPV1)
print(coef_HPV2)
# --- Piecewise prediction of f_HPV over ALL samples ---
fHPV_pred = np.empty_like(Hpev_all, dtype=float)
fHPV_pred[mask1] = poly_HPV1(Hpev_all[mask1])
fHPV_pred[mask2] = poly_HPV2(Hpev_all[mask2])

# --- Convert to mass-flow prediction using your Δp term (gas cooler side) ---
mf_pred = fHPV_pred * Dp_gc_all

# --- Metrics (keep only finite pairs) ---
finite = np.isfinite(mf_true) & np.isfinite(mf_pred)
yt = mf_true[finite] * 1e3
yp = mf_pred[finite] * 1e3

mape_hpv = mean_absolute_percentage_error(yt, yp) * 100
r2_hpv   = r2_score(yt, yp)

# --- Parity plot (same style as your other figures) ---
plt.figure()
plt.plot(yt, yt, c="k", label="Ideal line")

yt_sorted = np.sort(yt)
plt.plot(yt_sorted, 0.9*yt_sorted, "--", c="b", label="±10% error")
plt.plot(yt_sorted, 1.1*yt_sorted, "--", c="b")

plt.scatter(yt, yp, s=50, edgecolor="red", facecolor="lightgrey",
            label="Simulated", linewidths=2.0)
_annotate_metrics(f"MAPE: {mape_hpv:.1f}%\nR$^2$: {r2_hpv:.3f}")
plt.xlabel(r"Measured $\dot{m}_\mathrm{hpv}$ [g/s]", fontsize=14)
plt.ylabel(r"Predicted $\dot{m}_\mathrm{hpv}$ [g/s]", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_mdot_hpv.eps", format='eps', bbox_inches='tight')

# --- Arrays from your accumulators ---
Lpev_all   = np.asarray(a_Lpev)        # valve command
fLPV1_all  = np.asarray(f_LPV1)        # only for Lpev<=60 (length <= len(Lpev_all))
fLPV2_all  = np.asarray(f_LPV2)        # only for Lpev>60
Dp_ev_all  = np.asarray(a_Dp_ev)       # sqrt(2*rho*Δp_ev) that you computed
mf1_true   = np.asarray(a_mf1_dot)     # measured mass flow at evaporator side

# --- Build boolean masks on the master Lpev array ---
mask1 = Lpev_all <= 60
mask2 = Lpev_all > 60

# Sanity: ensure lengths match each mask (if you filled f_LPV1/2 in-order alongside Lpev_all)
# If they were appended synchronously inside the same loop, the following mapping works:
x1 = Lpev_all[mask1]
x2 = Lpev_all[mask2]
y1 = fLPV1_all
y2 = fLPV2_all

# --- Fit separate polynomials (choose degree as you like; here deg=3) ---
deg1, deg2 = 2, 2
coef_LPV1 = np.polyfit(x1, y1, deg1)
coef_LPV2 = np.polyfit(x2, y2, deg2)
poly_LPV1 = np.poly1d(coef_LPV1)
poly_LPV2 = np.poly1d(coef_LPV2)

# --- Piecewise prediction of f_LPV over ALL samples ---
fLPV_pred = np.empty_like(Lpev_all, dtype=float)
fLPV_pred[mask1] = poly_LPV1(Lpev_all[mask1])
fLPV_pred[mask2] = poly_LPV2(Lpev_all[mask2])

# --- Convert to mass-flow prediction using your Δp term (already computed in a_Dp_ev) ---
mf1_pred = fLPV_pred * Dp_ev_all

# --- Metrics (vs. measured a_mf1_dot) ---
# Keep only finite pairs (in case of NaNs)
finite = np.isfinite(mf1_true) & np.isfinite(mf1_pred)
mf1_t = mf1_true[finite] * 1e3
mf1_p = mf1_pred[finite] * 1e3

mape_lpv = mean_absolute_percentage_error(mf1_t, mf1_p) * 100
r2_lpv   = r2_score(mf1_t, mf1_p)

# --- Parity plot in your thesis style (ideal line + ±10% bands) ---
plt.figure()
plt.plot(mf1_t, mf1_t, c="k", label="Ideal line")

yt_sorted = np.sort(mf1_t)
plt.plot(yt_sorted, 0.9*yt_sorted, "--", c="b", label="±10% error")
plt.plot(yt_sorted, 1.1*yt_sorted, "--", c="b")

plt.scatter(mf1_t, mf1_p, s=50, edgecolor="red", facecolor="lightgrey",
            label="Simulated", linewidths=2.0)

plt.xlabel(r"Measured $\dot{m}_\mathrm{lpv}$ [g/s]", fontsize=14)
plt.ylabel(r"Predicted $\dot{m}_\mathrm{lpv}$ [g/s]", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.legend(fontsize=11)
_annotate_metrics(f"MAPE: {mape_lpv:.1f}%\nR$^2$: {r2_lpv:.3f}")
plt.tight_layout()
plt.savefig(r"C:\Users\ali.salame\Desktop\plots\Thesis figs\TCHP_data\steady\TCHP_mdot_lpv.eps", format='eps', bbox_inches='tight')

plt.show()

# # ===================== FIGURE 2: LPV =====================
# x2 = np.asarray(a_Lpev)
# y2 = np.asarray(f_LPV1)

# # Fit polynomial (degree 3)
# coef2 = np.polyfit(x2, y2, 3)
# poly2 = np.poly1d(coef2)

# # Predictions & metric
# y2_pred = poly2(x2)
# mape_lpv = mean_absolute_percentage_error(y2, y2_pred) * 100

# # Smooth curve
# x2_fit = np.linspace(x2.min(), x2.max(), 300)
# y2_fit = poly2(x2_fit)

# # Plot (independent figure)
# fig2, ax2 = plt.subplots(figsize=(8, 5))
# ax2.scatter(x2, y2, label='Data', s=60, edgecolors='k', zorder=3)
# ax2.plot(x2_fit, y2_fit, label='Polynomial Fit (deg=3)', lw=2, zorder=2)
# ax2.set_xlabel('LPV', fontsize=14)
# ax2.set_ylabel('f(LPV)', fontsize=14)
# ax2.set_title(f'f(LPV) vs LPV — MAPE: {mape_lpv:.2f}%', fontsize=16, pad=12)
# ax2.legend(fontsize=12, loc='best')
# ax2.grid(True, linestyle='--', alpha=0.6)
# fig2.tight_layout()
# # fig2.savefig('fit_lpv.png', dpi=300, bbox_inches='tight')

# print(f"LPV polynomial coefficients (deg=3): {coef2}")

# # Show both independent figures
# plt.show()

# print(f"MAPE — HPV: {mape_hpv:.2f}% | LPV: {mape_lpv:.2f}%")


# hpv_values = np.array(a_Hpev)  # assuming a_mf_dot corresponds to HPV
# # Example f(HPV) values (dependent variable)
# f_hpv1_values = np.array(f_HPV1)

# coefficients = np.polyfit(hpv_values, f_hpv1_values, 1)

# # Create a polynomial function from the coefficients
# polynomial = np.poly1d(coefficients)

# # Generate x values for plotting the fitted polynomial
# x_fit = np.linspace(10.9, 100, len(hpv_values))
# y_fit = polynomial(x_fit)

# # # Print the coefficients
# a, b = coefficients
# print(f"Fitted polynomial coefficients:\n a = {a}, b = {b}") #, c = {c}")

# plt.figure(figsize=(10, 6))
# plt.scatter(hpv_values, f_hpv1_values, label='Data Points', color='blue')
# plt.plot(x_fit, y_fit, label='Fitted Polynomial', color='orange')
# plt.xlabel('HPV', fontsize=14)
# plt.ylabel('f(HPV)', fontsize=14)
# plt.title('Polynomial Fit for f(HPV)', fontsize=16)
# plt.legend()
# plt.grid()

# from sklearn.metrics import mean_absolute_percentage_error
# mape_hpv = mean_absolute_percentage_error(f_hpv1_values, y_fit) * 100  # Convert to percentage

# Lpv_values = np.array(a_Lpev)  # assuming a_mf_dot corresponds to LPV
# # # Example f(LPV) values (dependent variable)
# f_Lpv1_values = np.array(f_LPV1)

# coefficients = np.polyfit(Lpv_values, f_Lpv1_values, 1)

# # Create a polynomial function from the coefficients
# polynomial = np.poly1d(coefficients)

# # Generate x values for plotting the fitted polynomial
# x_fit = np.linspace(10.9, 100, len(Lpv_values))
# y_fit = polynomial(x_fit)

# # # Print the coefficients
# a, b = coefficients
# print(f"Fitted polynomial coefficients:\n a = {a}, b = {b}") #, c = {c}")

# plt.figure(figsize=(10, 6))
# plt.scatter(Lpv_values, f_Lpv1_values, label='Data Points', color='blue')
# plt.plot(x_fit, y_fit, label='Fitted Polynomial', color='orange')
# plt.xlabel('LPV', fontsize=14)
# plt.ylabel('f(LPV)', fontsize=14)
# plt.title('Polynomial Fit for f(LPV)', fontsize=16)
# plt.legend()
# plt.grid()

# mape_lpv = mean_absolute_percentage_error(f_Lpv1_values, y_fit) * 100  # Convert to percentage

# print(mape_hpv, mape_lpv)
# plt.show()