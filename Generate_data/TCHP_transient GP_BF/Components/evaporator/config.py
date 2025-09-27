import numpy as np


# ----------------------------------------
# Evap
# ----------------------------------------

Ne = 10

A_mpg_orifice = np.pi / 4 * (20 * 10 ** (-3)) ** 2
dh_mpg = 20 * 10 ** (-3)
mu_mpg = 5.7 * 10 ** (-3)
k_mpg = 0.18
D_mpg = 1040
Tmpg = Ne * [0]
Qmpg_conv = Ne * [0]
dTmpgdt = Ne * [0]

Np_evap = 28
Ve_t = (Np_evap - 1) * 0.071 * 10 ** (-3)
ACO2_orifice = np.pi / 4 * (12.3 * 10 ** (-3)) ** 2
Ae_t = (Np_evap - 2) * 0.0475
me_wall_t = Np_evap * 0.32
Le = Ve_t/ACO2_orifice #73.68 * 10 ** (-3)

me_wall = Ne * [0]
Ve = Ne * [0]
V_mpg = Ne * [0]
Ae = Ne * [0]
Ae_orifice = Ne * [0]
dhe = Ne * [0]
dxe = Ne * [0]
me = Ne * [0]
mue_l = Ne * [0]
mue_v = Ne * [0]
m_mpg = Ne * [0]

Ae_m = (Ne + 1) * [0]
dxe_m = (Ne + 1) * [0]
me_m = (Ne + 1) * [0]
dhe_m = (Ne + 1) * [0]
Ve_m = (Ne + 1) * [0]
for i in range(Ne):
    Ve[i] = Ve_t / (Ne)
    Ae_orifice[i] = ACO2_orifice
    dhe[i] = 12.3 * 10 ** (-3)
    Ae[i] = Ae_t / (Ne)
    dxe[i] = Le / Ne
    me_wall[i] = me_wall_t / Ne
    V_mpg[i] = Ve_t / (Ne)
    m_mpg[i] = D_mpg * V_mpg[i]

# for i in range(Ne + 1):
#     Ae_m[i] = (Ae_orifice[i] + Ae_orifice[i + 1]) / 2
#     me_m[i] = (me[i] + me[i + 1]) / 2
#     Ve_m[i] = (Ve[i] + Ve[i + 1]) / 2
#     dhe_m[i] = (dhe[i] + dhe[i + 1]) / 2
#     dxe_m[i] = (dxe[i] + dxe[i + 1]) / 2

Ue = Ne * [0]
Qe_conv = Ne * [0]
De = Ne * [0]
me = Ne * [0]
pe = 0
Te = Ne * [0]

Te_wall = Ne * [0]

Xe = Ne * [0]
se = Ne * [0]
he = Ne * [0]
ke = Ne * [0]
dpdT_nue = Ne * [0]
dDdp_he = Ne * [0]
dDdh_pe = Ne * [0]
mue = Ne * [0]
cpe = Ne * [0]
cve = Ne * [0]
nue = Ne * [0]
ve_n = Ne * [0]
Tw = Ne * [0]
he_l = Ne * [0]
he_v = Ne * [0]

dTedt = Ne * [0]
dhedt = Ne * [0]
dpedt = 0
dDedt = Ne * [0]
dmedt = Ne * [0]
dTe_walldt = Ne * [0]

moe_m = (Ne - 2) * [0]

dvedt = (Ne + 1) * [0]
mef_dot = (Ne + 1) * [0]
ve = (Ne + 1) * [0]
Fe = (Ne + 1) * [0]
Ke = (Ne + 1) * [0]
De_m = (Ne + 1) * [0]
mue_m = (Ne + 1) * [0]
pe_m = (Ne + 1) * [0]
he_m = (Ne + 1) * [0]
Te_m = (Ne + 1) * [0]
ke_m = (Ne + 1) * [0]
cpe_m = (Ne + 1) * [0]
fe = (Ne + 1) * [0]

from utils import get_state

# ------------------------------------
# input and output states of the fluid
# ------------------------------------
g = 9.81
R = 188.9
gamma = 1.33
cp = 846
cv = 657
roughness_ss = 0.5 * 10 ** (-6)
# -------------------------------------
# phsycial properties of materials used
# --------------------------------------
# Stainless steel (ss)
k_ss = 16.3  # W/m.K
c_ss = 500  # J/Kg.K
D_ss = 7990  # Kg/m^3
mu_ss = 0.006  # Kg/m.s

D_cu = 8850
c_cu = 385
mu_cu = 0.0035
# ----------------------------
# Flash tank
# -------------------------

Lft = 0.29
dft = 0.1016
Vft = np.pi / 4 * dft**2 * Lft
Aft = np.pi * dft * Lft


Cd_hp = 0.0028

