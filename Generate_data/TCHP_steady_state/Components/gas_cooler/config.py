import numpy as np

# ----------------------------------------
# gas cooler
# ----------------------------------------
Nc = 10

Aw_orifice = np.pi / 4 * (20 * 10 ** (-3)) ** 2
dhc_w = 20 * 10 ** (-3)
muw = 8.9 * 10 ** (-4)
kw = 0.6
cpw = 4184
Dw = 1000

Np_c = 56
ACO2_orifice = np.pi / 4 * (12.3 * 10 ** (-3)) ** 2
Vc_t = (Np_c - 1) * 0.03 * 10 ** (-3)
Ac_t = (Np_c - 2) * 0.0193
mc_t = (Np_c) * 0.152
Lc = Vc_t/ACO2_orifice #125 * 10 ** (-3)

mc_wall = Nc * [0]
Vc = Nc * [0]
Vc_w = Nc * [0]
Ac = Nc * [0]
Ac_orifice = Nc * [0]
dhc = Nc * [0]
dxc = Nc * [0]
mc = Nc * [0]
muc_l = Nc * [0]
muc_v = Nc * [0]
mc_w = Nc * [0]

Ac_m = (Nc + 1) * [0]
dxc_m = (Nc + 1) * [0]
mc_m = (Nc + 1) * [0]
dhc_m = (Nc + 1) * [0]
Vc_m = (Nc + 1) * [0]
for i in range(Nc):
    Vc[i] = Vc_t / (Nc)
    Ac_orifice[i] = ACO2_orifice
    dhc[i] = 12.3 * 10 ** (-3)
    Ac[i] = Ac_t / (Nc)
    dxc[i] = Lc / Nc
    mc_wall[i] = mc_t / (Nc)
    Vc_w[i] = Vc_t / (Nc)
    mc_w[i] = Dw * Vc_w[i]

# for i in range(Nc + 1):
#     Ac_m[i] = (Ac[i] + Ac[i + 1]) / 2
#     mc_m[i] = (mc[i] + mc[i + 1]) / 2
#     Vc_m[i] = (Vc[i] + Vc[i + 1]) / 2
#     dhc_m[i] = (dhc[i] + dhc[i + 1]) / 2
#     dxc_m[i] = (dxc[i] + dxc[i + 1]) / 2

Uc = Nc * [0]
Qc_conv = Nc * [0]
Qc_w_conv = Nc * [0]

Dc = Nc * [0]
mc = Nc * [0]
pc = 0
Tc = Nc * [0]
Tc_wall = Nc * [0]
Xc = Nc * [0]
sc = Nc * [0]
hc = Nc * [0]
kc = Nc * [0]
dpdT_nuc = Nc * [0]
dDdp_hc = Nc * [0]
dDdh_pc = Nc * [0]
muc = Nc * [0]
cpc = Nc * [0]
cvc = Nc * [0]
nuc = Nc * [0]
vc_n = Nc * [0]
Tc_w = Nc * [0]

dTcdt = Nc * [0]
dhcdt = Nc * [0]
dpcdt = 0
dTc_wdt = Nc * [0]
dDcdt = Nc * [0]
dmcdt = Nc * [0]
dTc_walldt = Nc * [0]

dvcdt = (Nc + 1) * [0]
mcf_dot = (Nc + 1) * [0]
vc = (Nc + 1) * [0]
Fc = (Nc + 1) * [0]
Kc = (Nc + 1) * [0]
Dc_m = (Nc + 1) * [0]
muc_m = (Nc + 1) * [0]
pc_m = (Nc + 1) * [0]
hc_m = (Nc + 1) * [0]
Tc_m = (Nc + 1) * [0]
kc_m = (Nc + 1) * [0]
cpc_m = (Nc + 1) * [0]
fc = (Nc + 1) * [0]

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
t0 = (0, 750)


