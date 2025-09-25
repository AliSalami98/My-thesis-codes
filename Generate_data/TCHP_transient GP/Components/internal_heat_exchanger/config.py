import numpy as np

# # --------------------------------
# # ihx
# # -------------------------------
D_cu = 8850
c_cu = 385
mu_cu = 0.0035
ACO2_orifice = np.pi / 4 * (12.3 * 10 ** (-3)) ** 2

Nihx = 4

dihx1 = 12.3 * 10 ** (-3)
Aihx_orifice = np.pi / 4 * (dihx1) ** 2
Vihx2_t = 2.1 * 10 ** (-3)
Lihx2 = 4 * Vihx2_t / (np.pi * dihx1**2)  # 1.683
Vihx1_t = 0.2 * 10 ** (-3)
Aihx1_t = np.pi * dihx1 * Lihx2
Lihx1 = Vihx1_t / Aihx1_t
mihx_t = (np.pi / 4 * (dihx1 + 0.0015) ** 2 * Lihx2 - Vihx2_t) * D_cu


mihx_wall = 0
Vihx1 = 0
Vw = 0
Aihx1 = 0
Aihx1_orifice = 0
dhihx1 = 0
dxihx1 = 0
mihx1 = 0
muihx1_l = 0
muihx1_v = 0

Aihx1_m = 0
dxihx1_m = 0
mihx1_m = 0
dhihx1_m = 0
Vihx1_m = 0

Vihx1 = Vihx1_t
Aihx1_orifice = ACO2_orifice
dhihx1 = 12.3 * 10 ** (-3)
Aihx1 = Aihx1_t
dxihx1 = Lihx1 / Nihx
mihx_wall = mihx_t

Uihx1 = 0
Qihx1_conv = 0

Dihx1 = 0
mihx1 = 0
pihx1 = 0
Tihx1 = 0
Tihx_wall = 0
Xihx1 = 0
sihx1 = 0
hihx1 = 0
kihx1 = 0
dpdT_nuihx1 = 0
dDdp_hihx1 = 0
dDdh_pihx1 = 0
muihx1 = 0
cpihx1 = 0
cvihx1 = 0
nuihx1 = 0
vihx1_n = 0

dTihx1dt = 0
dhihx1dt = 0
dpihx1dt = 0
dDihx1dt = 0
dmihx1dt = 0
dTihx_walldt = 0

dvihx1dt = 0
mihx1f_dot = 0
vihx1 = 0
Fihx1 = 0
Kihx1 = 0
Dihx1_m = 0
muihx1_m = 0
pihx1_m = 0
hihx1_m = 0
Tihx1_m = 0
kihx1_m = 0
cpihx1_m = 0
fihx1 = 0

# 2


mihx2_wall = 0
Vihx2 = 0
Vw = 0
Aihx2 = 0
Aihx2_orifice = 0
dhihx2 = 0
dxihx2 = 0
mihx2 = 0
muihx2_l = 0
muihx2_v = 0

Aihx2_m = 0
dxihx2_m = 0
mihx2_m = 0
dhihx2_m = 0
Vihx2_m = 0
for i in range(Nihx):
    Vihx2 = Vihx2_t
    Aihx2_orifice = Aihx_orifice
    dhihx2 = 12.3 * 10 ** (-3)
    Aihx2 = Aihx1_t
    dxihx2 = Lihx2 / Nihx

Uihx2 = 0
Qihx2_conv = 0

Dihx2 = 0
mihx2 = 0
pihx2 = 0
Tihx2 = 0
Tihx2_wall = 0
Xihx2 = 0
sihx2 = 0
hihx2 = 0
kihx2 = 0
dpdT_nuihx2 = 0
dDdp_hihx2 = 0
dDdh_pihx2 = 0
muihx2 = 0
cpihx2 = 0
cvihx2 = 0
nuihx2 = 0
vihx2_n = 0
Tw = 0

dTihx2dt = 0
dhihx2dt = 0
dpihx2dt = 0
dDihx2dt = 0
dmihx2dt = 0

dvihx2dt = 0
mihx2f_dot = 0
vihx2 = 0
Fihx2 = 0
Kihx2 = 0
Dihx2_m = 0
muihx2_m = 0
pihx2_m = 0
hihx2_m = 0
Tihx2_m = 0
kihx2_m = 0
cpihx2_m = 0
fihx2 = 0