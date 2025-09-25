import numpy as np

# # --------------------------------
# # buffer 1
# # -------------------------------
# Nbuff1 = 1
D_cu = 8850
c_cu = 385
mu_cu = 0.0035
ACO2_orifice = np.pi / 4 * (12.3 * 10 ** (-3)) ** 2
Dw = 1000
muw = 8.9 * 10 ** (-4)
kw = 0.6
cpw = 4184
Dw = 1000

dbuff1 = 15.875 * 10 ** (-3)
Abuff1_orifice = np.pi / 4 * (dbuff1) ** 2
Vbuff1_w_t = 0.4 * 10 ** (-3)
Lbuff1 = 4 * Vbuff1_w_t / (np.pi * dbuff1**2)
Abuff1_t = np.pi * dbuff1 * Lbuff1
mbuff1_t = (np.pi / 4 * (dbuff1 + 0.0015) ** 2 * Lbuff1 - Vbuff1_w_t) * D_cu
Vbuff1_t = 1.8 * 10 ** (-3)

mbuff1_wall = 0
Vbuff1 = 0
Vbuff1_w = 0
Abuff1 = 0
Abuff1_orifibuff1e = 0
dhbuff1 = 0
dxbuff1 = 0
mbuff1 = 0
mubuff1_l = 0
mubuff1_v = 0
mbuff1_w = 0


Vbuff1 = Vbuff1_t
Abuff1_orifibuff1e = Abuff1_orifice
dhbuff1 = 15.875 * 10 ** (-3)
Abuff1 = Abuff1_t
dxbuff1 = Lbuff1
mbuff1_wall = mbuff1_t
Vbuff1_w = Vbuff1_w_t
mbuff1_w = Dw * Vbuff1_w

Ubuff1 = 0
Ubuff1_w = 0

Qbuff1_conv = 0
Qbuff1_w_conv = 0


Dbuff1 = 0
mbuff1 = 0
pbuff1 = 0
Tbuff1 = 0
Tbuff1_wall = 0
Xbuff1 = 0
sbuff1 = 0
hbuff1 = 0
kbuff1 = 0
dpdT_nubuff1 = 0
dDdp_hbuff1 = 0
dDdh_pbuff1 = 0
mubuff1 = 0
cpbuff1 = 0
cvbuff1 = 0
nubuff1 = 0
vbuff1_n = 0
Tbuff1_w = 0
hbuff1_v = 0

dTbuff1dt = 0
dhbuff1dt = 0
dpbuff1dt = 0
dTbuff1_wdt = 0
dDbuff1dt = 0
dmbuff1dt = 0
dTbuff1_walldt = 0



