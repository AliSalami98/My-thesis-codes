import numpy as np

# # --------------------------------
# # buffer 1
# # -------------------------------
# Nbuff2 = 1
D_cu = 8850
c_cu = 385
mu_cu = 0.0035
ACO2_orifice = np.pi / 4 * (12.3 * 10 ** (-3)) ** 2
Dw = 1000
muw = 8.9 * 10 ** (-4)
kw = 0.6
cpw = 4184
Dw = 1000

dbuff2 = 15.875 * 10 ** (-3)
Abuff2_orifice = np.pi / 4 * (dbuff2) ** 2
Vbuff2_w_t = 0.4 * 10 ** (-3)
Lbuff2 = 4 * Vbuff2_w_t / (np.pi * dbuff2**2)
Abuff2_t = np.pi * dbuff2 * Lbuff2
mbuff2_t = (np.pi / 4 * (dbuff2 + 0.0015) ** 2 * Lbuff2 - Vbuff2_w_t) * D_cu
Vbuff2_t = 1.8 * 10 ** (-3)

mbuff2_wall = 0
Vbuff2 = 0
Vbuff2_w = 0
Abuff2 = 0
Abuff2_orifibuff2e = 0
dhbuff2 = 0
dxbuff2 = 0
mbuff2 = 0
mubuff2_l = 0
mubuff2_v = 0
mbuff2_w = 0


Vbuff2 = Vbuff2_t
Abuff2_orifibuff2e = Abuff2_orifice
dhbuff2 = 15.875 * 10 ** (-3)
Abuff2 = Abuff2_t
dxbuff2 = Lbuff2
mbuff2_wall = mbuff2_t
Vbuff2_w = Vbuff2_w_t
mbuff2_w = Dw * Vbuff2_w

Ubuff2 = 0
Ubuff2_w = 0

Qbuff2_conv = 0
Qbuff2_w_conv = 0


Dbuff2 = 0
mbuff2 = 0
pbuff2 = 0
Tbuff2 = 0
Tbuff2_wall = 0
Xbuff2 = 0
sbuff2 = 0
hbuff2 = 0
kbuff2 = 0
dpdT_nubuff2 = 0
dDdp_hbuff2 = 0
dDdh_pbuff2 = 0
mubuff2 = 0
cpbuff2 = 0
cvbuff2 = 0
nubuff2 = 0
vbuff2_n = 0
Tbuff2_w = 0
hbuff2_v = 0

dTbuff2dt = 0
dhbuff2dt = 0
dpbuff2dt = 0
dTbuff2_wdt = 0
dDbuff2dt = 0
dmbuff2dt = 0
dTbuff2_walldt = 0



