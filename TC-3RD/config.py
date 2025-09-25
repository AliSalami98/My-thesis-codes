import numpy as np

from utils import (
	T0,
	p0,
	h0,
	s0,
	mu0,
	k0,
	C,
)

# -----------------------------------------------------
# Geometries of the compressor (SI unit is meter)
# -----------------------------------------------------
r_dis = 0.04215   #radius of the displacer
l_dis = 0.22165   # length of the displacer
r_cra = 0.0268         # radius of the crank
l_rod = 0.12           # length of the rod
r_rod = 0.0075         # radius of the rod
l_str = 2*r_cra        # length of stroke
t_cyl = 0.018             # heater wall thickness
r_cyl = 0.0434            # internal radius of the cylinder
r_shaft = 0.009
A_dis = np.pi*r_dis**2
A_rod = np.pi*r_rod**2
r_ce = r_cyl - r_dis
# ------------------------------------
# input and output states of the fluid
# ------------------------------------
g = 9.81
R= 188.9
gamma = 1.33
cp = 846
cv = 657
roughness_ss = 0.5*10**(-6)
roughness_al = 0.1*10**(-6)
#-------------------------------------
# phsycial properties of materials used
#--------------------------------------
# Stainless steel (ss)
k_ss = 16.3     # W/m.K
c_ss = 500      # J/Kg.K
D_ss = 7990     # Kg/m^3
mu_ss = 0.006   # Kg/m.s
# Aluminum (Al)
k_al = 167            # W/m.K
c_al = 900            # J/Kg.K
D_al = 2700           # Kg/m^3
mu_al = 0.012         # Kg/m.s
#-----------------------------------
# Expansion
#------------------------------------
Ace = np.pi*(r_cyl**2 - (r_cyl - 20*10**(-6))**2)
Vcyl_e = 36.2*10**(-5)
Ve_min = (Vcyl_e - np.pi*r_dis**2*l_str)#/1.962
Ve_max = np.pi*r_dis**2*l_str + Ve_min
xe_min = Ve_min/(np.pi*r_dis**2)
Ae_min = 2*np.pi*r_dis*xe_min
re_orifice = 0.0095       # radius of the orifice in the hot section
Ae_orifice = np.pi*re_orifice**2 + 24*np.pi*(0.0013/2)**2
#-----------------------------------
# Heater 
#----_-------------------------------
Vheater = 7.04*10**(-4) 
nh = 1
t_heater = 0.02
mheater = 5.56        # in Kg
r_sph_ext = 0.0645
r_sph_int = 0.0441
Ah_sph_int = 2*np.pi*r_sph_int**2
Ah_sph_ext = 2*np.pi*r_sph_ext**2

lh = 0.055
rh_int = 0.0434
rh_ext = 0.0441
dh_h = 2*(rh_ext - rh_int)

Ah_int = 2*np.pi*rh_int*lh
Ah_ext = 2*np.pi*rh_ext*lh

Ah = (Ah_int + Ah_ext)/2

Ah_orifice= np.pi*(rh_ext**2 - rh_int**2)
Vh = 10**(-5)  #2/3*np.pi*(rr_int**3 - r_cyl**3) + np.pi*(0.0012**2)*0.055537
# print(2/3*np.pi*(rh_ext**3 - rh_int**3) + np.pi* (lh - rh_int)*(rh_ext**2 - rh_int**2))
# ----------------------------------
# H-R
# ---------------------------------
Vheater_dv = 2.05*10**(-4)
nhr = 1
mheater_dv = 3.29
rhr_int = 0.05445
rhr_ext = 0.05565
Vhr = 5.8*10**(-5)  #np.pi*(rhr_ext**2 - rhr_int**2)*lhr #
lhr = Vhr/(np.pi*(rhr_ext**2 - rhr_int**2))  #0.05537
# print(lhr)
dhr_h = 2*(rhr_ext - rhr_int)

Ahr_int = 2*np.pi*rhr_int*lhr
Ahr_ext = 2*np.pi*rhr_ext*lhr
Ahr = Ahr_int + Ahr_ext

Ahr_orifice = np.pi*(rhr_ext**2 - rhr_int**2)
# ----------------------------
# Regenerator
# ------------------------
nreg = 8
mreg = 0.58

lreg = 0.0185            # length of the regenerator
rr_int = 0.0867/2         
rr_ext = 0.1341/2  
Vreg = Vr = 7.6*10**(-5)  #0.5*np.pi*(rr_ext**2 - rr_int**2)*lreg

Ar = Vr/lreg
phi = 0.5

dp = 0.06*10**(-3)
Ap = dp**2
Np = Vr/(lreg*Ap)
pp = 4*dp
p_total = Np*pp

Arg = 4*Ar*lreg/dp
dh_r = 4*Vr/Arg * phi/(1-phi)
# ----------------------------------
# K-R
# ---------------------------------
Vcooler_dv = 2.446*10**(-4)
nkr = 1
mcooler_dv = 1.93
r4 = 0.0075
A4= 17 * np.pi*r4**2
lkr1 = 0.026083
lkr2 = 0.0055
r2 = 0.002
Akr = 17*2*np.pi*(r2*lkr1 + r4*lkr2)
Vkr = 17*np.pi*(r2**2*lkr1 + r4**2*lkr2) #2.1*10**(-5)
#------------------------------------
# Cooler CV
# ----------------------------------
Vcooler = 7.823*10**(-4)
nk = 4
mcooler = 2.11
xc_min = 0.0003
lk =  0.087363
Vk = 17*(np.pi*r2**2*lk)  #3.2*10**(-5)- np.pi*(r_cyl**2 - r_shaft**2)*xc_min #
Ak = 17*(2*np.pi*r2*lk) #2*Vk/r2 #
# lk = Ak/(17*2*np.pi*r2)
dh_k = 2*r2
A2 = 17 * np.pi*r2**2          # area of the orifice in the compression part
#-----------------------------------
# Compression CV
# -----------------------------------
r_valve = 0.00575       # radius of the inlet valve
A_valve = 0.000132          # area of the orifice of the inlet and outlet valve
Vcv = 1.966*10**(-5)  # Volume of check valves
Vcyl_c = 0.962*Vcyl_e
Vc_min = (3.2*10**(-5) - Vk + Vcv)  # Vc_max - np.pi*(r_dis**2 - r_shaft**2)*l_str + Vcv
Vc_max = np.pi*(r_dis**2 - r_shaft**2)*(l_str) + Vc_min
md = 2

# ------------------------
# physical boundaries
# ------------------------
Tmin = 280
Tmax = 1400

t_ck = 0.004
tw_k= 0.006
Vw = 1.2375*10**(-4) #4*(0.5*np.pi*5.5**2 + 2*5.5*(8.4-5.5))*np.pi*124*10**(-9)
Aw_orifice = 7.95*10**(-5) #(0.5*np.pi*5.5**2 + 2*5.5*(8.4-5.5))*10**(-6)
Aw_k = 4*(2*np.pi*5.5 + 2*(8.4-5.5))*np.pi*114.6*10**(-6)
muw = 8.9*10**(-4)
kw = 0.6
cpw = 4184
mw_dot = 0.12
dh_w = 6.4*10**(-3)

Cd_ce = 1
Cd_io = 1

N = nk + nkr + nreg + nhr + nh + 2

Awall = N*[0]
mwall = N*[0]
Vwall = N*[0]

Vn = N*[0]
An = N*[0]
A = N*[0]
dh = N*[0]
dx = N*[0]
mn = N*[0]

Am = (N-1) *[0]
dxm = (N - 1)* [0]
mm = (N-1) *[0]
dhm = (N-1) *[0]
Vm = (N-1) *[0]
for i in range(N):
	if i ==0:
		Vn[i] = Vc_max
		A[i] = A_dis
		dh[i] = 2 * r_cyl
		An[i] = 2 * np.pi *r_dis * (l_str + xc_min)
		dx[i] = l_str/2
		Vwall[i] = Vc_max/2
		mwall[i] = Vwall[i] * D_ss
		Awall[i] = Vwall[i]/dx[i]
	if (1 <= i < nk+1):
		Vn[i] = Vk/nk
		A[i] = A2
		dh[i] = dh_k
		An[i] = Ak/nk
		dx[i] = lk/nk
		Vwall[i] = Vcooler/nk
		mwall[i] = Vwall[i] * D_al
		Awall[i] = Vwall[i]/dx[i]
	elif (nk+1 <= i < nk + nkr+1):
		Vn[i] = Vkr/nkr
		A[i] = A4
		dh[i] = dh_k
		An[i] = Akr/nkr
		dx[i] = (lkr1 + lkr2)/nkr
		Vwall[i] = Vcooler_dv/nkr
		mwall[i] = Vwall[i] * D_ss
		Awall[i] = Vwall[i]/dx[i]
	elif (nk+ nkr+1 <= i < nk + nkr + nreg+1):
		Vn[i] = Vr/nreg
		A[i] = Ar
		dh[i] = dh_r
		An[i] = Arg/nreg
		dx[i] = lreg/nreg
		Vwall[i] = Vr/nreg
		mwall[i] = Vwall[i] * D_ss
		Awall[i] = Vwall[i]/dx[i]
	elif (nk+ nkr + nreg+1 <= i < nk + nkr + nreg + nhr+1):
		Vn[i] = Vhr/nhr
		A[i] = Ahr_orifice
		dh[i] = dhr_h
		An[i] = Ahr/nhr
		dx[i] = lhr/nhr
		Vwall[i] = Vheater_dv/nhr
		mwall[i] = Vwall[i] * D_ss
		Awall[i] = Vwall[i]/dx[i]
	elif (nk + nkr + nreg + nhr+1 <= i < N-1):
		Vn[i] = Vh/nh
		A[i] = Ah_orifice
		dh[i] = dh_h
		An[i] = Ah/nh
		dx[i] = lh/nh
		Vwall[i] = Vheater/nh
		mwall[i] = Vwall[i] * D_ss
		Awall[i] = Vwall[i]/dx[i]
	elif i == N-1:
		Vn[i] = Ve_min
		A[i] = A_dis
		dh[i] = r_cyl
		An[i] = Ae_min
		dx[i] = l_str/2
		Vwall[i] = Ve_max/2
		mwall[i] = Vwall[i] * D_ss
		Awall[i] = Vwall[i]/dx[i]

for i in range(N-1):
	Am[i] = (A[i] + A[i+1])/2
	mm[i] = (mn[i] + mn[i+1])/2
	Vm[i] = (Vn[i] + Vn[i+1])/2
	dhm[i] = (dh[i] + dh[i+1])/2
	dxm[i] = (dx[i] + dx[i+1])/2

U = N*[0]
Q_conv_int = N*[0]
Q_cond = (N)*[0]
Qs_cond = (N)*[0]
Dn = N*[0]
mn = N*[0]
pn = N*[0]
Tn = N*[0]
Twall = N*[0]

sn = N*[0]
psin = N*[0]
psim = (N - 1) * [0]

hn = N*[0]
kn = N*[0]
dpndTn_nun = N*[0]
mun = N*[0]
cpn = N*[0]
cvn = N*[0]
nun = N*[0]
vn = N * [0]

dTndtheta = N*[0]
dDndtheta = N*[0]
dmndtheta = N*[0]
dTwalldtheta = N*[0]
dTwall_extdtheta = N*[0]

mom = (N - 2)*[0]

dvdtheta = (N - 1) * [0]
mf_dot = (N - 1) * [0]
v = (N - 1) * [0]
F = (N - 1)* [0]
K = (N - 1)* [0]
Dm = (N - 1)* [0]
mum = (N - 1)* [0]
pm = (N - 1) * [0]
hm = (N - 1) * [0]
Tm = (N - 1) * [0]
km = (N - 1) * [0]
cpm = (N - 1) * [0]
f = (N - 1) * [0]

theta0 = (0, 2 *np.pi)

