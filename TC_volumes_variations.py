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
# -------------------------
nreg = 10
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
A4= np.pi*r4**2
lkr1 = 0.026083
lkr2 = 0.0055
r2 = 0.002
Akr = 17*2*np.pi*(r2*lkr1 + r4*lkr2)
Vkr = 17*np.pi*(r2**2*lkr1 + r4**2*lkr2) #2.1*10**(-5)
#------------------------------------
# Cooler CV
# ----------------------------------
Vcooler = 7.823*10**(-4)
nk = 1
mcooler = 2.11
xc_min = 0.0003
lk =  0.087363
Vk = 17*(np.pi*r2**2*lk)  #3.2*10**(-5)- np.pi*(r_cyl**2 - r_shaft**2)*xc_min #
Ak = 17*(2*np.pi*r2*lk) #2*Vk/r2 #
# lk = Ak/(17*2*np.pi*r2)
dh_k = 2*r2
A2 = np.pi*r2**2          # area of the orifice in the compression part
#-----------------------------------
# Compression CV
# -----------------------------------
r_valve = 0.00575       # radius of the inlet valve
A_valve = 0.000132          # area of the orifice of the inlet and outlet valve
Vcv = 1.966*10**(-5)  # Volume of check valves
Vcyl_c = 0.962*Vcyl_e
Vc_min = 3.2*10**(-5) - Vk + Vcv  # Vc_max - np.pi*(r_dis**2 - r_shaft**2)*l_str + Vcv
Vc_max = np.pi*(r_dis**2 - r_shaft**2)*(l_str) + Vc_min
theta0 = (0, 2 *np.pi)
a_theta = np.linspace(0, 720 * np.pi/180, 720)
omega = 150
a_Ve = []
a_Vc = []
a_Vt = []

for i in range(len(a_theta)):
    theta = a_theta[i]
    omegas= omega*2*np.pi/60 
    xd = r_cra*(1 - np.cos(theta)) + l_rod*(1 - np.sqrt(1-(r_cra/l_rod * np.sin(theta))**2))
    vd = r_cra*omegas*np.sin(theta)*(1 + (r_cra/l_rod * np.cos(theta))/(np.sqrt(1-(r_cra/l_rod * np.sin(theta))**2)))
    a_Ve.append(np.pi*r_dis**2*xd + Ve_min)
    Ve_dot = np.pi*r_dis**2*vd 
    a_Vc.append(np.pi*(r_dis**2 - r_shaft**2)*(l_str - xd) + Vc_min)
    Vc_dot = -np.pi*(r_dis**2 - r_shaft**2)*vd 

    a_Vt.append(a_Vc[i] + a_Ve[i])


# Convert volumes from m³ to cm³
a_Ve_cm3 = np.array(a_Ve) * 1e6
a_Vc_cm3 = np.array(a_Vc) * 1e6
a_Vt_cm3 =  np.array(a_Vt) * 1e6
theta_deg = np.linspace(0, 720, 720)

# Plot
plt.plot(theta_deg, a_Vc_cm3, color='blue', linewidth=2, label="Compression space")
plt.plot(theta_deg, a_Ve_cm3, color='red', linewidth=2, label="Expansion space")
plt.plot(theta_deg, a_Vt_cm3, color='gray', linewidth=1.5, label="Total working space")

plt.xlabel(r"Crank angle $\theta_\text{m}$ [°]", fontsize=14)
plt.ylabel(r"Volume [cm$^3$]", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.title("Expansion and Compression Volumes vs Crank Angle", fontsize=14)
# plt.legend(fontsize=12, loc='best')
plt.tight_layout()
plt.ylim([0, 600])
save_path = "C:/Users/ali.salame/Desktop/plots/Thesis figs/TC_data/TC_volumes_variations.eps"
plt.savefig(save_path, format='eps', dpi=300)

plt.show()
