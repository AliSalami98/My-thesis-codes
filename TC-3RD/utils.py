import numpy as np
from scipy.optimize import fsolve
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP

# Create an AbstractState object using the HEOS backend and CO2
state = AbstractState("HEOS", "CO2")
def get_state(input_type, input1, input2):
    state = AbstractState("HEOS", "CO2")
    state.update(input_type, input1, input2)
    return state

T0 = 273.15
p0 = 101325
Air_state = AbstractState("HEOS", "Air")
Air_state.update(CP.PT_INPUTS, p0, T0)
mu0 = Air_state.viscosity()
k0 = Air_state.conductivity()
h0 = Air_state.hmass()
s0 = Air_state.smass()
C = 240 #Sutherland constant

def solve_f(eq,e,d,Re):
	if Re < 2300:
		f = 64/Re
	else:
		f0 = 1/(1.8*np.log10((e/(3.7*d))**1.11 + 6.9/Re)**2)  #Haaland
		fsol = fsolve(eq, f0, args=(e,d,Re))
		f = fsol[0]
	return f

def eq_f(f,e,d,Re):
    return 1/np.sqrt(f) + 2*np.log10(e/(3.71*d) + 2.51/(Re*np.sqrt(f)))  #Colebrook-white

def eq_epsilon(epsilon,NTU):
	return epsilon - (1 - np.exp(-NTU*(1-epsilon)))/(1 - epsilon*np.exp(-NTU*(1-epsilon)))

def position_displacer(r_cra, l_rod , theta):
    return r_cra*(1 - np.cos(theta)) + l_rod*(1 - np.sqrt(1-(r_cra/l_rod * np.sin(theta))**2))

def velocity_displacer(r_cra, omega, l_rod, theta):
    return r_cra*omega*np.sin(theta)*(1 + (r_cra/l_rod * np.cos(theta))/(np.sqrt(1-(r_cra/l_rod * np.sin(theta))**2)))

def acceleration_displacer(r_cra, omega, l_rod, theta):
	return omega**2*(r_cra*np.cos(theta) + r_cra**2/l_rod *(np.cos(2*theta)/(np.sqrt(1-(r_cra/l_rod * np.sin(theta))**2))+ (r_cra/l_rod *np.sin(theta)*np.cos(theta))**2/(np.sqrt(1-(r_cra/l_rod * np.sin(theta))**2))**3))

def leakage_flow(r_cyl, r_dis, mu, rho, pe, pc, l_dis, vd):
    return 2*np.pi*rho/np.log(r_cyl/r_dis) * (1/4 * (r_cyl**2 - r_dis**2) - 1/2 * np.log(r_cyl/r_dis)*r_dis**2)*np.abs(vd) - np.pi*rho/(8*mu*l_dis) * (r_cyl**4 - r_dis**4 - (r_cyl**2 - r_dis**2)**2/np.log(r_cyl/r_dis)) * (pe- pc)

def exp_area(r_dis, xd):
    return np.pi*r_dis**2 + 2*np.pi*r_dis*xd

def exp_volume(r_dis, xd, Ve_min):
    return np.pi*r_dis**2*xd + Ve_min

def exp_volume_dot(r_dis, vd):
    return np.pi*r_dis**2*vd

def comp_area(l_str,r_dis, xd,xc_min):
    return 2*np.pi*r_dis*(l_str - xd + xc_min)

def comp_volume(r_dis, l_str,r_shaft, xd,Vc_min):
    return np.pi*(r_dis**2 - r_shaft**2)*(l_str - xd) + Vc_min

def comp_volume_dot(r_dis,r_shaft, vd):
    return -np.pi*(r_dis**2 - r_shaft**2)*vd

def heat_transfer(A,U, Twall, T):
    return A*U*(Twall - T)

def heat_shuttle(l_str, k_ce, Te, Tc, r_ce, l_dis,r_dis):
    return (np.pi*l_str**2*k_ce*r_dis)/(l_dis*r_ce) * (Te - Tc)

def heat_conduction(k, A, L, T_ext, T_int):
    return k*A* (T_ext - T_int)/L

def friction_HX(D, mu, dh, v, roughness):
	Re = v*dh*D/mu
	# K = (1 - min(A1,A2)/max(A1,A2))**2
	if Re < 2*10**3:
		return 64/Re
	else:
		return 0.11*(roughness/dh + 68/Re)**0.25
	
def friction_reg(D, mu, dh, v):
	Re = v*dh*D/mu
	return 129/Re + 2.91/Re**(0.103)

def U_CO2_kh(mu, cp,k,dh, dx, v, D, muw, char):
	Pr = mu*cp/k
	Re = v*D*dh/mu
	if Re < 3000:
		Nu = 1.86*(Re*Pr*dh/dx)**(0.333)*(mu/muw)**(0.14)  #3.16  #
	else:
		# if char == 'h':
		# 	n = 0.4
		# else:
		# 	n = 0.3
		
		# Nu =  0.023*Re**0.8*(Pr**n)  #Dittus-Boelter correlation (mostly when Re> 4000)
		if 3050 < Re < 240000:
			fd = 0.351*Re**(-0.255)
		else:
			fd = 0.118*Re**(-0.165)
		Nu = ((fd/8)*(Re - 1000)*Pr)/(1+ 12.7 * (fd/8)**(0.5) * (Pr**(2/3) - 1)) #
	return k*Nu/dh

def U_CO2_r(mu, cp,k,dh,v, D, phi):
    Re = v*dh*D/mu
    Pr = mu*cp/k
    Nu = (1+0.99*(Re*Pr)**0.66)*phi**1.79  #or Nu = 0.51 + 0.4*Re**0.66 (Gedeon and wood))
    return k*Nu/dh

def U_water(mu, cp,k,dh, mdot, A):
    Pr = mu*cp/k
    Re = mdot*dh/(A*mu)
    Nu =  0.023*Re**0.8*(Pr**(0.3))
    return k*Nu/dh

def U_CO2_ce(mu, cp,k,dh,vd, mdot, A, pc,pe,pin,pout,D):
	vc = np.abs(mdot)/(A*D)
	vd = max(np.abs(vd),0.0001)
	if pc < pin:
		v = vd + vd**(-0.4)*vc**(1.4)
		Re = D*v*dh/mu
		Pr = mu*cp/k
		Nu = 0.08*Re**0.9*Pr**0.6
	elif pc > pout:
		v = vd + vd**(0.8)*vc**(0.2)
		Re = D*v*dh/mu
		Pr = mu*cp/k
		Nu = 0.08*Re**0.8*Pr**0.6
	elif pin <= pc <= pout and pc >= pe:
		v = vd
		Re = D*v*dh/mu
		Pr = mu*cp/k
		Nu = 0.08*Re**0.8*Pr**0.6
	elif pin <= pc <= pout and pc < pe:
		v = vd
		Re = D*v*dh/mu
		Pr = mu*cp/k
		Nu = 0.12*Re**0.8*Pr**0.6
	else:
		Nu = 100
	return k*Nu/dh

def orifice_flow(Cd,A1, A2,  D1,D2, gamma, p1, p2):
	Du = (D1 + D2)/2
	A = (A1+A2)/2
	if p1> p2:
		pu = p1
		pd = p2
		C1 = 2*gamma/(gamma-1)
		C2 = gamma*(2/(gamma+1))**(gamma+1)/(gamma-1)
		pcr = (2/(gamma +1))**(gamma/(gamma-1))
		# return A*Cd*np.sqrt(Du*(pu - pd))
		if (pd/pu <= pcr):
			return A*Cd*np.sqrt(Du*pu*C2)
		else:
			return A*Cd*(pd/pu)**(1/gamma)*np.sqrt(C1*Du*pu*(1-(pd/pu)**((gamma-1)/gamma)))
	else:
		pu = p2
		pd = p1
		C1 = 2*gamma/(gamma-1)
		C2 = gamma*(2/(gamma+1))**(gamma+1)/(gamma-1)
		pcr = (2/(gamma +1))**(gamma/(gamma-1))
		# return -A*Cd*np.sqrt(Du*(pu - pd))
		if (pd/pu <= pcr):
			return -A*Cd*np.sqrt(Du*pu*C2)
		else:
			return -A*Cd*(pd/pu)**(1/gamma)*np.sqrt(C1*Du*pu*(1-(pd/pu)**((gamma-1)/gamma)))

def orifice_flow2(Cd, A, Du, Dd, pu, pd):
	D = (Du+Dd)/2
	if (pu > pd):
		return Cd*A*np.sqrt(2*D*(pu - pd))
	else:
		return -Cd*A*np.sqrt(2*D*(pd - pu))

def valve_flow(Cd,A, Du, gamma, pu, pd):
	C1 = 2*gamma/(gamma-1)
	C2 = gamma*(2/(gamma+1))**(gamma+1)/(gamma-1)
	pcr = (2/(gamma +1))**(gamma/(gamma-1))
	if pu > pd:
		if (pd/pu <= pcr):
			return A*Cd*np.sqrt(Du*pu*C2)
		else:
			return A*Cd*(pd/pu)**(1/gamma)*np.sqrt(C1*Du*pu*(1-(pd/pu)**((gamma-1)/gamma)))
	else:
		return 0

def valve_flow2(Cd,A, Du, pu, pd):
	if (pu > pd+ 0.5*10**(5)):
		return Cd*A*np.sqrt(2*Du*(pu - pd))
	else:
		return 0

def valve_flow_io(Cd,A, hu, pu, pd, cp, gamma, R):
	if (pu > pd + 0.5*10**(5)):
		hd_is = hu*(pd/pu)**((gamma-1)/gamma)
		Dd_is = cp*pd/(R*hd_is)
		vout = np.sqrt(2*(hu - hd_is))
		return Cd*Dd_is*A*vout
	else:
		return 0	

def step(x):
	if x<= 0:
		return 0
	else:
		return 1
