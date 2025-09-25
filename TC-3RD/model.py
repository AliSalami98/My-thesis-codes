import numpy as np

from utils import (
	CP,
	state,

)

from config import (
	R,
	gamma,
    A,
    A2,
    A_dis,
    A_valve,
	Ace,
	Aw_k,
	Aw_orifice,
    Ae_orifice,
	Ah_sph_ext,
    An,
    Cd_ce,
	Q_cond,
    Cd_io,
    Vc_min,
    Ve_min,
    Vn,
	Vcv,
    c_al,
    c_ss,
	cpw,
	kw,
	k_al,
	k_ss,
    dh,
	xe_min,
	dh_w,
    dx,
    l_rod,
    l_str,
	l_dis,
    mu_al,
    mu_ss,
	muw,
    mwall,
	Awall,
	nh,
    nhr,
    nk,
    nkr,
    nreg,
    N,
    phi,
    r_ce,
    r_cra,
    r_dis,
    r_shaft,
    xc_min,
	tw_k,
	t_ck,
	t_heater,
	mw_dot,
	roughness_al,
	roughness_ss,
	dhm,
	Am,
	Vm,
	mm,
	U,
	Q_conv_int,
	Qs_cond,
	Dn,
	mn,
	pn,
	Tn,
	Twall,
	sn,
	hn,
	kn,
	dpndTn_nun,
	mun,
	cpn,
	cvn,
	nun,
	vn,
	dTndtheta,
	dmndtheta,
	dTwalldtheta,
	dTwall_extdtheta,
	mom,
	dvdtheta,
	mf_dot,
	v,
	F,
	K,
	Dm,
	mum,
	dxm,
	pm,
	hm,
	Tm,
	km,
	cpm,

	f,
)

def model(theta,x, pin, pout, sin, Din, hin, omega, Twall_ext, Tw_in, c, d):
	omegas= omega*2*np.pi/60 
	xd = r_cra*(1 - np.cos(theta)) + l_rod*(1 - np.sqrt(1-(r_cra/l_rod * np.sin(theta))**2))
	vd = r_cra*omegas*np.sin(theta)*(1 + (r_cra/l_rod * np.cos(theta))/(np.sqrt(1-(r_cra/l_rod * np.sin(theta))**2)))
	Vn[-1] = np.pi*r_dis**2*xd + Ve_min 
	Ve_dot = np.pi*r_dis**2*vd 
	Vn[0] = np.pi*(r_dis**2 - r_shaft**2)*(l_str - xd) + Vc_min
	Vc_dot = -np.pi*(r_dis**2 - r_shaft**2)*vd 
	An[0] = 2*np.pi*r_dis*(l_str - xd + xc_min)
	An[-1] = 2*np.pi*r_dis*(xd + xe_min) 
	dx[0] = np.abs(l_str - xd + xc_min)
	dx[-1] = np.abs(xd + xe_min) 
	

	Tk = 0
	Th = 0
	for i in range(N):
		mn[i] = x[i]
		Tn[i] = x[i+N]
		Twall[i] = x[i + 3 * N + 1]
		Dn[i]= mn[i]/Vn[i]
		nun[i] = 1/Dn[i]
		state.update(CP.DmassT_INPUTS, Dn[i], Tn[i])
		pn[i] = state.p()
		hn[i]= state.hmass()
		kn[i] = state.conductivity()
		sn[i] = state.smass()
		mun[i]= state.viscosity()
		cpn[i] = state.cpmass()
		cvn[i] = state.cvmass()
		dpndTn_nun[i] = state.first_partial_deriv(CP.iP, CP.iT, CP.iDmass)
		if 1 <= i < nk +1:
			Tk += Tn[i]
		elif (nk + nkr + nreg + nhr + 1 <= i < N-1):
			Th += Tn[i]
	Tk = Tk/nk
	Th = Th/nh
	# print(theta)
	# -----------------------------------
	# interface filing and velocity 
	# ------------------------------------
	for i in range(N-1):
		v[i] = x[i + 2*N]
		if v[i] >= 0:
			hm[i] = hn[i] 
			pm[i] = pn[i]
			Tm[i] = Tn[i]
			Dm[i] = Dn[i]
			mm[i] = (mn[i] + mn[i+1])/2
			mum[i] = mun[i]
			km[i] = kn[i]
			cpm[i] = cpn[i]
		else:
			hm[i] = hn[i+1] 
			pm[i] = pn[i+1]
			Tm[i] = Tn[i+1]
			Dm[i] = Dn[i+1]
			mm[i] = (mn[i] + mn[i+1])/2
			mum[i] = mun[i+1]
			km[i] = kn[i+1]
			cpm[i] = cpn[i+1]
		# -----------------------------------
		# friction and mass flow rate
		# ---------------------------------
		Re = np.abs(v[i])*dhm[i]*Dm[i]/mum[i]
		Pr = mum[i]*cpm[i]/km[i]
		if i == 0:
			dxm[i] = (l_str - xd + dx[i+1])/2
			if Re < 2*10**3:
				f[i] = 64/Re
			else:
				f[i] = 0.11*(roughness_al/dhm[i] + 68/Re)**0.25
			K[i] = 1.3 + (1- min(A[i], A[i+1])/max(A[i], A[i+1]))**2
		elif (1 <= i < nk+nkr):
			if Re < 2*10**3:
				f[i] = 64/Re
			else:
				f[i] = 0.11*(roughness_ss/dhm[i] + 68/Re)**0.25
			K[i] = 0.3 + (1- min(A[i], A[i+1])/max(A[i], A[i+1]))**2
		elif (nk+nkr <= i < nk + nkr + nreg):
			f[i] = 129/Re + 2.91*Re**(-0.103)
			K[i] = (1- min(A[i], A[i+1])/max(A[i], A[i+1]))**2
		elif (nk + nkr + nreg <= i < N-2):
			if Re < 2*10**3:
				f[i] = 64/Re
			else:
				f[i] = 0.11*(roughness_ss/dhm[i] + 68/Re)**0.25
			K[i] = (1- min(A[i], A[i+1])/max(A[i], A[i+1]))**2
		else:
			dxm[i] = (dx[i] + xd)/2
			if Re < 2*10**3:
				f[i] = 64/Re
			else:
				f[i] = 0.11*(roughness_ss/dhm[i] + 68/Re)**0.25
			K[i] = 1.8 + (1- min(A[i], A[i+1])/max(A[i], A[i+1]))**2

		F[i] = (f[i]/dhm[i] + K[i]/dxm[i]) * dxm[i] * Dm[i] * v[i] * np.abs(v[i])/2
		mf_dot[i] = Cd_ce * Am[i] * Dm[i] * v[i]
		
	# -----------------------------------
	# Nusselt number and heat convection filling
	# ------------------------------------
	Qloss = 0
	Qcooler_zabri_sum = 0
	for i in range(N):
		if i == 0:
			vn[i] = (vd + v[i])/2
		elif (1 <= i < N-1):
			vn[i] = (v[i-1] + v[i])/2
			mom[i-1] = A[i] * Dn[i] * vn[i] *np.abs(vn[i])
		else:
			vn[i] = (v[i-1] + vd)/2

		Re = np.abs(vn[i])*dh[i]*Dn[i]/mun[i]
		Pr = mun[i]*cpn[i]/kn[i]
		if i == 0:
			if Re < 2*10**3:
				Nu = 1.86*(Re*Pr*dh[i]/dx[i])**(0.333)*(mun[i]/mu_al)**(0.14) #3.16
			else:
				Nu = 0.023*Re**0.8*(Pr**(0.3))
		elif 1 <= i < nk+1:
			if Re < 2*10**3:
				Nu = 1.86*(Re*Pr*dh[i]/dx[i])**(0.333)*(mun[i]/mu_al)**(0.14) #3.16
			else:
				Nu =  0.023*Re**0.8*(Pr**(0.3))
		elif (nk+1 <= i < nk + nkr+1):
			if Re < 2*10**3:
				Nu = 1.86*(Re*Pr*dh[i]/dx[i])**(0.333)*(mun[i]/mu_ss)**(0.14) #3.16
			else:
				Nu =  0.023*Re**0.8*(Pr**(0.3))
		elif (nk+ nkr +1<= i < nk + nkr + nreg+1):
			Nu = (1+0.99*(Re*Pr)**0.66)*phi**1.79  #or Nu = 0.51 + 0.4*Re**0.66 (Gedeon and wood))
		else:
			if Re < 2*10**3:
				Nu = 1.86*(Re*Pr*dh[i]/dx[i])**(0.333)*(mun[i]/mu_ss)**(0.14) #3.16
			else:
				Nu =  0.023*Re**0.8*(Pr**(0.4))
		U[i] = kn[i]*Nu/dh[i]
		if i == 0:
			Q_conv_int[i] = An[i] * U[i] * (Tk - Tn[i]) 
			# Q_cond[i] = kn[i] * A[i] * (Tn[i + 1] - Tn[i])/dx[i]
			# Qs_cond[i] = k_ss * Awall[i] * (Twall[i + 1] - Twall[i])/dx[i]
		elif i == N-1:
			Q_conv_int[i] = An[i] * U[i] * (Th - Tn[i])
			# Q_cond[i] = kn[i] * A[i] * (Tn[i - 1] - Tn[i])/dx[i]
			# Qs_cond[i] = k_ss * Awall[i] * (Twall[i - 1] - Twall[i])/dx[i]
		else:
			Q_conv_int[i] = An[i] * U[i] * (Twall[i] - Tn[i])
			# Q_cond[i] = kn[i] * A[i] * (Tn[i + 1] - Tn[i])/dx[i]
			# if i < nk + 1:
			# 	Qs_cond[i] = k_al * Awall[i] * (Twall[i + 1] - Twall[i])/dx[i]
			# else:
			# 	Qs_cond[i] = k_ss * Awall[i] * (Twall[i + 1] - Twall[i])/dx[i]

	# print(theta)

	mint_dot = x[3*N-1]
	mout_dot = x[3*N]

	# state.update(CP.PSmass_INPUTS, pn[0], sin)
	hc_in = hin  #state.hmass()

	za = 0.1
	Q_shuttle = (np.pi*l_str**2*(kn[0]+kn[-1])/2*r_dis)/(l_dis*r_ce) * (Tn[-1] - Tn[0])
	# -----------------------------------------------
	# ODEs
	# -----------------------------------------------
	# if  pn[0] > pout: 
	# 	# state.update(CP.PSmass_INPUTS, pout, sn[0])
	# 	# hc_out = state.hmass()
	# 	# Dc_out = state.rhomass()
	# 	# vout = np.sqrt(2*(hn[0] - hc_out))
	# 	# mout_dot = Cd_io*Dc_out*A_valve*vout
	# 	dmout_dotdtheta =  A_valve/(Vcv) * (A_valve * (pn[0]  - pout) + A_valve * Dn[0] * vd * np.abs(vd) - mout_dot*np.abs(mout_dot/(Dn[0]*A_valve)))
	# else:
	# 	# mout_dot = 0
	# 	dmout_dotdtheta = - 50*mout_dot
	# if pin > pn[0]:
	# 	# state.update(CP.PSmass_INPUTS, pn[0], sin)
	# 	# hc_in= state.hmass()
	# 	# Dc_in = state.rhomass()
	# 	# vin = np.sqrt(2*(hin - hc_in))
	# 	# mint_dot = Cd_io*Dc_in*A_valve*vin
	# 	dmint_dotdtheta =  A_valve/(Vcv) * (A_valve * (pin - pn[0]) - A_valve * Dn[0] * vd * np.abs(vd) - mint_dot*np.abs(mint_dot/(Din*A_valve)))
	# else:
	# 	# mint_dot = 0
	# 	dmint_dotdtheta = - 50*mint_dot

	theta1 = 0
	theta2 = 0
	df1 = (0.1 + 0.9*np.exp(-0.1*(theta - theta1))) #2*(1 - 1/(1 + np.exp(-0.7*10**(-1)*(theta - theta1))) )
	if  pn[0] >= pout:
		# dmout_dotdtheta =  10*A_valve/(Vcv) * (A_valve * (pn[0]  - pout - 0.5*10**5) + A_valve * Dn[0] * vd * np.abs(vd) - mout_dot*np.abs(mout_dot/(Dn[0]*A_valve)))
		if pn[0] >= pout + 0.5*10**5 and c[0]==0:
			dmout_dotdtheta =  A_valve/(Vcv) * (A_valve * (pn[0]  - pout- 0.5*10**5) + A_valve * Dn[0] * vd * np.abs(vd) - mout_dot*np.abs(mout_dot/(Dn[0]*A_valve)))
			c[0] = 1
			theta1 = theta
			df1 = (0.1 + 0.9*np.exp(-0.1*(theta - theta1))) #2*(1 - 1/(1 + np.exp(-0.7*10**(-1)*(theta - theta1))) )
		elif pn[0] >= pout and c[0] == 1:
			dmout_dotdtheta =  A_valve/(Vcv) * (A_valve * df1*(pn[0]  - pout) + A_valve * Dn[0] * vd * np.abs(vd) - mout_dot*np.abs(mout_dot/(Dn[0]*A_valve)))
		else:
			dmout_dotdtheta = - 50 * mout_dot
			c[0] = 0
	else:
		dmout_dotdtheta = - 50 * mout_dot
		c[0] = 0
	df2 = (0.1 + 0.9*np.exp(-0.1*(theta - theta2))) #2*(1 - 1/(1 + np.exp(-10**(-1)*(theta - theta2))))
	if pin >= pn[0]:
		# dmint_dotdtheta =  10*A_valve/(Vcv) * (A_valve * (pin - pn[0] - 0.5*10**5) - A_valve * Dn[0] * vd * np.abs(vd) - mint_dot*np.abs(mint_dot/(Din*A_valve)))
		if pin >= pn[0] + 0.5*10**5 and d[0] == 0:
			dmint_dotdtheta =  A_valve/(Vcv) * (A_valve * (pin - pn[0]- 0.5*10**5) - A_valve * Dn[0] * vd * np.abs(vd) - mint_dot*np.abs(mint_dot/(Din*A_valve)))
			d[0] = 1
			theta2 = theta
			df2 = (0.1 + 0.9*np.exp(-0.1*(theta - theta2))) #2*(1 - 1/(1 + np.exp(-10**(-1)*(theta - theta2))))
		elif pin >= pn[0] and d[0] == 1:
			dmint_dotdtheta = A_valve/(Vcv) * (A_valve *df2* (pin - pn[0]) - A_valve * Dn[0] * vd * np.abs(vd) - mint_dot*np.abs(mint_dot/(Din*A_valve)))
		else:
			dmint_dotdtheta= - 50 * mint_dot
			d[0] = 0
	else:
		dmint_dotdtheta = - 50 * mint_dot
		d[0] = 0

	if pn[0] > pn[-1]:
		Dce = Dn[0]
		hce = hn[0]
	else:
		Dce = Dn[-1]
		hce = hn[-1]

	mce_dot = 0 #Ace*np.sign(pn[0] - pn[-1])*np.sqrt(2*Dce*np.abs(pn[0] - pn[-1]))
	
	# dDndtheta[0] = 1/(omegas*Vn[0]) * ((-mf_dot[0] + mint_dot- mout_dot) - Dn[0]*Vc_dot)
	dmndtheta[0] = 1/(omegas) * ((-mf_dot[0] + mint_dot- mout_dot) - mce_dot)
	dTndtheta[0] = 1/(omegas) * (Q_conv_int[0] + Q_shuttle + Q_cond[0] - Tn[0]*dpndTn_nun[0]*Vc_dot - (hn[0] - Tn[0]*dpndTn_nun[0]*nun[0])*(-mf_dot[0] + mint_dot- mout_dot - mce_dot) - mf_dot[0]*hm[0] + mint_dot*hc_in - mout_dot*hn[0] - mce_dot * hce)/(mn[0]*cvn[0])
	dvdtheta[0] = 1/(mm[0]*omegas) * (Am[0] * (pn[0] - pn[1]) + (Dn[0] * A_dis * vd * np.abs(vd)- mom[0]) - np.abs(mf_dot[0])*v[0] - Am[0]*F[0])
	dTwalldtheta[0] = -(1/omegas) * (-Qs_cond[0] + Q_conv_int[0])/(mwall[0] * c_al) 

	for i in range(1, N-1):
		
		# dDndtheta[i] = 1/(omegas*Vn[i])*(mf_dot[i-1] - mf_dot[i])
		dmndtheta[i] = 1/(omegas)*(mf_dot[i-1] - mf_dot[i])
		if i < nk+1:
			dTndtheta[i] = 1/(omegas) * ((Q_conv_int[i]- Q_conv_int[0]/nk) + Q_cond[i] - Q_cond[i - 1] - (hn[i] - Tn[i]*dpndTn_nun[i]*nun[i])*(mf_dot[i-1] - mf_dot[i]) + mf_dot[i-1]*hm[i-1] - mf_dot[i]*hm[i])/(mn[i]*cvn[i])
			dTwalldtheta[i] = -(1/omegas) * (Q_conv_int[i] - Qs_cond[i] + Qs_cond[i - 1])/(mwall[i] * c_al)
		elif nk + nkr + nreg + nhr + 1 <= i < N-1:
			dTndtheta[i] = 1/(omegas) * ((Q_conv_int[i] - Q_conv_int[-1]/nh) + Q_cond[i] - Q_cond[i - 1]  - (hn[i] - Tn[i]*dpndTn_nun[i]*nun[i])*(mf_dot[i-1] - mf_dot[i]) + mf_dot[i-1]*hm[i-1] - mf_dot[i]*hm[i])/(mn[i]*cvn[i])
			dTwalldtheta[i] = -(1/omegas) * (Q_conv_int[i] - Q_conv_int[-1]/nh - Qs_cond[i] + Qs_cond[i - 1])/(mwall[i] * c_ss)
		else:
			dTndtheta[i] = 1/(omegas) * (Q_conv_int[i] + Q_cond[i] - Q_cond[i - 1]  - (hn[i] - Tn[i]*dpndTn_nun[i]*nun[i])*(mf_dot[i-1] - mf_dot[i]) + mf_dot[i-1]*hm[i-1] - mf_dot[i]*hm[i])/(mn[i]*cvn[i])
			dTwalldtheta[i] = -(1/omegas) * (Q_conv_int[i] - Qs_cond[i] + Qs_cond[i - 1])/(mwall[i] * c_ss)
		if i < N-2:
			dvdtheta[i] = 1/(mm[i]*omegas) * (Am[i] * (pn[i] - pn[i+1]) + (mom[i-1]- mom[i]) - np.abs(mf_dot[i])*v[i]- Am[i]*F[i]) 
	# dDndtheta[N-1] = 1/(omegas*Vn[N-1])*(mf_dot[N-2] - Dn[N-1]*Ve_dot)
	dmndtheta[N-1] = 1/(omegas)*(mf_dot[N-2] + mce_dot)
	dTndtheta[N-1] = 1/(omegas) * (Q_conv_int[N-1] - Q_shuttle + Q_cond[N - 1] -Tn[N-1]*dpndTn_nun[N-1]*Ve_dot - (hn[N-1] - Tn[N-1]*dpndTn_nun[N-1]*nun[N-1]) * (mf_dot[N-2] + mce_dot) + mf_dot[N-2]*hm[N-2] + mce_dot * hce)/(mn[N-1]*cvn[N-1])
	dvdtheta[N-2] = 1/(mm[N-2]*omegas) * (Am[N-2] * (pn[N-2] - pn[N-1]) + (mom[N-3] - Dn[N-1] * A_dis * vd * np.abs(vd))- np.abs(mf_dot[N-2])*v[N-2] - Am[N-2]*F[N-2])
	dTwalldtheta[N - 1] = -(1/omegas) * (Q_conv_int[N - 1] - Qs_cond[N-1])/(mwall[N - 1] * c_ss) 

	dxdt = dmndtheta+ dTndtheta + dvdtheta + [dmint_dotdtheta, dmout_dotdtheta] + dTwalldtheta
	return dxdt
