import numpy as np
import math

from utils import (
	CP,
	state,

)

from config import (
    A,
	R,
	gamma,
    Ae_orifice,
	Ah_sph_ext,
    An,
    A_valve,
    Cd_ce,
    Cd_io,
    Vc_min,
    Ve_min,
	Vcv,
    Vn,
    dh,
	dh_w,
    dx,
    l_rod,
    l_str,
    mu_al,
    mu_ss,
	muw,
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
    roughness_ss,
    Dn,
    mn,
    pn,
    Tn,
    Twall,
    sn,
    psin,
    hn,
    kn,
    vn,
    mun,
    cpn,
    cvn,
    Q_conv_int,
    U,
    mf_dot,
    v,
    f,
    Dm,
    Tm,
	pm,
	hm,
	km,
	cpm,
    Am,
    mum,
    mm,
    psim,
)
from utils import (
	T0,
	p0,
	h0,
	s0,
	mu0,
	k0,
	C,
)
def post_process(y, theta, sin,  pin, pout, Din, hin, omega):
    a_Tk_wall = []
    a_Tr0_wall = []
    a_Tr_wall = []
    a_Tr1_wall = []
    a_Th_wall = []

    a_pc = []
    a_pk = []
    a_pr = []
    a_ph = []
    a_pe = []

    a_Tc = []
    a_Tr0 = []
    a_Tr1 = []
    a_Te = []
    a_Tk = []
    a_Th = []
    a_Tr = []

    a_Deltapk = []
    a_Deltapkr = []
    a_Deltapr = []
    a_Deltaphr = []
    a_Deltaph = []
    a_Deltap = []

    a_vk = []
    a_vkr = []
    a_vr = []
    a_vhr = []
    a_vh = []

    a_Qc = []
    a_Qk = []
    a_Qkr = []
    a_Qr = []
    a_Qhr = []
    a_Qh = []
    a_Qe = []
    a_Qs_cond = []

    rm = []
    a_W = []
    a_Pout = []
    a_Tout = []
    a_hout = []
    a_Ve = []
    a_Vc = []
    a_Vt = []

    a_mc = []
    a_mk = []
    a_mkr = []
    a_mr = []
    a_mhr = []
    a_mh = []
    a_me = []

    a_Dc = []
    a_Dk = []
    a_Dkr = []
    a_Dr = []
    a_Dhr = []
    a_Dh = []
    a_De = []

    a_mout_dot = []
    a_mint_dot = []
    a_mck_dot = []
    a_meh_dot = []
    a_mkr_dot = []
    a_mrh_dot = []

    a_theta = []
    a_mdot = []
    a_alpha = []
    Hout_dot = []

    a_Edest_k = []
    a_Edest_kr = []
    a_Edest_r = []
    a_Edest_hr = []
    a_Edest_h = []
    a_Edest_c = []
    a_Edest_e = []
    a_Ex_eff = []
    Q_array = np.zeros((len(theta), nreg))
    for k in range(len(theta)):
        omegas = omega * 2 * np.pi / 60
        xd = r_cra*(1 - np.cos(theta[k])) + l_rod*(1 - np.sqrt(1-(r_cra/l_rod * np.sin(theta[k]))**2))
        vd = r_cra*omegas*np.sin(theta[k])*(1 + (r_cra/l_rod * np.cos(theta[k]))/(np.sqrt(1-(r_cra/l_rod * np.sin(theta[k]))**2)))
        Vn[-1] = np.pi*r_dis**2*xd + Ve_min 
        Ve_dot = np.pi*r_dis**2*vd 
        Vn[0] = np.pi*(r_dis**2 - r_shaft**2)*(l_str - xd) + Vc_min
        Vc_dot = -np.pi*(r_dis**2 - r_shaft**2)*vd 
        An[0] = 2*np.pi*r_dis*(l_str - xd + xc_min)
        An[-1] = 2*np.pi*r_dis*xd

        mk = 0
        mkr = 0
        mr = 0
        mhr = 0
        mh = 0
        Qc_dot_sum = 0
        Qk_dot_sum = 0
        Qkr_dot_sum = 0
        Qr_dot_sum = 0
        Qhr_dot_sum = 0
        Qh_dot_sum = 0
        Qe_dot_sum = 0
    
        Ec_dot_sum = 0
        Ek_dot_sum = 0
        Ekr_dot_sum = 0
        Er_dot_sum = 0
        Ehr_dot_sum = 0
        Eh_dot_sum = 0
        Ee_dot_sum = 0

        Q_total = 0
        Tk = 0
        Th = 0
        for i in range(N):
            mn[i] = y[i, k]
            Tn[i] = y[i + N, k]
            Twall[i] = y[i + 3 * N + 1, k]
            Dn[i] = mn[i]/Vn[i]
            state.update(CP.DmassT_INPUTS, Dn[i], Tn[i])
            pn[i] = state.p()
            hn[i]= state.hmass()
            sn[i]= state.smass()
            psin[i] = (hn[i] - h0) - T0 * (sn[i] - s0)
            kn[i] = state.conductivity()
            mun[i]= state.viscosity()
            cpn[i] = state.cpmass()
            cvn[i] = state.cvmass() 
            if 1 <= i < nk +1:
                Tk += Tn[i]
            elif (nk + nkr + nreg + nhr + 1 <= i < N-1):
                Th += Tn[i]
        Tk = Tk/nk
        Th = Th/nh

        for i in range(N-1):
            v[i] = y[i + 2*N, k]
            if v[i] >= 0:
                hm[i] = hn[i] 
                pm[i] = pn[i]
                Tm[i] = Tn[i]
                Dm[i] = Dn[i]
                mm[i] = (mn[i] + mn[i+1])/2
                mum[i] = mun[i]
                km[i] = kn[i]
                cpm[i] = cpn[i]
                psim[i] = psin[i]
            else:
                hm[i] = hn[i+1] 
                pm[i] = pn[i+1]
                Tm[i] = Tn[i+1]
                Dm[i] = Dn[i+1]
                mm[i] = (mn[i] + mn[i+1])/2
                mum[i] = mun[i+1]
                km[i] = kn[i+1]
                cpm[i] = cpn[i+1]
                psim[i] = psin[i + 1]
            mf_dot[i] = Cd_ce * Am[i] * Dm[i] * v[i]

        mint_dot = y[3*N-1, k]
        mout_dot = y[3*N, k]

        for i in range(N):
            if i == 0:
                vn[i] = (vd + v[i])/2
            elif (1 <= i < N-1):
                vn[i] = (v[i-1] + v[i])/2
            else:
                vn[i] = (v[i-1] + vd)/2

            Re = np.abs(vn[i])*dh[i]*Dn[i]/mun[i]
            Pr = mun[i]*cpn[i]/kn[i]
            if i < nk+1:
                if Re < 2*10**3:
                    Nu = 1.86*(Re*Pr*dh[i]/dx[i])**(0.333)*(mun[i]/mu_al)**(0.14) #3.16
                else:
                    Nu =  0.023*Re**0.8*(Pr**(0.3))
                U[i] = kn[i]*Nu/dh[i]
                if i == 0:
                    Q_conv_int[i] = An[i] * U[i] * (Tk - Tn[i])
                    Qk_dot_sum += Q_conv_int[i]
                    Ec_dot_sum += (1 - T0/Tk) * Q_conv_int[i]
                elif 1 <= i:
                    mk += mn[i]
                    Q_conv_int[i] = An[i] * U[i] * (Twall[i] - Tn[i])
                    Qk_dot_sum += Q_conv_int[i]
                    Ek_dot_sum += (1 - T0/Twall[i]) * Q_conv_int[i]
            elif (nk+1 <= i < nk + nkr+1):
                if Re < 2*10**3:
                    Nu = 1.86*(Re*Pr*dh[i]/dx[i])**(0.333)*(mun[i]/mu_ss)**(0.14) #3.16
                else:
                    Nu =  0.023*Re**0.8*(Pr**(0.3))
                U[i] = kn[i]*Nu/dh[i]
                Q_conv_int[i] = An[i] * U[i] * (Twall[i] - Tn[i])
                Qk_dot_sum += Q_conv_int[i]
                Ekr_dot_sum += (1 - T0/Twall[i]) * Q_conv_int[i]
                mk += mn[i]
            elif (nk+ nkr +1<= i < nk + nkr + nreg+1):
                Nu = (1+0.99*(Re*Pr)**0.66)*phi**1.79  #or Nu = 0.51 + 0.4*Re**0.66 (Gedeon and wood))
                U[i] = kn[i]*Nu/dh[i]
                Q_conv_int[i] = An[i] * U[i] * (Twall[i] - Tn[i])
                Qr_dot_sum += Q_conv_int[i]
                Er_dot_sum += (1 - T0/Twall[i]) * Q_conv_int[i]
                mr += mn[i]
            else:
                if Re < 2*10**3:
                    Nu = 1.86*(Re*Pr*dh[i]/dx[i])**(0.333)*(mun[i]/mu_ss)**(0.14) #3.16
                else:
                    Nu =  0.023*Re**0.8*(Pr**(0.4))
                U[i] = kn[i]*Nu/dh[i]
                if nk + nkr + nreg + 1 <= i < nk + nkr + nreg + nhr + 1:
                    Q_conv_int[i] = An[i] * U[i] * (Twall[i] - Tn[i])
                    mhr += mn[i]
                    Qhr_dot_sum += Q_conv_int[i]
                    Ehr_dot_sum += (1 - T0/Twall[i]) * Q_conv_int[i]
                elif nk + nkr + nreg + nhr + 1 <= i < N-1:
                    Q_conv_int[i] = An[i] * U[i] * (Twall[i] - Tn[i])
                    mh += mn[i]
                    Qh_dot_sum += Q_conv_int[i]
                    Eh_dot_sum += (1 - T0/Twall[i]) * Q_conv_int[i]
                else:
                    Q_conv_int[i] = An[i] * U[i] * (Th - Tn[i])
                    Qe_dot_sum += Q_conv_int[i]
                    Ee_dot_sum += (1 - T0/Th) * Q_conv_int[i]

        Q_total = np.sum(Q_conv_int[1:N-1])

        # a_Tk_wall.append(Twall[1])
        # a_Tr0_wall.append(Twall[nk+ nkr])
        # a_Tr1_wall.append(Twall[nk+ nkr + nreg-1])
        # a_Th_wall.append(Twall[-2])
        # a_Tc.append(Tn[0])
        # a_Tk.append(Tn[1])
        # a_Tr0.append(Tn[nk+ nkr])
        # a_Tr1.append(Tn[nk+ nkr+nreg-1])
        # a_Th.append(Tn[-2])
        # a_Te.append(Tn[-1])
        # a_mc.append(Dc*Vc)
        # a_mk.append(mk)
        # a_mkr.append(mkr)
        # a_mr.append(mr)
        # a_mhr.append(mhr)
        # a_mh.append(mh)
        # a_me.append(De*Ve)
        a_Deltapk.append((pn[0] - pn[nk+nkr + 2])*10**(-5))
        # a_Deltapkr.append((pn[nk] - pn[nk + nkr+1])*10**(-5))
        a_Deltapr.append((pn[nk + nkr+1] - pn[nk + nkr + nreg+2])*10**(-5))
        # a_Deltaphr.append((pn[nk + nkr + nreg] - pn[-1])*10**(-5))
        a_Deltaph.append((pn[nk + nkr + nreg + nhr+1] - pn[-1])*10**(-5))
        a_vk.append(v[0])
        a_vkr.append(v[nk+nkr])
        a_vr.append(v[nk+nkr+nreg])
        a_vhr.append(v[nk+nkr+nreg+nhr])
        a_vh.append(v[-1])

        a_theta.append(theta[k]*180/np.pi)
        
        a_mint_dot.append(mint_dot*10**3)
        a_mout_dot.append(mout_dot*10**3)
        a_mck_dot.append(mf_dot[0])
        a_meh_dot.append(mf_dot[-1])
        a_mkr_dot.append(mf_dot[nk + nkr])
        a_mrh_dot.append(mf_dot[nk + nkr + nreg])
        rm.append(mf_dot[-1]/mf_dot[0])
        hc_out = 0
        if (mout_dot > 1e-3):
            state.update(CP.PSmass_INPUTS, pout, sn[0])
            a_Tout.append(state.T())
            a_hout.append(state.hmass())
            hc_out = state.hmass()
        hc_in = 0
        if (mint_dot > 1e-3):
            # state.update(CP.PSmass_INPUTS, Tn[0], sin)
            hc_in = hin #state.hmass()

        psi_in = (hc_in - h0) - T0 * (sin - s0)
        W = pn[-1]*Ve_dot + pn[0]*Vc_dot 
        a_mdot.append(np.abs(mint_dot - mout_dot)/2)
        a_W.append(W)
        a_Pout.append(mout_dot*hn[0] - mint_dot*hc_in)
        a_mc.append(Dn[0]*Vn[0]*10**3)
        a_mk.append(mk*10**3)
        a_mkr.append(mkr*10**3)
        a_mr.append(mr*10**3)
        a_mhr.append(mhr*10**3)
        a_mh.append(mh*10**3)
        a_me.append(Dn[-1]*Vn[-1]*10**3)
        a_Vc.append(Vn[0]*10**(6))
        a_Ve.append(Vn[-1]*10**(6))
        a_Vt.append((Vn[-1]+Vn[0])*10**(6))
        a_Tc.append(Tn[0])
        a_Tk.append(np.mean(Tn[1:nk+nkr+1]))
        a_Tr.append(np.mean(Tn[nk+nkr+1:nk+nkr+nreg+1]))
        a_Th.append(np.mean(Tn[nk+nkr+nreg+1:nk+nkr+nreg+nhr+nh+1]))
        a_Te.append(Tn[-1])
        a_Tk_wall.append(np.mean(Twall[1:nk+nkr+1]))
        a_Tr_wall.append(np.mean(Twall[nk+nkr+1:nk+nkr+nreg+1]))
        a_Th_wall.append(np.mean(Twall[nk+nkr+nreg+1:nk+nkr+nreg+nhr+nh+1]))
        a_pc.append(pn[0]*10**(-5))
        a_pk.append(np.mean(pn[1:nk+nkr+1])*10**(-5))
        a_pr.append(np.mean(pn[nk+nkr+1:nk+nkr+nreg+1])*10**(-5))
        a_ph.append(np.mean(pn[nk+nkr+nreg+1:nk+nkr+nreg+nhr+nh+1])*10**(-5))
        a_pe.append(pn[-1]*10**(-5))
        a_Deltap.append((pn[0] - pn[-1])*10**(-5))
        a_Qc.append(Qc_dot_sum)
        a_Qk.append(Qk_dot_sum)
        a_Qkr.append(Qkr_dot_sum)
        a_Qr.append(Qr_dot_sum)
        a_Qhr.append(Qhr_dot_sum)
        a_Qh.append(Qh_dot_sum)
        a_Qe.append(Qe_dot_sum)
        a_Dc.append(Dn[0])
        a_Dk.append(np.mean(Dn[1:nk+nkr+1]))
        a_Dr.append(np.mean(Dn[nk+nkr+1:nk+nkr+nreg+1]))
        a_Dh.append(np.mean(Dn[nk+nkr+nreg+1:nk+nkr+nreg+nhr+nh+1]))
        a_De.append(Dn[-1])
        a_alpha.append(Q_total - W - (mout_dot*hn[0] - mint_dot*hc_in))

        a_Edest_c.append(- mf_dot[0] * psim[0] - pn[0] * Vc_dot + Ec_dot_sum) #+mint_dot * psi_in - mout_dot * psin[0]
        a_Edest_k.append(mf_dot[0] * psim[0] - mf_dot[nk] * psim[nk] + Ek_dot_sum)
        a_Edest_kr.append(mf_dot[nk] * psim[nk] - mf_dot[nk + nkr] * psim[nk + nkr] + Ekr_dot_sum)
        a_Edest_r.append(mf_dot[nk + nkr] * psim[nk + nkr] -  mf_dot[nk + nkr + nreg] * psim[nk + nkr + nreg] + Er_dot_sum)
        a_Edest_hr.append(mf_dot[nk + nkr + nreg] * psim[nk + nkr + nreg] - mf_dot[nk + nkr + nreg + nhr] * psim[nk + nkr + nreg + nhr] + Ehr_dot_sum)
        a_Edest_h.append(mf_dot[nk + nkr + nreg + nhr] * psim[nk + nkr + nreg + nhr] - mf_dot[N-2] * psim[N-2]  + Eh_dot_sum)
        a_Edest_e.append(mf_dot[N-2] * psim[N-2]  - pn[-1] * Ve_dot + Ee_dot_sum)

        # a_Ex_eff.append((mout_dot * psin[0] - mint_dot * psi_in - (min(Ek_dot_sum, 0) + min(Ekr_dot_sum, 0) + min(Er_dot_sum, 0)))/(- W + max(Eh_dot_sum, 0) + max(Ehr_dot_sum, 0) + max(Er_dot_sum, 0)))
        a_Ex_eff.append((mout_dot * psin[0] - mint_dot * psi_in - (min(Ek_dot_sum, 0) + min(Ekr_dot_sum, 0) + min(Er_dot_sum, 0)))/(- W + max(Eh_dot_sum, 0) + max(Ehr_dot_sum, 0) + max(Er_dot_sum, 0)))

        # a_Ex_eff.append(1 - (a_Edest_c[-1] + a_Edest_k[-1] + a_Edest_kr[-1] + a_Edest_r[-1] + a_Edest_hr[-1] + a_Edest_h[-1] + a_Edest_e[-1])/(- W + max(Eh_dot_sum, 0) + max(Ehr_dot_sum, 0) + max(Er_dot_sum, 0)))

        Q_array[k, :] = np.array(Q_conv_int[nk + nkr + 1: nk + nkr + nreg + 1])

    return (
        a_Pout,
        a_mck_dot,
        a_meh_dot,
        a_mkr_dot,
        a_mrh_dot,
        a_mint_dot,
        a_mout_dot,
        a_alpha,
        a_W,
        a_Deltap,
        a_Deltapk,
        a_Deltapkr,
        a_Deltapr,
        a_Deltaphr,
        a_Deltaph,
        a_mdot,
        a_Te,
        a_Th,
        a_Tr,
        a_Tc,
        a_Tk,
        a_Tk_wall,
        a_Tr_wall,
        a_Th_wall,
        a_Tout,
        a_hout,
        a_Vc,
        a_Ve,
        a_pc,
        a_pk,
        a_pr,
        a_ph,
        a_pe,
        a_theta,
        a_Dc,
        a_Dk,
        a_Dkr,
        a_Dr,
        a_Dhr,
        a_Dh,
        a_De,
        a_mc,
        a_mk,
        a_mkr,
        a_mr,
        a_mhr,
        a_mh,
        a_me,
        a_vk,
        a_vkr,
        a_vr,
        a_vhr,
        a_vh,
        a_Qc,
        a_Qk,
        a_Qkr,
        a_Qr,
        a_Qhr,
        a_Qh,
        a_Qe,
        a_Edest_c,
        a_Edest_k,
        a_Edest_kr,
        a_Edest_r,
        a_Edest_hr,
        a_Edest_h,
        a_Edest_e,
        a_Ex_eff,
        Q_array
    )

