import numpy as np
from scipy.optimize import fsolve
from CoolProp.CoolProp import AbstractState

def get_state(input_type, input1, input2):
    state = AbstractState("HEOS", "CO2")
    state.update(input_type, input1, input2)
    return state

def keyed_output(input):
    state = AbstractState("HEOS", "CO2")
    return state.keyed_output(input)

def heat_transfer(A,U, Twall, T):
    return A*U*(Twall - T)

def friction_HX(D, mu, dh, v, roughness):
    Re = v*dh*D/mu
    # K = (1 - min(A1,A2)/max(A1,A2))**2
    if Re < 2*10**3:
        return 64/Re
    else:
        return 0.11*(roughness/dh + 68/Re)**0.25

def _f_haaland(Re, rel_rough=0.0):
    """Haaland friction factor (works for smooth if rel_rough=0)."""
    Re = np.asarray(Re, float)
    return (-1.8 * np.log10((6.9/np.maximum(Re, 1e-12)) + (rel_rough/3.7)**1.11))**-2

def _Nu_gnielinski(Re, Pr, rel_rough=0.0):
    """Gnielinski turbulent single-phase Nu."""
    Re = np.asarray(Re, float); Pr = np.asarray(Pr, float)
    f = _f_haaland(Re, rel_rough)
    num = (f/8.0) * (Re - 1000.0) * Pr
    den = 1.0 + 12.7 * np.sqrt(f/8.0) * (Pr**(2.0/3.0) - 1.0)
    return num / np.maximum(den, 1e-30)

def smooth_phase_blend_GC(Nu_tp, Nu_tr, pc):
    if 70e5 <= pc < 76e5:
        return Nu_tp + (Nu_tr - Nu_tp) / 2.0 * (1 + np.sin(np.pi * (pc - 70e5) / (2.0 * 6e5)))
    else:
        return Nu_tr
    
def Nu_gc_DB(pc, Xc, Re, Pr, dhc, dxc, muc, mu_ss, Tb, Db):
    # --- pseudo-critical temperature correlation (CO2), valid ~7.4–9.0 MPa ---
    def T_pc_corr(p_pa):
        p_mpa = float(p_pa) / 1e6
        Tpc = 304.13 + 7.5*(p_mpa - 7.38)   # K
        # clamp to a reasonable range for robustness
        return min(max(Tpc, 304.13), 330.0)

    T_pc = T_pc_corr(pc)

    Nu_tp = 0.2 * Re**0.85 * Pr**0.6
    Nu_sp = 1.5 * _Nu_gnielinski(Re, Pr, rel_rough=0.0) #0.02 * Re**0.8 * Pr**0.4
    Nu0 = 0.023 * Re**0.8 * Pr**0.4 * (muc / mu_ss)**0.11
    Fpc = 1.0 + 0.35 * np.exp(-((Tb - T_pc)/8.0)**2)  # width ~8 K
    Nu_tr = 0.001 * Re**0.95 * Pr**0.66 #Nu0 * Fpc
    
    if pc >= 70e5:
        Nu = smooth_phase_blend_GC(Nu_tp, Nu_tr, pc) #Nu0 * Fpc
    else:
        # Subcritical
        if 0 <= Xc <= 1:
            Nu = Nu_tp
        else:
            Nu = Nu_sp

    return Nu

def smooth_phase_blend(X, dx, Ul, Utp, Uv):
    if 0 <= X <= 1:
        return Utp
    else:
        return Uv
    
def Nu_evap_DB(pe, Xe, Re, Pr, dhe, dxe, mue, mu_cu, mue_l, mue_v):
    Nu_tp = 0.1 * Re**(0.8)* Pr ** (0.66) #* Xe ** (0.2) # 0.08 * Re**(0.9)* Pr ** (0.5) * Xc ** (0.5)
    Nu_sp = 0.2 * _Nu_gnielinski(Re, Pr, rel_rough=0.0) #0.04 * Re**(0.8) * Pr ** (0.4) #0.029 * Re**(0.637) * Pr ** (0.179) #0.023 * Re**0.8 * Pr ** (0.4)
    return smooth_phase_blend(Xe, 0, Nu_sp, Nu_tp, Nu_sp)

def Nu_DB(pe, Xe, Re, Pr, dhe, dxe, mue, mu_cu, mue_l, mue_v):
    if 0 <= Xe <= 1:
        # if Re < 2 * 10**3:
        #     Nu = (
        #         1.86
        #         * (Re * Pr * dhc / dxc) ** (0.333)
        #         * (muc / mu_ss) ** (0.14)
        #     )  # 3.16
        # else:
        Nu = 0.04 * Re**(0.8)* Pr ** (0.2) * Xe ** (0.2) # 0.08 * Re**(0.9)* Pr ** (0.5) * Xc ** (0.5)
        # if Re < 2*10**3:
        #     Nu = 1.86*(Re*Pr*dhe/dxe)**(0.333)*(mue/mu_cu)**(0.14) #3.16
        # else:
        # Nu =  0.023*Re**0.8*(Pr**(0.4))*(Xe**0.8 + (1 - Xe)**0.8)/2 * (mue_l/mue_v)**0.1
    else:
        # if Re < 2 * 10**3:
        #     Nu = (
        #         1.86
        #         * (Re * Pr * dhc / dxc) ** (0.333)
        #         * (muc / mu_ss) ** (0.14)
        #     )  # 3.16
        # else:
        Nu = 0.02 * Re**(0.8) * Pr ** (0.4) #0.029 * Re**(0.637) * Pr ** (0.179) #0.023 * Re**0.8 * Pr ** (0.4)
        # if Re < 2*10**3:
        #     Nu = 1.86*(Re*Pr*dhe/dxe)**(0.333)*(mue/mu_cu)**(0.14) #3.16
        # else:
        # Nu =  0.023*Re**0.8*Pr**(0.4)
    return Nu

def Nu_ihx_DB(pc, Xc, Re, Pr, dhc, dxc, muc, mu_ss):
    if pc > 73.8e5:
        # if Re < 2 * 10**3:
        #     Nu = (
        #         1.86
        #         * (Re * Pr * dhc / dxc) ** (0.333)
        #         * (muc / mu_ss) ** (0.14)
        #     )
        # else:
        Nu =  0.01 * Re**(0.9) * Pr ** (0.18)
        
    else:
        if 0 <= Xc <= 1:
            # if Re < 2 * 10**3:
            #     Nu = (
            #         1.86
            #         * (Re * Pr * dhc / dxc) ** (0.333)
            #         * (muc / mu_ss) ** (0.14)
            #     )  # 3.16
            # else:
            Nu = 0.02 * Re**(0.8)* Pr ** (0.2) * Xc ** (0.2) # 0.08 * Re**(0.9)* Pr ** (0.5) * Xc ** (0.5)
        else:
            # if Re < 2 * 10**3:
            #     Nu = (
            #         1.86
            #         * (Re * Pr * dhc / dxc) ** (0.333)
            #         * (muc / mu_ss) ** (0.14)
            #     )  # 3.16
            # else:
            Nu = 0.02 * Re**0.8 * Pr ** (0.4) #0.2 * Re**(0.8) * Pr ** (0.2) #0.029 * Re**(0.637) * Pr ** (0.179) #
    return Nu