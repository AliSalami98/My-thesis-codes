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

def Nu_gc_DB(pc, Xc, Re, Pr, dhc, dxc, muc, mu_ss):
    if pc > 73.8e5:
        # if Re < 2 * 10**3:
        #     Nu = (
        #         1.86
        #         * (Re * Pr * dhc / dxc) ** (0.333)
        #         * (muc / mu_ss) ** (0.14)
        #     )
        # else:
        Nu =  0.003 * Re**(0.9) * Pr ** (0.18)
        
    else:
        if 0 <= Xc <= 1:
            # if Re < 2 * 10**3:
            #     Nu = (
            #         1.86
            #         * (Re * Pr * dhc / dxc) ** (0.333)
            #         * (muc / mu_ss) ** (0.14)
            #     )  # 3.16
            # else:
            Nu = 0.05 * Re**(0.8)* Pr ** (0.2) * Xc ** (0.2) # 0.08 * Re**(0.9)* Pr ** (0.5) * Xc ** (0.5)
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


def Nu_evap_DB(pe, Xe, Re, Pr, dhe, dxe, mue, mu_cu, mue_l, mue_v):
    if 0 <= Xe <= 1:
        # if Re < 2 * 10**3:
        #     Nu = (
        #         1.86
        #         * (Re * Pr * dhc / dxc) ** (0.333)
        #         * (muc / mu_ss) ** (0.14)
        #     )  # 3.16
        # else:
        Nu = 0.05 * Re**(0.8)* Pr ** (0.2) * Xe ** (0.2) # 0.08 * Re**(0.9)* Pr ** (0.5) * Xc ** (0.5)
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