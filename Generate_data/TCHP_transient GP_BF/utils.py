import numpy as np
from scipy.optimize import fsolve
from CoolProp.CoolProp import AbstractState
import CoolProp.CoolProp as CP

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

def smooth_step(x, x0, width):
    """Smooth step function transitioning from 0 to 1 around x0 within a given width"""
    return 1 / (1 + np.exp(-(x - x0) / width))


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

    Nu_tp = 0.05 * Re**0.8 * Pr**0.2
    Nu_sp = 0.02 * Re**0.8 * Pr**0.4
    Nu0 = 0.023 * Re**0.8 * Pr**0.4 * (muc / mu_ss)**0.11
    Fpc = 1.0 + 0.35 * np.exp(-((Tb - T_pc)/8.0)**2)  # width ~8 K
    Nu_tr = 0.014 * Re**0.66 * Pr**0.6 #Nu0 * Fpc
    
    if pc >= 70e5:
        # Supercritical: baseline + pseudo-critical bump (enhancement near T_pc)
        Nu0 = 0.023 * Re**0.8 * Pr**0.4 * (muc / mu_ss)**0.11
        Fpc = 1.0 + 0.35 * np.exp(-((Tb - T_pc)/8.0)**2)  # width ~8 K
        Nu = smooth_phase_blend_GC(Nu_tp, Nu_tr, pc) #Nu0 * Fpc
    else:
        # Subcritical
        if 0 <= Xc <= 1:
            Nu = Nu_tp
        else:
            Nu = Nu_sp

    return Nu

# def smooth_phase_blend(X, dx, Ul, Utp, Uv):
#     if X <= -dx:
#         return Ul
#     elif  0 < X < dx:
#         return Ul + (Utp - Ul) / 2.0 * (1 + np.sin(np.pi * X / (2.0 * dx)))
#     elif dx <= X <= 1 - dx:
#         return Utp
#     elif 1 - dx < X < 1:
#         return Utp + (Uv - Utp) / 2.0 * (1 + np.sin(np.pi * (X - 1) / (2.0 * dx)))
#     else:
#         return Uv
    
def Nu_evap_DB(pe, Xe, Re, Pr, dhe, dxe, mue, mu_cu, mue_l, mue_v):
    Nu_tp = 0.05 * Re**(0.8)* Pr ** (0.2) #* Xe ** (0.2) # 0.08 * Re**(0.9)* Pr ** (0.5) * Xc ** (0.5)
    Nu_sp = 0.02 * Re**(0.8) * Pr ** (0.4) #0.029 * Re**(0.637) * Pr ** (0.179) #0.023 * Re**0.8 * Pr ** (0.4)
    if 0 < Xe < 1:
        return Nu_tp
    else:
        return Nu_sp
    # return smooth_phase_blend(Xe, 0.1, Nu_sp, Nu_tp, Nu_sp)

def Nu_DB(pe, Xe, Re, Pr, dhe, dxe, mue, mu_cu, mue_l, mue_v):
    if 0 <= Xe <= 1:
        Nu = 0.04 * Re**(0.8)* Pr ** (0.2) * Xe ** (0.2) # 0.08 * Re**(0.9)* Pr ** (0.5) * Xc ** (0.5)
    else:
        Nu = 0.02 * Re**(0.8) * Pr ** (0.4) #0.029 * Re**(0.637) * Pr ** (0.179) #0.023 * Re**0.8 * Pr ** (0.4)
    return Nu

def interpolate_single(x_values, y_values, target_x):
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("x_values must be sorted in ascending order.")
    
    interpolated_y = np.interp(target_x, x_values, y_values)
    return interpolated_y

def calculate_T_diff_sp(T_supply, T_supply_ref, offset=0):
    T_diff = 7 + ((T_supply - T_supply_ref) / 30) * 10 + offset
    return T_diff

def calculate_supply_temp(T_supply, T_outdoor, T_room_sp, T_supply_ref, T_design=263.15, slope=1.5, offset=0):
    T_diff_sp = calculate_T_diff_sp(T_supply, T_supply_ref, offset)
    
    if T_outdoor < T_design:  # Case 1
        T_supply_sp = (
            -14 / 5 
            + (6 / 5) * (T_room_sp - 273.15) 
            + (6 / 5) * ((T_supply_ref - T_diff_sp / 2 - T_room_sp)) 
            + 273.15
        )
    elif T_outdoor > 289.15:  # Case 2
        T_supply_sp = (
            -14 / 5 
            + (6 / 5) * (T_room_sp - 273.15) 
            + (6 / 5) * (
                ((289.15 - T_room_sp) / (T_design - T_room_sp)) ** (1 / slope) 
                * (T_supply_ref - T_diff_sp / 2 - T_room_sp)
            ) 
            + 273.15
        )
    else:  # Case 3
        T_supply_sp = (
            -14 / 5 
            + (6 / 5) * (T_room_sp - 273.15) 
            + (6 / 5) * (
                ((T_outdoor - T_room_sp) / (T_design - T_room_sp)) ** (1 / slope) 
                * (T_supply_ref - T_diff_sp / 2 - T_room_sp)
            ) 
            + 273.15
        )
    return T_supply_sp

def fuzzy_control(error, prev_error, y, Hpev_opt):
    if error/y > 0.01:
        Hpev_opt -= 1
    elif error/y < - 0.01:
        Hpev_opt += 1
    return min(max(Hpev_opt, 11), 100)

def fuzzy_control_mw(error, prev_error, y, mw_dot):
    if error/y > 0.01:
        mw_dot -= 0.01
    elif error/y < - 0.01:
        mw_dot += 0.01
    return min(max(mw_dot, 0.1), 0.4)

def fuzzy_control_Theater(error, prev_error, y, Theater_opt):
    if error/y > 0.001:
        Theater_opt += 5
    elif error/y < - 0.001:
        Theater_opt -= 5
    return min(max(Theater_opt, 773.15), 1073.15)

def run_pid_velocity(error, kp, ki, prev_error, prev_Hpev):
    new_Hpev = prev_Hpev + (kp + ki) * error - kp * prev_error
    print(new_Hpev) 
    return min(max(-(new_Hpev), 11), 100)

def run_pi_aw(error, I_state, kp, ki, dt, u_min, u_max, direction=-1.0, kaw=None):
    """
    PI controller with anti-windup.

    error:      setpoint - measurement
    I_state:    integral accumulator (already scaled as a control term)
    kp, ki:     gains (continuous-time); integral increment uses ki*error*dt
    dt:         sampling time [s]
    u_min/max:  actuator limits (e.g., valve opening [%])
    direction:  +1 if positive error should increase u, -1 if it should decrease u
    kaw:        back-calculation gain (1/s). If None, uses integrator clamping.
    """
    # Proportional term (with direction)
    P = direction * kp * error

    # Pre-update integrator (as control contribution)
    I_candidate = I_state + direction * ki * error * dt

    # Unsaturated control
    u_unsat = P + I_candidate

    # Apply saturation
    u_sat = max(u_min, min(u_unsat, u_max))

    if kaw is None:
        # CLAMPING: only accept integrator update if not saturating,
        # or if the update would move u back inside the limits.
        saturating_high = u_unsat > u_max
        saturating_low  = u_unsat < u_min

        if not (saturating_high or saturating_low):
            I_state = I_candidate
        else:
            # Allow integration only if it RELIEVES saturation
            # (i.e., moves u back toward the feasible region).
            if saturating_high and (direction * error) < 0:
                I_state = I_candidate
            elif saturating_low and (direction * error) > 0:
                I_state = I_candidate
            # else: hold I_state (freeze)
    else:
        # BACK-CALCULATION: bleed the integrator with tracking error
        I_state = I_candidate + kaw * (u_sat - u_unsat) * dt

    return u_sat, I_state

def run_pid_Hpev(error, I_Hpev, kp, ki, dt, kaw):
    # If increasing error should CLOSE HPEV, keep direction = -1
    return run_pi_aw(error, I_Hpev, kp, ki, dt, u_min=11.0, u_max=100.0, direction=-1.0, kaw=kaw)

def run_pid_Lpev(error, I_Lpev, kp, ki, dt, kaw):
    # If increasing error should CLOSE LPEV as well
    return run_pi_aw(error, I_Lpev, kp, ki, dt, u_min=11.0, u_max=100.0, direction=-1.0, kaw=kaw)

def run_pid_omegab(error, I_omegab, kp_omegab, ki_omegab, dt, kaw):
    return run_pi_aw(error, I_omegab, kp_omegab, ki_omegab, dt, u_min=1950, u_max=9500, direction= 1.0, kaw=kaw)

def run_pid_Theater(error, I, J, kp, ki):
    I += error
    I = min(max(ki * I, -400),  400)

    # Proportional term for PID
    P = kp * error
    new_Theater = (P + I)
    # print(I)
    
    return new_Theater, I

def run_pid_omega2(error, I_omega2, kp_omega2, ki_omega2, dt, kaw):
    return run_pi_aw(error, I_omega2, kp_omega2, ki_omega2, dt, u_min=80, u_max=240, direction= -1.0, kaw=kaw)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt



r_sph_ext = 0.0645
c_ss = 500      # J/Kg.K
mheater = 5.56

LHVf = 50 * 10**6
xi_air = 17.08 / 18.08
xi_CH4 = 1 / 18.08
M_CH4 = 16.15
M_air = 28.97
M_mix = xi_air * M_air + xi_CH4 * M_CH4

Vg = 0.00495
Dg = 1.29  # ambient density of air
Df = 0.657
mg = Dg * Vg
st_ratio = 17.125

cpa = 1005
cpf = 35800
cpg = xi_CH4 * cpf + xi_air * cpa
mug = 1.48 * 10**(-5)
kg = 4 * 10**(-2)
Tg_in = 298.15
dhg = 2.54 * 10**(-2)
Ag_orifice = np.pi / 4 * dhg**2