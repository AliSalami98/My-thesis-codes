import config
from utils import get_state, Nu_gc_DB
from config import CP
import numpy as np


def init_gas_cooler(x, pc_in, hc_in):
    config.pc = min(max(x[1 + 3 * config.Ne], 50e5), 100e5) #pc_in #
    if 73.3e5 < config.pc < 73.8e5:
        config.pc = 73.3e5
    for i in range(config.Nc):
        config.hc[i] = min(max(x[i + 2 + 3 * config.Ne], 220e3), hc_in) #max(x[i + config.Nc], 220e3)

        # print(config.hc[i])
        config.Tc_wall[i] = x[i + config.Nc + 2 + 3 * config.Ne]
        config.Tc_w[i] = x[i + 2 * config.Nc + 2 + 3 * config.Ne]

        state = get_state(CP.HmassP_INPUTS, config.hc[i], config.pc)
        config.Xc[i] = state.Q()
        config.Tc[i] = state.T()
        
        config.Dc[i] = state.rhomass()
        config.kc[i] = state.conductivity()
        config.sc[i] = state.smass()
        config.muc[i] = state.viscosity()
        config.cpc[i] = np.abs(state.cpmass())
        config.cvc[i] = state.cvmass()
        config.mc[i] = config.Dc[i] * config.Vc[i]
        config.dDdp_hc[i] = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
        config.dDdh_pc[i] = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
        if 0 <= config.Xc[i] <= 1:
            # config.dDdh_pc[i] = state.first_two_phase_deriv_splined(CP.iDmass, CP.iHmass, CP.iP, 0.3)
            # config.dDdp_hc[i] = state.first_two_phase_deriv_splined(CP.iDmass, CP.iP, CP.iHmass, 0.3)
            state = get_state(CP.QT_INPUTS, 0, config.Tc[i])
            config.muc_l[i] = state.viscosity()
            state = get_state(CP.QT_INPUTS, 1, config.Tc[i])
            config.muc_v[i] = state.viscosity()

def calc_gas_cooler(mc_in_dot, mc_out_dot, Dc_in, mw2_dot, Tw_in):
    for i in range(config.Nc + 1):
        if i == 0:
            config.Dc_m[i] = (Dc_in + config.Dc[i]) / 2
            config.vc[i] = mc_in_dot / (config.Dc_m[i] * config.ACO2_orifice)
        elif i == config.Nc:
            config.Dc_m[i] = (config.Dc[i-1])
            config.vc[i] = mc_in_dot / (config.Dc_m[i] * config.ACO2_orifice)
        else:
            config.Dc_m[i] = (config.Dc[i-1] + config.Dc[i]) / 2
            config.vc[i] = mc_in_dot / (config.Dc_m[i] * config.ACO2_orifice)
        # config.hc_m[i] = (config.hc[i] + config.hc[i + 1]) / 2
        # config.pc_m[i] = (config.pc + config.pc[i + 1]) / 2
        # config.Tc_m[i] = (config.Tc[i] + config.Tc[i + 1]) / 2
        # config.mc_m[i] = (config.mc[i] + config.mc[i + 1]) / 2
        # config.muc_m[i] = (config.muc[i] + config.muc[i + 1]) / 2
        # config.kc_m[i] = (config.kc[i] + config.kc[i + 1]) / 2
        # config.cpc_m[i] = (config.cpc[i] + config.cpc[i + 1]) / 2

    for i in range(config.Nc):
        config.vc_n[i] = (config.vc[i] + config.vc[i+1]) / 2

        Re = np.abs(config.vc_n[i]) * config.dhc[i] * config.Dc[i] / config.muc[i]
        Pr = config.muc[i] * config.cpc[i] / config.kc[i]

        Nu = Nu_gc_DB(config.pc, config.Xc[i], Re, Pr, config.dhc[i], config.dxc[i], config.muc[i], config.mu_cu, config.Tc[i], config.Dc[i])
        # Nu = Nu_gc_DB_modified(config.pc, config.Xc[i], Re, Pr, config.dhc[i], config.dxc[i], config.muc[i], config.mu_ss, config.Dc[i], config.Dc_pc)
        config.Uc[i] = config.kc[i] * Nu / config.dhc[i]
        
        config.Qc_conv[i] = (
            config.Ac[i] * config.Uc[i] * (config.Tc_wall[i] - config.Tc[i])
        )

        Rew = np.abs(mw2_dot) * config.dhc_w / (config.Aw_orifice * config.muw)
        Prw = config.muw * config.cpw / config.kw
        Nuw = 0.023 * Rew**0.8 * Prw ** (0.4)
        Uw = config.kw * Nuw / config.dhc_w
        config.Qc_w_conv[i] = config.Ac[i] * Uw * (config.Tc_wall[i] - config.Tc_w[i])
    return config.Qc_conv, config.Qc_w_conv


def model_gas_cooler(mc_in_dot, mc_out_dot, hc_in, mw2_dot, Tw_in):

    xc = np.hstack(([config.pc], config.hc, config.Tc_wall, config.Tc_w))  # The state vector consists of pressure and enthalpy values

    Zc = np.zeros((3 * config.Nc + 1, 3 * config.Nc + 1))

    Zc[0, 0] = config.dDdp_hc[0] * config.Vc[0]
    Zc[0, 1] = config.dDdh_pc[0] * config.Vc[0]
    
    for i in range(1, config.Nc + 1):
        Zc[i, 0] = -1 * config.Vc[i-1]  # Pressure-related term
        Zc[i, i] = config.Vc[i-1] * config.Dc[i-1]  # Enthalpy-related ter

    for i in range(config.Nc + 1, 2 * config.Nc + 1):
        Zc[i, i] = config.mc_wall[i - (config.Nc + 1)] * config.c_ss

    for i in range(2 * config.Nc + 1, 3 * config.Nc + 1):
        Zc[i, i] = config.mc_w[i - (2 * config.Nc + 1)] * config.cpw

    f_vec = np.zeros(3 * config.Nc + 1)

    f_vec[0] = mc_in_dot - mc_out_dot

    f_vec[1] = config.Qc_conv[0] + mc_in_dot * (hc_in - xc[1])
    for i in range(2, config.Nc+1):
        f_vec[i] = config.Qc_conv[i-1] + mc_in_dot * (xc[i-1] - xc[i])

    for i in range(config.Nc + 1, 2 * config.Nc + 1):
        f_vec[i] = -(config.Qc_w_conv[i - (config.Nc + 1)] + config.Qc_conv[i - (config.Nc + 1)])

    for i in range(2 * config.Nc + 1, 3 * config.Nc + 1):
        f_vec[i] = (mw2_dot * config.cpw *  config.Qc_w_conv[i - (2 * config.Nc + 1)])

    f_vec[-1] = mw2_dot * config.cpw * (Tw_in - xc[0]) + config.Qc_w_conv[-1]
    Z_inv = np.linalg.inv(Zc)  # Inverse of Z_c matrix
    dxcdt = np.dot(Z_inv, f_vec)

    config.dpcdt = dxcdt[0]
    config.dhcdt = dxcdt[1:config.Nc + 1].tolist()
    config.dTc_walldt = dxcdt[config.Nc + 1:].tolist()
    config.dTc_wdt = dxcdt[2 * config.Nc + 1:].tolist()
    

def init_gas_cooler_post(y, k, pc_in):
    config.pc = min(max(y[1 + 3 * config.Ne, k], 50e5), 100e5) # #pc_in #
    if 73.3e5 < config.pc < 73.8e5:
        config.pc = 73.3e5
    for i in range(config.Nc):
        config.hc[i] = min(max(y[i + 2 + 3 * config.Ne, k], 220e3), 650e3)
        config.Tc_wall[i] = y[i + config.Nc + 2 + 3 * config.Ne, k]
        config.Tc_w[i] = y[i + 2 * config.Nc + 2 + 3 * config.Ne, k]
        state = get_state(CP.HmassP_INPUTS, config.hc[i], config.pc)
        config.Xc[i] = state.Q()
        config.Tc[i] = state.T()
        
        config.Dc[i] = state.rhomass()
        config.kc[i] = state.conductivity()
        config.sc[i] = state.smass()
        config.muc[i] = state.viscosity()
        config.cpc[i] = np.abs(state.cpmass())
        config.cvc[i] = state.cvmass()
        config.mc[i] = config.Dc[i] * config.Vc[i]
        config.dDdp_hc[i] = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
        config.dDdh_pc[i] = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
        if 0 <= config.Xc[i] <= 1:
            # config.dDdh_pc[i] = state.first_two_phase_deriv_splined(CP.iDmass, CP.iHmass, CP.iP, 0.3)
            # config.dDdp_hc[i] = state.first_two_phase_deriv_splined(CP.iDmass, CP.iP, CP.iHmass, 0.3)
            state = get_state(CP.QT_INPUTS, 0, config.Tc[i])
            config.muc_l[i] = state.viscosity()
            state = get_state(CP.QT_INPUTS, 1, config.Tc[i])
            config.muc_v[i] = state.viscosity()
