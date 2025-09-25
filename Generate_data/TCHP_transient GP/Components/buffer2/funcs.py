import config
from utils import get_state, Nu_DB
from config import CP
import numpy as np


def init_buffer2(x, pbuff2_in, pe):
    config.pbuff2 = x[3 * (config.Nc + config.Ne) + 10] # #pbuff2_in #
    if 73.3e5 < config.pbuff2 < 73.8e5:
        config.pbuff2 = 73.3e5
    config.hbuff2 = x[3 * (config.Nc + config.Ne) + 11]
    config.Tbuff2_wall = x[3 * (config.Nc + config.Ne) + 12]
    state = get_state(CP.HmassP_INPUTS, config.hbuff2, config.pbuff2)
    config.Xbuff2 = state.Q()
    config.Tbuff2 = state.T()
    config.Dbuff2 = state.rhomass()
    config.kbuff2 = state.conductivity()
    config.sbuff2 = state.smass()
    config.mubuff2 = state.viscosity()
    config.cpbuff2 = np.abs(state.cpmass())
    config.cvbuff2 = state.cvmass()
    config.mbuff2 = config.Dbuff2 * config.Vbuff2
    config.dDdp_hbuff2 = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pbuff2 = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    if 0 <= config.Xbuff2 <= 1:
        state = get_state(CP.QT_INPUTS, 0, config.Tbuff2)
        config.mubuff2_l = state.viscosity()
        state = get_state(CP.QT_INPUTS, 1, config.Tbuff2)
        config.mubuff2_v = state.viscosity()


def calc_buffer2(mbuff2_in_dot, mbuff2_out_dot, Dbuff2_in, mw1_dot_1, Tw_in):
    config.vbuff2_n = (mbuff2_in_dot + mbuff2_out_dot) / (2 * config.Dbuff2 * config.ACO2_orifice)

    Re = (np.abs(config.vbuff2_n)* config.dhbuff2* config.Dbuff2 / config.mubuff2)
    Pr = config.mubuff2 * config.cpbuff2 / config.kbuff2
    Nu = Nu_DB(config.pbuff2, config.Xbuff2, Re, Pr, config.dhbuff2, config.dxbuff2, config.mubuff2, config.mu_cu, config.mubuff2_l, config.mubuff2_v)
    
    config.Ubuff2 = config.kbuff2 * Nu / config.dhbuff2
    config.Qbuff2_conv = (config.Abuff2* config.Ubuff2 * (config.Tbuff2_wall - config.Tbuff2))

    Rew = (np.abs(mw1_dot_1) * config.dhbuff2 / (config.Abuff2_orifice * config.muw))
    Prw = config.muw * config.cpw / config.kw
    Nu_w = 0.023 * Rew**0.8 * Prw ** (0.4)
    config.Ubuff2_w = config.kw * Nu_w / config.dhbuff2
    config.Qbuff2_w_conv = (config.Abuff2* config.Ubuff2_w * (config.Tbuff2_wall - Tw_in) )

    return config.Qbuff2_conv, config.Qbuff2_w_conv

def model_buffer2(mbuff2_in_dot, mbuff2_out_dot, hbuff2_in, mw1_dot_1):
    # z11 = (config.dDdp_hbuff2 * config.hbuff2 - 1) * config.Vbuff2
    # z12 = (
    #     config.dDdh_pbuff2 * config.hbuff2 + config.Dbuff2
    # ) * config.Vbuff2
    z11 = (- 1) * config.Vbuff2
    z12 = (config.Dbuff2) * config.Vbuff2
    z21 = config.dDdp_hbuff2 * config.Vbuff2
    z22 = config.dDdh_pbuff2 * config.Vbuff2
    z33 = (config.mbuff2_wall * config.c_cu)
    Z = np.array([[z11, z12, 0], [z21, z22, 0], [0, 0, z33]])

    Z_inv = np.linalg.inv(Z)
    f1 = config.Qbuff2_conv + mbuff2_in_dot * (hbuff2_in - config.hbuff2)
    f2 = mbuff2_in_dot - mbuff2_out_dot
    f3 = -(config.Qbuff2_w_conv + config.Qbuff2_conv)
    f_vector = np.array([f1, f2, f3])
    result_vector = np.dot(Z_inv, f_vector)
    config.dpbuff2dt = result_vector[0]
    config.dhbuff2dt = result_vector[1]
    config.dTbuff2_walldt = result_vector[2]

    return config.dpbuff2dt, config.dhbuff2dt, config.dTbuff2_walldt

# post processing

def init_buffer2_post(y, k, pbuff2_in, pe):
    config.pbuff2 = y[3 * (config.Nc + config.Ne) + 10,  k] # pbuff2_in #
    if 73.3e5 < config.pbuff2 < 73.8e5:
        config.pbuff2 = 73.3e5
    config.hbuff2 = y[3 * (config.Nc + config.Ne) + 11, k]
    config.Tbuff2_wall = y[3 * (config.Nc + config.Ne) + 12, k]
    state = get_state(CP.HmassP_INPUTS, config.hbuff2, config.pbuff2)
    config.Xbuff2 = state.Q()
    config.Tbuff2 = state.T()
    config.Dbuff2 = state.rhomass()
    config.kbuff2 = state.conductivity()
    config.sbuff2 = state.smass()
    config.mubuff2 = state.viscosity()
    config.cpbuff2 = np.abs(state.cpmass())
    config.cvbuff2 = state.cvmass()
    config.mbuff2 = config.Dbuff2 * config.Vbuff2
    config.dDdp_hbuff2 = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pbuff2 = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    if 0 <= config.Xbuff2 <= 1:
        state = get_state(CP.QT_INPUTS, 0, config.Tbuff2)
        config.mubuff2_l = state.viscosity()
        state = get_state(CP.QT_INPUTS, 1, config.Tbuff2)
        config.mubuff2_v = state.viscosity()