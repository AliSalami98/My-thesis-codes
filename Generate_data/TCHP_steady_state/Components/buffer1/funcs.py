import config
from utils import get_state, Nu_DB
from config import CP
import numpy as np


def init_buffer1(x, pbuff1_in, pe):
    config.pbuff1 = x[3 * (config.Nc + config.Ne) + 7] # #pbuff1_in #
    config.hbuff1 = x[3 * (config.Nc + config.Ne) + 8]
    config.Tbuff1_wall = x[3 * (config.Nc + config.Ne) + 9]
    state = get_state(CP.HmassP_INPUTS, config.hbuff1, config.pbuff1)
    config.Xbuff1 = state.Q()
    config.Tbuff1 = state.T()
    config.Dbuff1 = state.rhomass()
    config.kbuff1 = state.conductivity()
    config.sbuff1 = state.smass()
    config.mubuff1 = state.viscosity()
    config.cpbuff1 = np.abs(state.cpmass())
    config.cvbuff1 = state.cvmass()
    config.mbuff1 = config.Dbuff1 * config.Vbuff1
    config.dDdp_hbuff1 = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pbuff1 = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    if 0 <= config.Xbuff1 <= 1:
        state = get_state(CP.QT_INPUTS, 0, config.Tbuff1)
        config.mubuff1_l = state.viscosity()
        state = get_state(CP.QT_INPUTS, 1, config.Tbuff1)
        config.mubuff1_v = state.viscosity()


def calc_buffer1(mbuff1_in_dot, mbuff1_out_dot, Dbuff1_in, mw1_dot_1, Tw_in):
    config.vbuff1_n = (mbuff1_in_dot + mbuff1_out_dot) / (2 * config.Dbuff1 * config.ACO2_orifice)

    Re = (np.abs(config.vbuff1_n)* config.dhbuff1* config.Dbuff1 / config.mubuff1)
    Pr = config.mubuff1 * config.cpbuff1 / config.kbuff1
    Nu = Nu_DB(config.pbuff1, config.Xbuff1, Re, Pr, config.dhbuff1, config.dxbuff1, config.mubuff1, config.mu_cu, config.mubuff1_l, config.mubuff1_v)
    
    config.Ubuff1 = config.kbuff1 * Nu / config.dhbuff1
    config.Qbuff1_conv = (config.Abuff1* config.Ubuff1 * (config.Tbuff1_wall - config.Tbuff1))

    Rew = (np.abs(mw1_dot_1) * config.dhbuff1 / (config.Abuff1_orifice * config.muw))
    Prw = config.muw * config.cpw / config.kw
    Nu_w = 0.023 * Rew**0.8 * Prw ** (0.4)
    config.Ubuff1_w = config.kw * Nu_w / config.dhbuff1
    config.Qbuff1_w_conv = (config.Abuff1* config.Ubuff1_w * (config.Tbuff1_wall - Tw_in) )

    return config.Qbuff1_conv, config.Qbuff1_w_conv

def model_buffer1(mbuff1_in_dot, mbuff1_out_dot, hbuff1_in, mw1_dot_1):
    # z11 = (config.dDdp_hbuff1 * config.hbuff1 - 1) * config.Vbuff1
    # z12 = (
    #     config.dDdh_pbuff1 * config.hbuff1 + config.Dbuff1
    # ) * config.Vbuff1
    z11 = (- 1) * config.Vbuff1
    z12 = (config.Dbuff1) * config.Vbuff1
    z21 = config.dDdp_hbuff1 * config.Vbuff1
    z22 = config.dDdh_pbuff1 * config.Vbuff1
    z33 = (config.mbuff1_wall * config.c_cu)
    Z = np.array([[z11, z12, 0], [z21, z22, 0], [0, 0, z33]])

    Z_inv = np.linalg.inv(Z)
    f1 = config.Qbuff1_conv + mbuff1_in_dot * (hbuff1_in - config.hbuff1)
    f2 = mbuff1_in_dot - mbuff1_out_dot
    f3 = -(config.Qbuff1_w_conv + config.Qbuff1_conv)
    f_vector = np.array([f1, f2, f3])
    result_vector = np.dot(Z_inv, f_vector)
    config.dpbuff1dt = result_vector[0]
    config.dhbuff1dt = result_vector[1]
    config.dTbuff1_walldt = result_vector[2]

    return config.dpbuff1dt, config.dhbuff1dt, config.dTbuff1_walldt

# post processing

def init_buffer1_post(y, k, pbuff1_in, pe):
    config.pbuff1 = y[3 * (config.Nc + config.Ne) + 7,  k] # pbuff1_in #
    config.hbuff1 = y[3 * (config.Nc + config.Ne) + 8, k]
    config.Tbuff1_wall = y[3 * (config.Nc + config.Ne) + 9, k]
    state = get_state(CP.HmassP_INPUTS, config.hbuff1, config.pbuff1)
    config.Xbuff1 = state.Q()
    config.Tbuff1 = state.T()
    config.Dbuff1 = state.rhomass()
    config.kbuff1 = state.conductivity()
    config.sbuff1 = state.smass()
    config.mubuff1 = state.viscosity()
    config.cpbuff1 = np.abs(state.cpmass())
    config.cvbuff1 = state.cvmass()
    config.mbuff1 = config.Dbuff1 * config.Vbuff1
    config.dDdp_hbuff1 = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pbuff1 = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    if 0 <= config.Xbuff1 <= 1:
        state = get_state(CP.QT_INPUTS, 0, config.Tbuff1)
        config.mubuff1_l = state.viscosity()
        state = get_state(CP.QT_INPUTS, 1, config.Tbuff1)
        config.mubuff1_v = state.viscosity()