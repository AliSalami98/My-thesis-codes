import config
from utils import get_state, Nu_ihx_DB, Nu_evap_DB, Nu_gc_DB
from config import CP
import numpy as np


def init_internal_heat_exchanger(x, pihx1_in, pihx2_in, hihx1_in, hihx2_in, he_out, hc_out):
    config.pihx1 = pihx1_in #x[4*(config.Ne+config.Nc) + config.Nihx - 1]
    config.hihx1 = max(min(x[3 * (config.Nc + config.Ne) + 3], hc_out), config.hft_l)
    state = get_state(CP.HmassP_INPUTS, config.hihx1, config.pihx1)
    config.Xihx1 = state.Q()
    config.Tihx1 = state.T()
    config.Dihx1 = state.rhomass()
    config.kihx1 = state.conductivity()
    config.sihx1 = state.smass()
    config.muihx1 = state.viscosity()
    config.cpihx1 = np.abs(state.cpmass())
    config.cvihx1 = state.cvmass()
    config.mihx1 = config.Dihx1 * config.Vihx1
    config.dDdp_hihx1 = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pihx1 = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    if 0 <= config.Xihx1 <= 1:
        config.dDdp_hihx1 = state.first_two_phase_deriv(CP.iDmass, CP.iP, CP.iHmass)
        config.dDdh_pihx1 = state.first_two_phase_deriv(CP.iDmass, CP.iHmass, CP.iP)
        state = get_state(CP.QT_INPUTS, 0, config.Tihx1)
        config.muihx1_l = state.viscosity()
        state = get_state(CP.QT_INPUTS, 1, config.Tihx1)
        config.muihx1_v = state.viscosity()

    config.Tihx_wall = x[3 * (config.Nc + config.Ne) + 4]

    config.pihx2 = pihx2_in #x[4 * (config.Ne + config.Nc) + 4 * config.Nihx - 1]
    config.hihx2 = max(x[3 * (config.Ne + config.Nc) + 6], he_out)
    state = get_state(CP.HmassP_INPUTS, config.hihx2, config.pihx2)
    config.Xihx2 = state.Q()
    config.Tihx2 = state.T()
    config.Dihx2 = state.rhomass()
    config.kihx2 = state.conductivity()
    config.sihx2 = state.smass()
    config.muihx2 = state.viscosity()
    config.cpihx2 = np.abs(state.cpmass())
    config.cvihx2 = state.cvmass()
    config.mihx2 = config.Dihx2 * config.Vihx2
    config.dDdp_hihx2 = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pihx2 = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    if 0 <= config.Xihx2 <= 1:
        config.dDdp_hihx2 = state.first_two_phase_deriv(CP.iDmass, CP.iP, CP.iHmass)
        config.dDdh_pihx2 = state.first_two_phase_deriv(CP.iDmass, CP.iHmass, CP.iP)
        state = get_state(CP.QT_INPUTS, 0, config.Tihx2)
        config.muihx2_l = state.viscosity()
        state = get_state(CP.QT_INPUTS, 1, config.Tihx2)
        config.muihx2_v = state.viscosity()


def calc_internal_heat_exchanger(mihx1_in_dot, mihx1_out_dot, Dihx1_in, mihx2_in_dot, mihx2_out_dot, Dihx2_in):
    config.Dihx1_m = (Dihx1_in + config.Dihx1) / 2
    config.vihx1 = (mihx1_in_dot + mihx1_out_dot) / (2 * config.Dihx1_m * config.ACO2_orifice)
    config.Dihx2_m = (Dihx2_in + config.Dihx2) / 2
    config.vihx2 = (mihx2_in_dot + mihx2_out_dot)/ (2 * config.Dihx2_m * config.ACO2_orifice)


    Re = (np.abs(config.vihx1)* config.dhihx1* config.Dihx1 / config.muihx1)
    Pr = config.muihx1 * config.cpihx1 / config.kihx1
    Nu = Nu_gc_DB(config.pihx1, config.Xihx1, Re, Pr, config.dhihx1, config.dxihx1, config.muihx1, config.mu_cu, config.Tihx1, config.Dihx1)

    config.Uihx1 = config.kihx1 * Nu / config.dhihx1
    config.Qihx1_conv = (config.Aihx1 * config.Uihx1 * (config.Tihx_wall - config.Tihx1))
    
    Re = (np.abs(config.vihx2)* config.dhihx2 * config.Dihx2/ config.muihx2)
    Pr = config.muihx2 * config.cpihx2 / config.kihx2
    Nu = Nu_evap_DB(config.pihx2, config.Xihx2, Re, Pr, config.dhihx2, config.dxihx2, config.muihx2, config.mu_cu, config.muihx2_l, config.muihx2_v)
    
    config.Uihx2 = config.kihx2 * Nu / config.dhihx2
    config.Qihx2_conv = (config.Aihx2 * config.Uihx2 * (config.Tihx_wall - config.Tihx2))
    return config.Qihx1_conv, config.Qihx2_conv


def model_internal_heat_exchanger(mihx1_in_dot, mihx1_out_dot, hihx1_in, mihx2_in_dot, mihx2_out_dot, hihx2_in):
    z11 = (- 1) * config.Vihx1
    z12 = (config.Dihx1) * config.Vihx1
    z21 = config.dDdp_hihx1 * config.Vihx1
    z22 = config.dDdh_pihx1 * config.Vihx1
    z33 = (config.mihx_wall * config.c_cu)
    z44 = (- 1) * config.Vihx2
    z45 = (config.Dihx2) * config.Vihx2
    z54 = config.dDdp_hihx2 * config.Vihx2
    z55 = config.dDdh_pihx2 * config.Vihx2
    Z = np.array([[z11, z12, 0, 0, 0], [z21, z22, 0, 0, 0], [0, 0, z33, 0, 0], [0, 0, 0, z44, z45], [0, 0, 0, z54, z55]])
    Z_inv = np.linalg.inv(Z)
    f1 = config.Qihx1_conv + mihx1_in_dot * (hihx1_in - config.hihx1)
    f2 = mihx1_in_dot - mihx1_out_dot
    f3 = -(config.Qihx1_conv + config.Qihx2_conv)
    f4 = config.Qihx2_conv + mihx2_out_dot * (hihx2_in - config.hihx2)
    f5 = mihx2_in_dot - mihx2_out_dot
    f_vector = np.array([f1, f2, f3, f4, f5])
    result_vector = np.dot(Z_inv, f_vector)
    config.dpihx1dt = result_vector[0]
    config.dhihx1dt = result_vector[1]
    config.dTihx_walldt = result_vector[2]
    config.dpihx2dt = result_vector[3]
    config.dhihx2dt = result_vector[4]
    
    return config.dpihx1dt, config.dhihx1dt, config.dTihx_walldt, config.dpihx2dt, config.dhihx2dt

# post processing

def init_internal_heat_exchanger_post(y, k, pihx1_in, pihx2_in, hihx1_in, hihx2_in):
    config.pihx1 = pihx1_in #y[4*(config.Ne+config.Nc) + config.Nihx - 1, k]
    config.hihx1 = y[3 * (config.Ne + config.Nc) + 3, k]
    state = get_state(CP.HmassP_INPUTS, config.hihx1, config.pihx1)
    config.Xihx1 = state.Q()
    config.Tihx1 = state.T()
    config.Dihx1 = state.rhomass()
    config.kihx1 = state.conductivity()
    config.sihx1 = state.smass()
    config.muihx1 = state.viscosity()
    config.cpihx1 = np.abs(state.cpmass())
    config.cvihx1 = state.cvmass()
    config.mihx1 = config.Dihx1 * config.Vihx1
    config.dDdp_hihx1 = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pihx1 = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    if 0 <= config.Xihx1 <= 1:
        config.dDdp_hihx1 = state.first_two_phase_deriv(CP.iDmass, CP.iP, CP.iHmass)
        config.dDdh_pihx1 = state.first_two_phase_deriv(CP.iDmass, CP.iHmass, CP.iP)
        state = get_state(CP.QT_INPUTS, 0, config.Tihx1)
        config.muihx1_l = state.viscosity()
        state = get_state(CP.QT_INPUTS, 1, config.Tihx1)
        config.muihx1_v = state.viscosity()

    config.Tihx_wall = y[3 * (config.Nc + config.Ne) + 4, k]

    config.pihx2 = pihx2_in #y[4 * (config.Ne + config.Nc) + 4 * config.Nihx - 1, k]
    config.hihx2 = y[3 * (config.Nc + config.Ne) + 6, k]
    state = get_state(CP.HmassP_INPUTS, config.hihx2, config.pihx2)
    config.Xihx2 = state.Q()
    config.Tihx2 = state.T()
    config.Dihx2 = state.rhomass()
    config.kihx2 = state.conductivity()
    config.sihx2 = state.smass()
    config.muihx2 = state.viscosity()
    config.cpihx2 = np.abs(state.cpmass())
    config.cvihx2 = state.cvmass()
    config.mihx2 = config.Dihx2 * config.Vihx2
    config.dDdp_hihx2 = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pihx2 = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    if 0 <= config.Xihx2 <= 1:
        config.dDdp_hihx2 = state.first_two_phase_deriv(CP.iDmass, CP.iP, CP.iHmass)
        config.dDdh_pihx2 = state.first_two_phase_deriv(CP.iDmass, CP.iHmass, CP.iP)
        state = get_state(CP.QT_INPUTS, 0, config.Tihx2)
        config.muihx2_l = state.viscosity()
        state = get_state(CP.QT_INPUTS, 1, config.Tihx2)
        config.muihx2_v = state.viscosity()
