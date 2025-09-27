import config
from utils import get_state
from config import CP
import numpy as np


def init_flash_tank(x, pft_in, pe):
    config.pft = min(max(x[3 * (config.Nc + config.Ne) + 10], 20e5), 70e5) #config.pbuff1 #
    state = get_state(CP.PQ_INPUTS, config.pft, 0)
    config.muft_l = state.viscosity()
    config.hft_l = state.hmass()
    config.pft_l = state.p()
    config.Dft_l = state.rhomass()
    state = get_state(CP.PQ_INPUTS, config.pft, 1)
    config.muft_v = state.viscosity()
    config.hft_v = state.hmass()
    config.pft_v = state.p()
    config.Dft_v = state.rhomass()
    config.hft = min(max(x[3 * (config.Nc + config.Ne) + 11], config.hft_l + 20e3), config.hft_v - 20e3) #config.hihx1 
    # config.Tft_wall = x[4*(config.Ne+config.Nc+config.Nihx + config.Nbuff1 + config.Nbuff2) + config.Nihx + 2]
    state = get_state(CP.HmassP_INPUTS, config.hft, config.pft)
    config.Xft = state.Q()
    config.Tft = state.T()
    config.Dft= state.rhomass()
    config.kft = state.conductivity()
    config.sft = state.smass()
    config.muft= state.viscosity()
    config.cpft = np.abs(state.cpmass())
    config.cvft = state.cvmass()
    config.mft = config.Dft * config.Vft
    config.dDdp_hft = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pft = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    if config.hft < config.hft_l:
        config.hft_lo = config.hft
        config.hft_vo = 0
    elif config.hft > config.hft_v:
        config.hft_lo = 0
        config.hft_vo = config.hft
    else:
        config.hft_lo = (config.hft_l + config.hft)/2
        config.hft_vo = (config.hft_v + config.hft)/2
    # if config.Dft > config.Dft_v:
    #     config.hft_lo = (config.hft_l + config.hft)/2
    #     config.hft_vo = (config.hft_v + config.hft)/2
    # else:
    #     config.hft_lo = config.hft
    #     config.hft_vo = config.hft

def model_flash_tank(mft_in_dot, mft_l_dot, mft_v_dot, hft_in):
    # z11 = (config.dDdp_hft*config.hft - 1)*config.Vft
    # z12 = (config.dDdh_pft*config.hft + config.Dft)*config.Vft
    z11 = (- 1) * config.Vft
    z12 = (config.Dft) * config.Vft
    z21 = config.dDdp_hft*config.Vft
    z22 = config.dDdh_pft*config.Vft
    Z = np.array([[z11, z12],
                [z21, z22]])

    Z_inv = np.linalg.inv(Z)
    # if config.hft_l < config.hft < config.hft_v:
    f1 = (mft_in_dot*hft_in - mft_l_dot*config.hft_lo - mft_v_dot*config.hft_vo)
    # elif config.hft <= config.hft_l: 
    #     f1 = (mft_in_dot*hft_in - mft_l_dot*config.hft)
    # else:
    #     f1 = (mft_in_dot*hft_in - mft_v_dot*config.hft)
    f2 = (mft_in_dot - mft_l_dot - mft_v_dot)
    f_vector = np.array([f1, f2])
    result_vector = np.dot(Z_inv, f_vector)
    config.dpftdt = result_vector[0] #- 0.5 * config.pft
    config.dhftdt = result_vector[1]

    return config.dpftdt, config.dhftdt

# post processing

def init_flash_tank_post(y, k, pft_in, pe):
    config.pft = min(max(y[3 * (config.Nc + config.Ne) + 10, k], 20e5), 70e5) #cconfig.pbuff1 #
    state = get_state(CP.PQ_INPUTS, config.pft, 0)
    config.muft_l = state.viscosity()
    config.hft_l = state.hmass()
    config.pft_l = state.p()
    config.Dft_l = state.rhomass()
    state = get_state(CP.PQ_INPUTS, config.pft, 1)
    config.muft_v = state.viscosity()
    config.hft_v = state.hmass()
    config.pft_v = state.p()
    config.Dft_v = state.rhomass()
    config.hft = min(max(y[3 * (config.Nc + config.Ne) + 11, k], config.hft_l + 20e3), config.hft_v - 20e3) #config.hihx1 #
    # config.Tft_wall = y[4*(config.Ne+config.Nc+config.Nihx + config.Nbuff1 + config.Nbuff2) + 2, k]
    state = get_state(CP.HmassP_INPUTS, config.hft, config.pft)
    config.Xft = state.Q()
    config.Tft = state.T()
    config.Dft= state.rhomass()
    config.kft = state.conductivity()
    config.sft = state.smass()
    config.muft= state.viscosity()
    config.cpft = np.abs(state.cpmass())
    config.cvft = state.cvmass()
    config.mft = config.Dft * config.Vft
    config.dDdp_hft = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
    config.dDdh_pft = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
    # # if 0 < config.Xft < 1:
    #     config.dDdp_hft = state.first_two_phase_deriv(CP.iDmass, CP.iP, CP.iHmass)
    #     config.dDdh_pft = state.first_two_phase_deriv(CP.iDmass, CP.iHmass, CP.iP)

    #     Hliq = (config.Dft - config.Dft_v)/(config.Dft_l - config. Dft_v) * config.Lft
    #     rliq = Hliq/config.Lft
    #     if rliq >= 0.5:
    #         state = get_state(CP.PQ_INPUTS, config.pft, 0)
    #         config.muft_l = state.viscosity()
    #         config.hft_l = state.hmass()
    #         config.pft_l = state.p()
    #         config.Dft_l = state.rhomass()
    #         config.muft_v = config.muft_l
    #         config.hft_v = config.hft_l
    #         config.pft_v = config.pft_l
    #         config.Dft_v = config.Dft_l
    #     else:
    #         state = get_state(CP.PQ_INPUTS, config.pft, 1)
    #         config.muft_v = state.viscosity()
    #         config.hft_v = state.hmass()
    #         config.pft_v = state.p()
    #         config.Dft_v = state.rhomass()
    #         config.muft_l = config.muft_v 
    #         config.hft_l = config.hft_v
    #         config.pft_l = config.pft_v
    #         config.Dft_l = config.Dft_v

    # else:
    #     config.hft_l = config.hft
    #     config.hft_v = config.hft