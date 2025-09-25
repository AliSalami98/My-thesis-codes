import config
from utils import get_state, Nu_evap_DB
from config import CP
import numpy as np

def init_evaporator(x, pe_in, he_in, p25):
    config.pe = min(max(x[0], 20e5), 50e5) #pe_in  #
    for i in range(config.Ne):
        config.he[i] = min(max(x[i + 1], he_in + 20e3), 600e3) #x[i + 1] #
        config.Te_wall[i] = x[i + 1 + config.Ne]
        config.Tmpg[i] = x[i + 1 + 2 * config.Ne]
        # try:
        state = get_state(CP.HmassP_INPUTS, config.he[i], config.pe)
        config.Xe[i] = state.Q()
        config.Te[i] = state.T()
        config.De[i] = state.rhomass()
        config.ke[i] = state.conductivity()
        config.se[i] = state.smass()
        config.mue[i] = state.viscosity()
        config.cpe[i] = np.abs(state.cpmass())
        config.cve[i] = state.cvmass()
        config.me[i] = config.De[i] * config.Ve[i]
        config.dDdp_he[i] = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
        config.dDdh_pe[i] = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
        if 0 < config.Xe[i] < 1:
            config.dDdh_pe[i] = state.first_two_phase_deriv_splined(CP.iDmass, CP.iHmass, CP.iP, config.Xe[i])
            config.dDdp_he[i] = state.first_two_phase_deriv_splined(CP.iDmass, CP.iP, CP.iHmass, config.Xe[i])
            state = get_state(CP.QT_INPUTS, 0, config.Te[i])
            config.mue_l[i] = state.viscosity()
            state = get_state(CP.QT_INPUTS, 1, config.Te[i])
            config.mue_v[i] = state.viscosity()
            config.he_v[i] = state.hmass()

        
def calc_evaporator(me_in_dot, me_out_dot, De_in, mmpg_dot, Tmpg_in):
    for i in range(config.Ne + 1):
        if i == 0:
            config.De_m[i] = (De_in + config.De[i]) / 2
            config.ve[i] = me_out_dot / (config.De_m[i] * config.ACO2_orifice)
        elif i == config.Ne:
            config.De_m[i] = (config.De[i-1])
            config.ve[i] = me_out_dot / (config.De_m[i] * config.ACO2_orifice)   
        else:
            config.De_m[i] = (config.De[i-1]+ config.De[i]) / 2
            config.ve[i] = me_out_dot / (config.De_m[i] * config.ACO2_orifice)
        # config.he_m[i] = (config.he[i] + config.he[i + 1]) / 2
        # config.pe_m[i] = (config.pe + config.pe[i + 1]) / 2
        # config.Te_m[i] = (config.Te[i] + config.Te[i + 1]) / 2
        # config.me_m[i] = (config.me[i] + config.me[i + 1]) / 2
        # config.mue_m[i] = (config.mue[i] + config.mue[i + 1]) / 2
        # config.ke_m[i] = (config.ke[i] + config.ke[i + 1]) / 2
        # config.cpe_m[i] = (config.cpe[i] + config.cpe[i + 1]) / 2

    for i in range(config.Ne):
        config.ve_n[i] = (config.ve[i] + config.ve[i+1]) / 2

        Re = np.abs(config.ve_n[i]) * config.dhe[i] * config.De[i] / config.mue[i]
        Pr = config.mue[i] * config.cpe[i] / config.ke[i]
        Nu = Nu_evap_DB(config.pe, config.Xe[i], Re, Pr, config.dhe[i], config.dxe[i], config.mue[i], config.mu_cu, config.mue_l[i], config.mue_v[i])

        config.Ue[i] = config.ke[i] * Nu / config.dhe[i]
        config.Qe_conv[i] = max(
            config.Ae[i] * config.Ue[i] * (config.Te_wall[i] - config.Te[i])
        , 0)

        Re_mpg = (
            np.abs(mmpg_dot) * config.dh_mpg / (config.A_mpg_orifice * config.mu_mpg)
        )
        Pr_mpg = config.mu_mpg * config.cp_mpg / config.k_mpg
        Nu_mpg = 0.023 * Re_mpg**0.8 * Pr_mpg ** (0.4)
        U_mpg = config.k_mpg * Nu_mpg / config.dh_mpg
        # print(U_mpg)
        config.Qmpg_conv[i] = min(
            config.Ae[i] * U_mpg * (config.Te_wall[i] - config.Tmpg[i])
        , 0)
def model_evaporator(me_in_dot, me_out_dot, he_in, mmpg_dot, cp_mpg):

    xe = np.hstack(([config.pe], config.he, config.Te_wall))

    # Define the Z_e matrix (size: Ne+1 x Ne+1)
    Ze = np.zeros((2 * config.Ne + 1, 2 * config.Ne + 1))

    # First row (pressure-related terms)
    Ze[0, 0] = config.dDdp_he[0] * config.Ve[0]
    Ze[0, 1] = config.dDdh_pe[0] * config.Ve[0]

    # Enthalpy-related terms
    for i in range(1, config.Ne + 1):
        Ze[i, 0] = -1 * config.Ve[i-1]
        Ze[i, i] = config.De[i-1] * config.Ve[i-1]
    for i in range(config.Ne + 1, 2 * config.Ne + 1):
        Ze[i, i] = (config.me_wall[i - (config.Ne + 1)] * config.c_ss)

    f_vec = np.zeros(2 * config.Ne + 1)
    f_vec[0] = me_in_dot - me_out_dot

    # Enthalpy equations
    f_vec[1] = config.Qe_conv[0] + me_out_dot * (he_in - xe[1])
    for i in range(2, config.Ne + 1):
        f_vec[i] = config.Qe_conv[i-1] + me_out_dot * (xe[i-1] - xe[i])
    for i in range(config.Ne + 1, 2 * config.Ne + 1):
        f_vec[i] = -(config.Qmpg_conv[i - (config.Ne + 1)] + config.Qe_conv[i - (config.Ne + 1)])

    Ze_inv = np.linalg.inv(Ze)  # Inverse of Z_e matrix
    dxedt = np.dot(Ze_inv, f_vec)

    # Assign results back to configuration
    config.dpedt = dxedt[0]  # Pressure time derivative
    config.dhedt = dxedt[1:config.Ne + 1].tolist()  # Enthalpy time derivatives
    config.dTe_walldt = dxedt[config.Ne + 1:].tolist()

def init_evaporator_post(y, k, pe_in, he_in, p25):
    config.pe = min(max(y[0, k], 20e5), 50e5) #pe_in  #
    for i in range(config.Ne):
        config.he[i] = min(max(y[i + 1, k], he_in + 20e3), 600e3) #y[i + 1, k] #
        config.Te_wall[i] = y[i + 1 + config.Ne, k]
        config.Tmpg[i] = y[i + 1 + 2 * config.Ne, k]
        state = get_state(CP.HmassP_INPUTS, config.he[i], config.pe)
        config.Xe[i] = state.Q()
        config.Te[i] = state.T()
        config.De[i] = state.rhomass()
        config.ke[i] = state.conductivity()
        config.se[i] = state.smass()
        config.mue[i] = state.viscosity()
        config.cpe[i] = np.abs(state.cpmass())
        config.cve[i] = state.cvmass()
        config.me[i] = config.De[i] * config.Ve[i]
        config.dDdp_he[i] = state.first_partial_deriv(CP.iDmass, CP.iP, CP.iHmass)
        config.dDdh_pe[i] = state.first_partial_deriv(CP.iDmass, CP.iHmass, CP.iP)
        if 0 < config.Xe[i] < 1:
            config.dDdh_pe[i] = state.first_two_phase_deriv_splined(CP.iDmass, CP.iHmass, CP.iP, config.Xe[i])
            config.dDdp_he[i] = state.first_two_phase_deriv_splined(CP.iDmass, CP.iP, CP.iHmass, config.Xe[i])
            state = get_state(CP.QT_INPUTS, 0, config.Te[i])
            config.mue_l[i] = state.viscosity()
            state = get_state(CP.QT_INPUTS, 1, config.Te[i])
            config.mue_v[i] = state.viscosity()
            config.he_v[i] = state.hmass()
