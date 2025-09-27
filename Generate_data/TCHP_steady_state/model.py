import numpy as np

from utils import get_state
from config import CP
import config
from sklearn.preprocessing import PolynomialFeatures

from funcs import (
    init_evaporator,
    init_gas_cooler,
    init_buffer1,
    init_internal_heat_exchanger,
    init_flash_tank,

    calc_evaporator,
    calc_gas_cooler,
    calc_buffer1,
    calc_internal_heat_exchanger,

    model_evaporator,
    model_gas_cooler,
    model_buffer1,
    model_internal_heat_exchanger,
    model_flash_tank,
    combined_state_space_model
)
# from ML_Models.compressor_PR import (
#     poly1_reg,
#     model1_PR,
#     poly2_reg,
#     model2_PR,
#     poly3_reg,
#     model3_PR,
# )
from ML_Models.comps import (
    model1_1_LR,
    model1_2_LR,
    model1_3_LR,
    model2_1_LR,
    model2_2_LR,
    model2_3_LR,
    model3_1_LR,
    model3_2_LR,
    model3_3_LR,
)
# from PR_compressor2 import (
#     model1_comp2_PR,
#     model2_comp2_PR,
#     model3_comp2_PR,
# )
# from PR_compressor3 import (
#     model1_comp3_PR,
#     model2_comp3_PR,
#     model3_comp3_PR,
# )
# from compressor_ANN import (
#     model1_ANN
# )
def cycle(
    t,
    x,
    mw_dot,
    Tw_in,
    Tmpg_in,
    mmpg_dot,
    Theater1,
    Theater2,
    omega1,
    omega2,
    omega3,
    Lpev,
    Hpev,
    omegab,
    tau_LPV,
    tau_HPV
):
    pbuff1_in = config.pbuff1
    pft_in = config.pft
    pc_in = config.pc
    pe_in = config.pe
    De_in = config.De[0]
    mw2_dot = 0.7 * mw_dot
    mw1_dot_1 = 0.5 * 0.3 * mw_dot
    mw1_dot_2 = mw1_dot_1
    lpev = x[3 * (config.Nc + config.Ne) + 12]
    hpev = x[3 * (config.Nc + config.Ne) + 13]

    # print(t)
    
    f_hpv = 8e-9 * hpev #min(max(4.765e-9 * hpev + 6.3625e-8, 1e-7), 4.5e-7) #8e-9 * hpev #
    f_lpv = 4e-9 * lpev # min(max(6.023e-9 * lpev - 1e-7, 0.2e-7), 3e-7)  #4e-9 * lpev #

    Tr1 = Theater1/Tw_in
    Pr1 = config.pbuff1/config.pe

    a1 = np.array([[Pr1], [Theater1], [Tw_in], [omega1]])
    a1_T = a1.T

    he_in = config.hft_l

    Pr23 = config.pc/config.pbuff1
    a3 = np.array([[Pr23], [Theater2], [Tw_in], [omega2]])
    a3_T = a3.T
    Tc_in = model3_2_LR.predict(a3_T)[0] #(3.46e-3 * omegab + 7.157e1 * Pr3 + 230.33) #3(- 6.99e-3 * omegab - 1.656e1 * Pr + 7.47e-3 * omegab * Pr + 330.65) #

    state = get_state(CP.PT_INPUTS, config.pc, Tc_in)
    hc_in = state.hmass()
    Dc_in = state.rhomass()

    mc_in_dot = 10**(-3) * model3_1_LR.predict(a3_T)[0] #10**(-3) * (5.73e-3 * omegab - 6.5657e1 * Pr3 + 59.877)
    mc_out_dot = f_hpv * np.sqrt(2*config.Dc[-1] * max(config.pc - config.pft, 0)) #0.000327 * Hpev #

    me_out_dot = min(10**(-3) * model1_1_LR.predict(a1_T)[0], mc_in_dot) #10**(-3) * (3.843e-3 * omegab - 3.99e+1 * Pr1 + 2.18e-2 * omega1 + 42.36)

    dlpevdt = (Lpev - lpev)/tau_LPV
    dhpevdt = (Hpev - hpev)/tau_HPV

    if config.hft_l > config.hft:
        # mft_l_dot = 5 * 10**(-9) * lpev * np.sqrt(2*config.Dft * max(config.pft - config.pe, 0)) #0.000145 * Lpev  #
        mft_l_dot = f_lpv * np.sqrt(2*config.Dft * max(config.pft - config.pe, 0)) #0.000145 * Lpev  #
        mft_v_dot = 1e-6
        # print('r')
    if config.hft > config.hft_v:
        mft_l_dot = 1e-6 
        mft_v_dot = 1e-6 * (((hpev - lpev) + 90)/(2 * 90)) * np.sqrt(2*config.Dft * max(config.pft - config.pbuff1, 0)) #max(mbuff2_in_dot - me_out_dot, 0)
    else:
        # mft_l_dot = 5 * 10**(-9) * lpev * np.sqrt(2*config.Dft_l * max(config.pft - config.pe, 0)) #0.000145 * Lpev  # 
        mft_l_dot = f_lpv * np.sqrt(2*config.Dft_l * max(config.pft - config.pe, 0)) #0.000145 * Lpev  # 
        mft_v_dot = 1e-6 * (((hpev - lpev) + 90)/(2 * 90)) * np.sqrt(2*config.Dft_v * max(config.pft - config.pbuff1, 0)) #max(mbuff2_in_dot - me_out_dot, 0.0001)

    me_in_dot = mft_l_dot
    # mft_v_dot = mc_out_dot - me_in_dot
    mft_in_dot = mc_out_dot

    mbuff1_in_dot = mft_v_dot + me_out_dot
    mbuff1_out_dot = mc_in_dot

    mihx1_in_dot = mc_in_dot
    mihx1_out_dot = mihx1_in_dot

    mihx2_in_dot = me_out_dot
    mihx2_out_dot = mihx2_in_dot

    TTC1_out = max(model1_2_LR.predict(a1_T)[0], 300) # model2_PR.predict(poly2_reg.transform(a1_T))[0]
    state = get_state(CP.PT_INPUTS, pbuff1_in, TTC1_out)
    
    hTC1_out = state.hmass()
    hbuff1_in = hTC1_out #min(max((mft_v_dot * config.hft_v + me_out_dot * hTC1_out)/(mft_v_dot + me_out_dot), config.hft_v), hTC1_out)
    # print(hbuff1_in)
    state = get_state(CP.HmassP_INPUTS, hbuff1_in, pbuff1_in)

    Tbuff1_in = state.T()
    Dbuff1_in = state.rhomass()

    hihx1_in = config.hc[-1]
    pihx1_in = config.pc
    state = get_state(CP.HmassP_INPUTS, hihx1_in, pihx1_in)
    Dihx1_in = state.rhomass()
    Tihx1_in = state.T()

    hihx2_in = config.he[-1]
    pihx2_in = config.pe
    state = get_state(CP.HmassP_INPUTS, hihx2_in, pihx2_in)
    Dihx2_in = state.rhomass()
    Tihx2_in = state.T()
    # mbuff2_in_dot =  10**(-3) * (5.66e-3 * omegab - 4.6e+1 * Pr1 + 1.23e-2 * omega1 + 50.3)
    hft_in = config.hihx1


    init_gas_cooler(x, pc_in, hc_in)
    calc_gas_cooler(mc_in_dot, mc_out_dot, Dc_in, mw2_dot, Tw_in)

    init_evaporator(x, pe_in, he_in, config.pbuff1)
    calc_evaporator(me_in_dot, me_out_dot, De_in, mmpg_dot, Tmpg_in)
    
    
    init_buffer1(x, pbuff1_in, config.pe)
    init_internal_heat_exchanger(x, pihx1_in, pihx2_in, hihx1_in, hihx2_in, config.he[-1], config.hc[-1])
    init_flash_tank(x, pft_in, config.pe)

    calc_buffer1(mbuff1_in_dot, mbuff1_out_dot, Dbuff1_in, mw1_dot_1, Tw_in)
    calc_internal_heat_exchanger(mihx1_in_dot, mihx1_out_dot, Dihx1_in, mihx2_in_dot, mihx2_out_dot, Dihx2_in)

    combined_state_space_model(
        me_in_dot, me_out_dot, he_in, mmpg_dot, Tmpg_in,
        mc_in_dot, mc_out_dot, hc_in, mw2_dot, Tw_in,
        mbuff1_in_dot, mbuff1_out_dot, hbuff1_in, mw1_dot_1,
        mihx1_in_dot, mihx1_out_dot, hihx1_in, mihx2_in_dot, mihx2_out_dot, hihx2_in,
        mft_in_dot, mft_l_dot, mft_v_dot, hft_in
    )
    dxdt = (
        [config.dpedt]
        + config.dhedt
        + config.dTe_walldt
        + config.dTmpgdt
        + [config.dpcdt]
        + config.dhcdt
        + config.dTc_walldt
        + config.dTc_wdt
        + [config.dpihx1dt, config.dhihx1dt, config.dTihx_walldt, config.dpihx2dt, config.dhihx2dt,
            config.dpbuff1dt, config.dhbuff1dt, config.dTbuff1_walldt,
            config.dpftdt, config.dhftdt, 
            dlpevdt, dhpevdt]
    )
    return dxdt