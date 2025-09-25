import numpy as np

from utils import get_state
from config import CP
import config
from sklearn.preprocessing import PolynomialFeatures

# import os
# import joblib

# GP_models_dir = 'ML_Models/GP_models'
# models_GP = [joblib.load(os.path.join(GP_models_dir, f'GP_model_{i+1}.pkl')) for i in range(3)]

# scalers_dir = 'ML_Models/scalers'
# scaler_X = joblib.load(os.path.join(scalers_dir, 'scaler_X.pkl'))
# scalers_y = [joblib.load(os.path.join(scalers_dir, f'scaler_y_{i+1}.pkl')) for i in range(3)]

from funcs import (
    init_evaporator,
    init_gas_cooler,
    init_buffer1,
    init_buffer2,
    init_internal_heat_exchanger,
    init_flash_tank,

    calc_evaporator,
    calc_gas_cooler,
    calc_buffer1,
    calc_buffer2,
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
    mc_in_dot,
    mc_out_dot,
    me_in_dot,
    me_out_dot,
    mft_l_dot,
    mft_v_dot,
    mtc2_dot,
    Ttc2_out,
    hc_in,
    Dc_in,
    hbuff1_in,
    Dbuff1_in
):
    j = int(np.floor(t))
    pbuff1_in = config.pbuff1
    pbuff2_in = config.pbuff2
    pft_in = config.pft
    pc_in = config.pc
    pe_in = config.pe
    De_in = config.De[0]
    mw2_dot = 0.7 * mw_dot
    mw1_dot = 0.3 * mw_dot
    mw1_dot_1 = 0.5 * 0.3 * mw_dot
    mw1_dot_2 = mw1_dot_1

    state = get_state(CP.PT_INPUTS, pbuff2_in, Ttc2_out)
    htc2_out = state.hmass()
    hbuff2_in = max(htc2_out, config.hbuff2_v)
    state = get_state(CP.HmassP_INPUTS, hbuff2_in, pbuff2_in)
    Dbuff2_in = state.rhomass()

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
    he_in = config.hft_l
    
    # mft_v_dot = mc_out_dot - me_in_dot
    mft_in_dot = mc_out_dot

    
    mbuff1_in_dot = mft_v_dot + me_out_dot
    mbuff1_out_dot = mc_in_dot

    mbuff2_in_dot = mtc2_dot
    mbuff2_out_dot = mc_in_dot

    mihx1_in_dot = mc_in_dot
    mihx1_out_dot = mihx1_in_dot

    mihx2_in_dot = me_out_dot
    mihx2_out_dot = mihx2_in_dot

    init_gas_cooler(x, pc_in, hc_in)
    calc_gas_cooler(mc_in_dot, mc_out_dot, Dc_in, mw2_dot, Tw_in)

    init_evaporator(x, pe_in, he_in, config.pbuff1)
    calc_evaporator(me_in_dot, me_out_dot, De_in, mmpg_dot, Tmpg_in)
    
    
    init_buffer1(x, pbuff1_in, config.pe)
    init_buffer2(x, pbuff2_in, config.pbuff1)
    init_internal_heat_exchanger(x, pihx1_in, pihx2_in, hihx1_in, hihx2_in, config.he[-1], config.hc[-1])
    init_flash_tank(x, pft_in, config.pe)

    calc_buffer1(mbuff1_in_dot, mbuff1_out_dot, Dbuff1_in, mw1_dot_1, Tw_in, me_out_dot)
    calc_buffer2(mbuff2_in_dot, mbuff2_out_dot, Dbuff2_in, mw1_dot_2, Tw_in)
    calc_internal_heat_exchanger(mihx1_in_dot, mihx1_out_dot, Dihx1_in, mihx2_in_dot, mihx2_out_dot, Dihx2_in)
    # print(config.Qihx2_conv)
    combined_state_space_model(
        me_in_dot, me_out_dot, he_in, mmpg_dot, Tmpg_in,
        mc_in_dot, mc_out_dot, hc_in, mw2_dot, Tw_in,
        mbuff1_in_dot, mbuff1_out_dot, hbuff1_in, mw1_dot_1,
        mbuff2_in_dot, mbuff2_out_dot, hbuff2_in, mw1_dot_2,
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
            config.dpbuff1dt, config.dhbuff1dt, config.dTbuff1_walldt,config.dpbuff2dt, config.dhbuff2dt, config.dTbuff2_walldt,
            config.dpftdt, config.dhftdt]
    )
    return dxdt