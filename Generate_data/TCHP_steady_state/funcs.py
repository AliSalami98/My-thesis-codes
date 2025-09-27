from Components.evaporator.funcs import(
    init_evaporator,
    calc_evaporator,
    model_evaporator,

    init_evaporator_post,
)

from Components.gas_cooler.funcs import(
    init_gas_cooler,
    calc_gas_cooler,
    model_gas_cooler,

    init_gas_cooler_post,
)

from Components.internal_heat_exchanger.funcs import(
    init_internal_heat_exchanger,
    calc_internal_heat_exchanger,
    model_internal_heat_exchanger,

    init_internal_heat_exchanger_post,
)

from Components.buffer1.funcs import(
    init_buffer1,
    calc_buffer1,
    model_buffer1,

    init_buffer1_post,
)

from Components.flash_tank.funcs import(
    init_flash_tank,
    model_flash_tank,

    init_flash_tank_post
)

import config
import numpy as np
def predict_next_Theater1(X1_current, U1_current, D1_current, A1_coeffs, B1_coeffs, D1_coeffs, poly_U1):
    # Compute A_k (parameter-independent in this case)
    A1_k = A1_coeffs

    # Compute parameter-dependent B matrix
    input_vector = poly_U1.transform(U1_current.reshape(1, -1))  # Transform current input
    B1_k = B1_coeffs @ input_vector.T  # Input-dependent B


    D1_k = D1_coeffs
    # Compute next state (D remains independent)
    X1_next = (A1_k @ X1_current) + (B1_k.flatten()) + (D1_k @ D1_current)
    return X1_next

def predict_next_Theater2(X2_current, U2_current, D2_current, A2_coeffs, B2_coeffs, D2_coeffs, poly_U2):
    # Compute A_k (parameter-independent in this case)
    A2_k = A2_coeffs

    # Compute parameter-dependent B matrix
    input_vector = poly_U2.transform(U2_current.reshape(1, -1))  # Transform current input
    B2_k = B2_coeffs @ input_vector.T  # Input-dependent B


    D2_k = D2_coeffs
    # Compute next state (D remains independent)
    X2_next = (A2_k @ X2_current) + (B2_k.flatten()) + (D2_k @ D2_current)
    return X2_next
def combined_state_space_model(
    me_in_dot, me_out_dot, he_in, mmpg_dot, Tmpg_in,
    mc_in_dot, mc_out_dot, hc_in, mw2_dot, Tw_in,
    mbuff1_in_dot, mbuff1_out_dot, hbuff1_in, mw1_dot_1,
    mihx1_in_dot, mihx1_out_dot, hihx1_in, mihx2_in_dot, mihx2_out_dot, hihx2_in,
    mft_in_dot, mft_l_dot, mft_v_dot, hft_in
):

    xe = np.hstack(([config.pe], config.he, config.Te_wall, config.Tmpg))

    Ze = np.zeros((3 * config.Ne + 1, 3 * config.Ne + 1))

    # First row (pressure-related terms)
    Ze[0, 0] = np.mean(config.dDdp_he) * np.sum(config.Ve)
    Ze[0, 1] = config.dDdh_pe[0] * config.Ve[0]

    # Enthalpy-related terms
    for i in range(1, config.Ne + 1):
        Ze[i, 0] = -1 * np.sum(config.Ve)
        Ze[i, i] = config.De[i-1] * config.Ve[i-1]
    for i in range(config.Ne + 1, 2 * config.Ne + 1):
        Ze[i, i] = (config.me_wall[i - (config.Ne + 1)] * config.c_ss)
    for i in range(2 * config.Ne + 1, 3 * config.Ne + 1):
        Ze[i, i] = (config.m_mpg[i - (2 * config.Ne + 1)] * config.cp_mpg)

    fe_vec = np.zeros(3 * config.Ne + 1)
    fe_vec[0] = me_in_dot - me_out_dot

    # Enthalpy equations
    fe_vec[1] = config.Qe_conv[0] + me_out_dot * (he_in - xe[1])
    for i in range(2, config.Ne + 1):
        fe_vec[i] = config.Qe_conv[i-1] + me_out_dot * (xe[i-1] - xe[i])
    for i in range(config.Ne + 1, 2 * config.Ne + 1):
        fe_vec[i] = -(config.Qmpg_conv[i - (config.Ne + 1)] + config.Qe_conv[i - (config.Ne + 1)])

    for i in range(2 * config.Ne + 1, 3 * config.Ne):
        fe_vec[i] = (mmpg_dot * config.cp_mpg * (xe[i + 1] - xe[i]) + config.Qmpg_conv[i - (2 * config.Ne + 1)])

    fe_vec[-1] = mmpg_dot * config.cp_mpg * (Tmpg_in - xe[-1]) + config.Qmpg_conv[-1]

    Ze_inv = np.linalg.inv(Ze)  # Inverse of Z_e matrix
    dxedt = np.dot(Ze_inv, fe_vec)

    # Assign results back to configuration
    config.dpedt = dxedt[0]  # Pressure time derivative
    config.dhedt = dxedt[1:config.Ne + 1].tolist()  # Enthalpy time derivatives
    config.dTe_walldt = dxedt[config.Ne + 1:2 * config.Ne + 1].tolist()
    config.dTmpgdt = dxedt[2 * config.Ne + 1:].tolist()

    xc = np.hstack(([config.pc], config.hc, config.Tc_wall, config.Tc_w))  # The state vector consists of pressure and enthalpy values

    Zc = np.zeros((3 * config.Nc + 1, 3 * config.Nc + 1))

    Zc[0, 0] = np.mean(config.dDdp_hc) * np.sum(config.Vc)
    Zc[0, 1] = config.dDdh_pc[0] * config.Vc[0]
    
    for i in range(1, config.Nc + 1):
        Zc[i, 0] = -1 * np.sum(config.Vc)  # Pressure-related term
        Zc[i, i] = config.Vc[i-1] * config.Dc[i-1]  # Enthalpy-related ter

    for i in range(config.Nc + 1, 2 * config.Nc + 1):
        Zc[i, i] = config.mc_wall[i - (config.Nc + 1)] * config.c_ss

    for i in range(2 * config.Nc + 1, 3 * config.Nc + 1):
        Zc[i, i] = config.mc_w[i - (2 * config.Nc + 1)] * config.cp_w

    fc_vec = np.zeros(3 * config.Nc + 1)

    fc_vec[0] = mc_in_dot - mc_out_dot

    fc_vec[1] = config.Qc_conv[0] + mc_in_dot * (hc_in - xc[1])
    for i in range(2, config.Nc + 1):
        fc_vec[i] = config.Qc_conv[i-1] + mc_in_dot * (xc[i-1] - xc[i])
    # fc_vec[config.Nc] = config.Qc_conv[config.Nc - 1] + mc_in_dot * xc[config.Nc-1] - mc_out_dot * xc[config.Nc]
    for i in range(config.Nc + 1, 2 * config.Nc + 1):
        fc_vec[i] = -(config.Qc_w_conv[i - (config.Nc + 1)] + config.Qc_conv[i - (config.Nc + 1)])

    for i in range(2 * config.Nc + 1, 3 * config.Nc):
        fc_vec[i] = (mw2_dot * config.cp_w *  (xc[i + 1] - xc[i]) + config.Qc_w_conv[i - (2 * config.Nc + 1)])

    fc_vec[-1] = mw2_dot * config.cp_w * (Tw_in - xc[-1]) + config.Qc_w_conv[-1]

    Z_inv = np.linalg.inv(Zc)  # Inverse of Z_c matrix
    dxcdt = np.dot(Z_inv, fc_vec)

    config.dpcdt = dxcdt[0]
    config.dhcdt = dxcdt[1:config.Nc + 1].tolist()
    config.dTc_walldt = dxcdt[config.Nc + 1:2 * config.Nc + 1].tolist()
    config.dTc_wdt = dxcdt[2 * config.Nc + 1:].tolist()

    # xihx = np.hstack((config.pihx1, config.hihx1, config.Tihx_wall, config.pihx2, config.hihx2))

    # z11 = (- 1) * config.Vihx1
    # z12 = (config.Dihx1) * config.Vihx1
    # z21 = config.dDdp_hihx1 * config.Vihx1
    # z22 = config.dDdh_pihx1 * config.Vihx1
    # z33 = (config.mihx_wall * config.c_cu)
    # z44 = (- 1) * config.Vihx2
    # z45 = (config.Dihx2) * config.Vihx2
    # z54 = config.dDdp_hihx2 * config.Vihx2
    # z55 = config.dDdh_pihx2 * config.Vihx2
    # Zihx = np.array([[z11, z12, 0, 0, 0], [z21, z22, 0, 0, 0], [0, 0, z33, 0, 0], [0, 0, 0, z44, z45], [0, 0, 0, z54, z55]])

    # f1 = config.Qihx1_conv + mihx1_in_dot * (hihx1_in - xihx[1])
    # f2 = 0 #mihx1_in_dot - mihx1_out_dot
    # f3 = -(config.Qihx1_conv + config.Qihx2_conv)
    # f4 = config.Qihx2_conv + mihx2_out_dot * (hihx2_in - xihx[4])
    # f5 = 0 #mihx2_in_dot - mihx2_out_dot
    # fihx_vec = np.array([f1, f2, f3, f4, f5])

    # Zihx_inv = np.linalg.inv(Zihx)
    # dxihxdt = np.dot(Zihx_inv, fihx_vec)

    # config.dpihx1dt = dxihxdt[0]
    # config.dhihx1dt = dxihxdt[1]
    # config.dTihx_walldt = dxihxdt[2]
    # config.dpihx2dt = dxihxdt[3]
    # config.dhihx2dt = dxihxdt[4]


    xihx = np.hstack((config.hihx1, config.Tihx_wall, config.hihx2))

    z11 = (config.Dihx1) * config.Vihx1
    z22 = (config.mihx_wall * config.c_cu)
    z33 = (config.Dihx2) * config.Vihx2
    Zihx = np.array([[z11, 0, 0], [0, z22, 0], [0, 0, z33]])

    f1 = config.Qihx1_conv + mihx1_in_dot * (hihx1_in - xihx[0])
    f2 = -(config.Qihx1_conv + config.Qihx2_conv)
    f3 = config.Qihx2_conv + mihx2_out_dot * (hihx2_in - xihx[2])
    fihx_vec = np.array([f1, f2, f3])

    Zihx_inv = np.linalg.inv(Zihx)
    dxihxdt = np.dot(Zihx_inv, fihx_vec)

    config.dhihx1dt = dxihxdt[0]
    config.dTihx_walldt = dxihxdt[1]
    config.dhihx2dt = dxihxdt[2]

    xbuff1 = np.hstack((config.pbuff1, config.hbuff1, config.Tbuff1_wall))

    z11 = (- 1) * config.Vbuff1
    z12 = (config.Dbuff1) * config.Vbuff1
    z21 = config.dDdp_hbuff1 * config.Vbuff1
    z22 = config.dDdh_pbuff1 * config.Vbuff1
    z33 = (config.mbuff1_wall * config.c_cu)
    Zbuff1 = np.array([[z11, z12, 0], [z21, z22, 0], [0, 0, z33]])

    f1 = config.Qbuff1_conv + mbuff1_in_dot * (hbuff1_in - xbuff1[1])
    f2 = mbuff1_in_dot - mbuff1_out_dot
    f3 = -(config.Qbuff1_w_conv + config.Qbuff1_conv)
    fbuff1_vec = np.array([f1, f2, f3])

    Zbuff1_inv = np.linalg.inv(Zbuff1)
    dxbuff1dt = np.dot(Zbuff1_inv, fbuff1_vec)

    config.dpbuff1dt = dxbuff1dt[0]
    config.dhbuff1dt = dxbuff1dt[1]
    config.dTbuff1_walldt = dxbuff1dt[2]

    xft = np.hstack((config.pft, config.hft))

    z11 = (- 1) * config.Vft
    z12 = (config.Dft) * config.Vft
    z21 = config.dDdp_hft*config.Vft
    z22 = config.dDdh_pft*config.Vft
    Zft = np.array([[z11, z12],
                [z21, z22]])

    Zft_inv = np.linalg.inv(Zft)
    f1 = (mft_in_dot*hft_in - mft_l_dot*config.hft_lo - mft_v_dot*config.hft_vo)
    f2 = (mft_in_dot - mft_l_dot - mft_v_dot)
    fft_vec = np.array([f1, f2])

    Zft_inv = np.linalg.inv(Zft)
    dxftdt = np.dot(Zft_inv, fft_vec)
    
    config.dpftdt = dxftdt[0]
    config.dhftdt = dxftdt[1]