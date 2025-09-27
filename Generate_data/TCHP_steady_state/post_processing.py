import numpy as np
import math

from utils import (
	get_state,
)
import config
from config import CP
from funcs import (
    init_evaporator_post,
    init_gas_cooler_post,
    init_buffer1_post,
    init_internal_heat_exchanger_post,
    init_flash_tank_post,

    calc_evaporator,
    calc_gas_cooler,
    calc_buffer1,
    calc_internal_heat_exchanger,
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
def post_process(
    y,
    t,
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
    ):

    a_t = []
    a_pc = []
    a_Tw_out = []
    
    a_Qe_conv = []
    a_Qmpg_conv = []
    a_Pevap = []

    a_Qc_conv = []
    a_Qc_w_conv = []

    a_j = []

    a_Tc_out = []
    a_hc_out = []
    a_pe = []
    a_Te_out = []
    a_he_out = []

    a_me_dot = []
    a_mihx2_dot = []
    a_mbuff1_dot = []
    a_mbuff2_dot = []
    a_mc_dot = []
    a_mihx1_dot = []
    a_Pgc = []
    a_Pevap = []
    a_Pihx1 = []
    a_Pihx2 = []
    a_Pbuff1 = []
    a_Pbuff2 = []
    a_COP = []
    a_Pheat_out = []
    a_Pcoolers = []
    a_Pbuffers = []
    a_mc_in_dot = []
    a_mbuff1_in_dot = []
    a_pft = []
    a_hft = []
    a_mc_out_dot = []
    a_mft_in_dot = []
    a_mft_out_dot = []
    a_me_in_dot = []
    a_me_out_dot = []
    a_mbuff1_out_dot = []
    a_pbuff1 = []
    a_pbuff2 = []
    a_mft_dot = []
    a_hihx1_out = []
    a_hihx2_out = []
    a_Tmpg_out = []
    a_Prec = []
    a_Pcomb = []
    a_SH = []
    a_hc_in= []
    a_hTC2_in = []
    a_hbuff1_in = []
    a_Prec_total = []
    a_Hpev = []
    for k in range(len(t)):

        j = int(np.floor(t[k]))
        a_j.append(j)
        pbuff1_in = config.pbuff1
        pft_in = config.pft
        pc_in = config.pc
        pe_in = config.pe
        De_in = config.De[0]
        mw2_dot = 0.7 * mw_dot
        mw1_dot = 0.3 * mw_dot
        mw1_dot_1 = 0.5 * mw1_dot
        mw1_dot_2 = mw1_dot_1

        lpev = y[3 * (config.Ne + config.Nc) + 12, k]
        hpev = y[3 * (config.Ne + config.Nc) + 13, k]

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

        # print(t)
        me_in_dot = mft_l_dot
        mft_in_dot = mc_out_dot
        # mft_v_dot = mc_out_dot - me_in_dot

        mft_out_dot = mft_v_dot + mft_l_dot
        
        mbuff1_in_dot = mft_v_dot + me_out_dot
        mbuff1_out_dot = mc_in_dot

        mihx1_in_dot = mc_in_dot
        mihx1_out_dot = mihx1_in_dot

        mihx2_in_dot = me_out_dot
        mihx2_out_dot = mihx2_in_dot

        TTC1_out = model1_2_LR.predict(a1_T)[0] # model2_PR.predict(poly2_reg.transform(a1_T))[0]
        state = get_state(CP.PT_INPUTS, pbuff1_in, TTC1_out)
        hTC1_out = state.hmass()

        hbuff1_in = hTC1_out #(mft_v_dot * config.hft_v + me_out_dot * hTC1_out)/(mft_v_dot + me_out_dot)
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

        Pcooler1_pred = model1_3_LR.predict(a1_T)[0] # model3_PR.predict(poly3_reg.transform(a1_T))[0]
        Pcooler23_pred = model3_3_LR.predict(a3_T)[0] #model3_.predict(poly3_reg.transform(a3_T))[0]

        init_gas_cooler_post(y, k, pc_in)
        calc_gas_cooler(mc_in_dot, mc_out_dot, Dc_in, mw2_dot, Tw_in)

        init_evaporator_post(y, k, pe_in, he_in, config.pbuff1)
        calc_evaporator(me_in_dot, me_out_dot, De_in, mmpg_dot, Tmpg_in)
        
        init_buffer1_post(y, k, pbuff1_in, config.pe)
        init_internal_heat_exchanger_post(y, k, pihx1_in, pihx2_in, hihx1_in, hihx2_in)
        init_flash_tank_post(y, k, pft_in, config.pe)
        
        calc_buffer1(mbuff1_in_dot, mbuff1_out_dot, Dbuff1_in, mw1_dot_1, Tw_in)
        calc_internal_heat_exchanger(mihx1_in_dot, mihx1_out_dot, Dihx1_in, mihx2_in_dot, mihx2_out_dot, Dihx2_in)

        # if j < config.t0[-1]:
        #     if (np.floor(t[k + 1]) - np.floor(t[k]) > 0):
        a_Qe_conv.append(np.sum(config.Qe_conv))
        a_Qmpg_conv.append(np.sum(config.Qmpg_conv))
        a_Qc_conv.append(np.sum(config.Qc_conv))
        a_Qc_w_conv.append(np.sum(config.Qc_w_conv))

        a_pe.append(config.pe * 10**(-5))
        a_Te_out.append(config.Tihx2 - 273.15)
        a_pc.append(config.pc * 10**(-5))
        a_Tc_out.append(config.Tc[-1]- 273.15)
        a_he_out.append(config.he[-1])
        a_hc_out.append(config.hc[-1])

        a_me_dot.append(me_in_dot - me_out_dot)
        a_mihx2_dot.append(mihx2_in_dot - mihx2_out_dot)
        a_mbuff1_dot.append(mbuff1_in_dot - mbuff1_out_dot)
        a_mft_dot.append(mft_in_dot - mft_out_dot)
        a_mc_dot.append(mc_in_dot - mc_out_dot)
        a_mihx1_dot.append(mihx1_in_dot - mihx1_out_dot)
        a_t.append(t[k])
        # print(mw2_dot * config.cpw * (config.Tc_w[0] - Tw_in))
        a_Pgc.append(mc_in_dot * (hc_in - config.hc[-1])) #mw2_dot * config.cpw * (config.Tc_w[0] - Tw_in)) #
        a_Pevap.append(me_out_dot * (config.he[-1] - he_in)) #mmpg_dot * config.cp_mpg * (Tmpg_in - config.Tmpg[0])) #
        # a_Pihx1.append(mihx1_in_dot*(hihx1_in - config.hihx1[0]))
        # a_Pihx2.append(mihx2_in_dot*(config.hihx2[-1] - hihx2_in))
        a_Pbuff1.append(mbuff1_in_dot*(hbuff1_in - config.hbuff1))

        mCH4_dot = (0.0022 * omegab - 2.5965) * 0.657/60000
        Pcomb = mCH4_dot * 50e6

        Tfume_in = 28.4*(0.0022*omegab - 2.5965) + 267
        Trwo_target = (a_Pbuff1[-1] + Pcooler1_pred + Pcooler23_pred)/(config.cpw * mw1_dot) + Tw_in
        config.Trwo = Trwo_target #config.Trwo + 1/50 * (Trwo_target - config.Trwo)
        Tw_rec_in = (mw2_dot * config.Tc_w[0] + mw1_dot * config.Trwo)/mw_dot
        mg_dot = 18.125 * mCH4_dot
        
        Prec_pred = (-3.59 * mw_dot - 624.78 * mg_dot + 0.128 * Tw_rec_in - 33.54) * (Tfume_in - Tw_rec_in) #  (2005.8 * mg_dot - 3.87 * mw_dot + 0.083) * (Tfume_in - Tw_rec_in) #(477.74 * mg_dot + 1.25) * (Tfume_in - Tw_rec_in)

        Pheat_out_target = a_Pgc[-1] + a_Pbuff1[-1] + Pcooler1_pred + Pcooler23_pred + Prec_pred
        config.Pheating_out = Pheat_out_target #config.Pheating_out + 1/20 * (Pheat_out_target - config.Pheating_out)

        a_Tw_out.append(Tw_rec_in + (Prec_pred)/(mw_dot * config.cpw) - 273.15) #+ config.Pheating_out/(mw_dot * config.cpw) + Tw_in - 273.15)
        a_Tmpg_out.append(config.Tmpg[0] - 273.15) #- a_Pevap[-1]/(mmpg_dot * config.cpw) + Tmpg_in - 273.15)
        a_Pcoolers.append(Pcooler1_pred + Pcooler23_pred)
        a_Pbuffers.append(a_Pbuff1[-1])
        # print(Prec_pred)
        a_Pheat_out.append(config.Pheating_out)
        state = get_state(CP.PQ_INPUTS, config.pe, 1)
        a_SH.append(config.Tihx2 - state.T())
        a_COP.append(config.Pheating_out/Pcomb)
        a_mc_in_dot.append(mc_in_dot)
        a_mc_out_dot.append(mc_out_dot)
        a_mft_in_dot.append(mft_in_dot)
        a_mft_out_dot.append(mft_out_dot)
        a_me_in_dot.append(me_in_dot)
        a_me_out_dot.append(me_out_dot)
        a_mbuff1_in_dot.append(mbuff1_in_dot)
        a_mbuff1_out_dot.append(mbuff1_out_dot)
        a_pbuff1.append(config.pbuff1 * 10**(-5))
        a_pft.append(config.pft * 10**(-5))
        a_hft.append(config.hft)
        a_hihx1_out.append(config.hihx1)
        a_hihx2_out.append(config.hihx2)
        a_hc_in.append(hc_in)
        a_hTC2_in.append(config.hbuff1)
        a_hbuff1_in.append(hbuff1_in)

    return (
        a_t,
        a_Pgc,
        a_Pevap,
        a_Pihx1,
        a_Pihx2,
        a_Pbuff1,
        a_Pbuff2,
        Pcooler1_pred,
        Pcooler23_pred,
        a_me_dot,
        a_mihx2_dot,
        a_mbuff1_dot,
        a_mbuff2_dot,
        a_mc_dot,
        a_mihx1_dot,
        a_COP,
        a_Pheat_out,
        a_Pcoolers,
        a_Pbuffers,
        a_mc_in_dot,
        a_mbuff1_in_dot,
        a_pft,
        a_pc,
        a_pe,
        a_Tc_out,
        a_Te_out,
        a_hc_out,
        a_he_out,
        a_mc_out_dot,
        a_mft_in_dot,
        a_mft_out_dot,
        a_me_in_dot,
        a_me_out_dot,
        a_hft,
        a_mbuff1_out_dot,
        a_pbuff1,
        a_pbuff2,
        a_mft_dot,
        a_hihx1_out,
        a_hihx2_out,
        a_Tw_out,
        a_Tmpg_out,
        a_SH,
        a_Pcomb,
        a_Prec_total,
        a_Hpev,
        a_hc_in,
        a_hTC2_in,
        a_hbuff1_in
    )

